#!/usr/bin/env python3
"""PatchCore v6 — Tile-based + date-balanced + self-validation.

Tile 128x128, stride 128, full-res 1920x1200
Group 1 only (cam 1, 10)
Self-validation 1 round (MAD k=3.5)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, time, json, re, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import defaultdict

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v6")
TARGET_SPEC = "596x199"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 128
TILE_STRIDE = 128

TRIM_HEAD = 100
TRIM_TAIL = 100

# Date-balanced sampling
DEFECT_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "20250630160852"
SUBSAMPLE_DEFECT_DATE = 15
SUBSAMPLE_NORMAL = 3

# Coreset
CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128

# Self-validation
SELF_VAL_ROUNDS = 1
SELF_VAL_MAD_K = 3.5
SELF_VAL_MAX_REJECT_PCT = 5.0

# Tile mask
MASK_SAMPLE_COUNT = 20
MASK_BRIGHTNESS_THRESHOLD = 30

# GPU
BATCH_SIZE = 256
GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== UTILS =====
def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def tile_positions(img_w, img_h, tile_size, stride):
    positions = []
    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            positions.append((x, y))
    return positions


def get_image_paths(folder, cam_id, subsample_step=1):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=natural_sort_key
    )
    if len(images) <= TRIM_HEAD + TRIM_TAIL:
        return []
    images = images[TRIM_HEAD:len(images) - TRIM_TAIL]
    if subsample_step > 1:
        images = images[::subsample_step]
    return images


# ===== BACKBONE =====
class TileFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        local_weights = cache_dir / "wide_resnet50_2-95faca4d.pth"
        if local_weights.exists():
            print(f"  Loading WideResNet50 from local cache")
            backbone = wide_resnet50_2(weights=None)
            state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        else:
            print("  Loading WideResNet50 (downloading)...")
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        """Input: (B, 3, 128, 128) -> Output: (B, 1536)"""
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = self.layer1(x)
            f2 = self.layer2(h)
            f3 = self.layer3(f2)
            f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
            features = torch.cat([f2, f3_up], dim=1)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features.float()


# ===== TILE MASK =====
def compute_tile_mask(folder, cam_ids, positions):
    """Compute brightness-based mask for tiles."""
    num_tiles = len(positions)
    brightness_accum = np.zeros(num_tiles, dtype=np.float64)
    count = 0

    for cam_id in cam_ids:
        images = get_image_paths(folder, cam_id)[:MASK_SAMPLE_COUNT]
        for img_path in images:
            try:
                img = Image.open(img_path).convert("L")
            except:
                continue
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                continue
            img_arr = np.array(img)
            for t_idx, (tx, ty) in enumerate(positions):
                tile = img_arr[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                brightness_accum[t_idx] += tile.mean()
            count += 1

    if count == 0:
        return np.ones(num_tiles, dtype=bool)
    avg_brightness = brightness_accum / count
    return avg_brightness >= MASK_BRIGHTNESS_THRESHOLD


# ===== DATA DISCOVERY =====
def discover_spec_folders(spec_pattern):
    date_folders = defaultdict(list)
    defect_folder = None
    for entry in sorted(NAS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if re.match(r'^\d{8}$', entry.name):
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and spec_pattern in sub.name:
                        if (sub / "camera_1").is_dir():
                            if DEFECT_FOLDER_PREFIX in sub.name:
                                defect_folder = sub
                            else:
                                date_folders[entry.name].append(sub)
            except PermissionError:
                continue
    return date_folders, defect_folder


# ===== FEATURE EXTRACTION =====
def extract_tile_features(date_folders, cam_ids, mirror_cam, extractor, positions, tile_masks):
    """Extract tile features from all training images with date-balanced sampling."""
    all_features = []
    all_keys = []  # (folder_name, cam_id, img_name, tile_idx)
    total_images = 0
    total_tiles = 0

    for date_str, folders in sorted(date_folders.items()):
        if date_str == DEFECT_DATE:
            print(f"\n    {date_str} (DEFECT) — SKIPPED (not used for training)")
            continue
        subsample = SUBSAMPLE_NORMAL
        label = f"{date_str} (normal - dense)"

        print(f"\n    {label}: {len(folders)} folders, subsample={subsample}")
        date_images = 0

        for folder in folders:
            mask = tile_masks.get(folder.name)
            if mask is None:
                mask = np.ones(len(positions), dtype=bool)
            valid_indices = np.where(mask)[0]

            for cam_id in cam_ids:
                images = get_image_paths(folder, cam_id, subsample)
                mirror = (cam_id == mirror_cam)

                for img_path in images:
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except:
                        continue
                    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                        continue
                    if mirror:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    # Extract valid tiles
                    tiles = []
                    tile_idxs = []
                    for t_idx in valid_indices:
                        tx, ty = positions[t_idx]
                        tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                        tiles.append(TRANSFORM(tile))
                        tile_idxs.append(t_idx)

                    if not tiles:
                        continue

                    # Batch inference
                    for batch_start in range(0, len(tiles), BATCH_SIZE):
                        batch_tiles = tiles[batch_start:batch_start + BATCH_SIZE]
                        batch_idxs = tile_idxs[batch_start:batch_start + BATCH_SIZE]
                        batch_tensor = torch.stack(batch_tiles)
                        feats = extractor(batch_tensor).cpu().numpy()
                        all_features.append(feats)
                        for i, t_idx in enumerate(batch_idxs):
                            all_keys.append((folder.name, cam_id, img_path.name, t_idx))

                    total_images += 1
                    total_tiles += len(tiles)
                    date_images += 1

            if date_images % 500 == 0 and date_images > 0:
                print(f"      {date_images} images, {total_tiles} tiles")

        print(f"    -> {date_str}: {date_images} images")

    features = np.concatenate(all_features, axis=0)
    print(f"\n    Total: {total_images} images, {features.shape[0]} tiles ({features.shape[1]}-dim)")
    return features, all_keys, total_images


# ===== CORESET =====
def greedy_coreset_selection(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj_matrix = rng.randn(d, proj_dim).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

    device = "cuda"
    proj_t = torch.from_numpy(proj_matrix).to(device)
    feat_t = torch.from_numpy(features.astype(np.float32)).to(device)
    projected = feat_t @ proj_t

    selected = [rng.randint(n)]
    min_distances = torch.full((n,), float('inf'), device=device)

    for _ in tqdm(range(target - 1), desc="Coreset", leave=True, unit="pt"):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, dists)
        next_idx = torch.argmax(min_distances).item()
        selected.append(next_idx)

    indices = np.array(selected)
    return features[indices], indices


# ===== SELF-VALIDATION =====
def self_validation(features, keys, extractor, mad_k=SELF_VAL_MAD_K):
    """One round of self-validation: remove tiles exceeding MAD threshold."""
    print(f"\n  [Self-Validation] {len(features)} tiles, MAD k={mad_k}")

    # Fit coreset
    memory_bank, _ = greedy_coreset_selection(features)
    print(f"    Coreset: {memory_bank.shape[0]} / {features.shape[0]}")

    # Score all tiles
    scores = score_tiles(features, memory_bank)

    # MAD threshold
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_std = 1.4826 * mad
    threshold = median + mad_k * mad_std

    # Safety cap
    pct_threshold = np.percentile(scores, 100 - SELF_VAL_MAX_REJECT_PCT)
    effective_threshold = max(threshold, pct_threshold)

    reject_mask = scores >= effective_threshold
    n_reject = reject_mask.sum()
    reject_pct = 100 * n_reject / len(scores)

    print(f"    Median={median:.4f}, MAD_std={mad_std:.4f}, Threshold={effective_threshold:.4f}")
    print(f"    Rejecting {n_reject} tiles ({reject_pct:.2f}%)")

    # Filter
    keep_mask = ~reject_mask
    features_clean = features[keep_mask]
    keys_clean = [k for k, keep in zip(keys, keep_mask) if keep]

    return features_clean, keys_clean, scores, effective_threshold


def score_tiles(features, memory_bank, batch_size=4096):
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(memory_bank).cuda()
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i + batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        min_d, _ = dists.min(dim=1)
        scores[i:i + batch_size] = min_d.cpu().numpy()
    return scores


# ===== INFERENCE ON DEFECT FOLDER =====
def run_defect_inference(defect_folder, memory_bank, extractor, positions, output_dir, threshold):
    """Score defect folder images and generate heatmaps."""
    print(f"\n  [Inference] {defect_folder.name}")
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    results = []

    heatmap_dir = output_dir / "heatmaps" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in CAM_IDS:
        images = get_image_paths(defect_folder, cam_id)
        mirror = (cam_id == MIRROR_CAM)

        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                continue

            img_display = img.copy()
            if mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Extract all tiles
            tiles = []
            for tx, ty in positions:
                tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                tiles.append(TRANSFORM(tile))

            # Batch inference
            all_feats = []
            for batch_start in range(0, len(tiles), BATCH_SIZE):
                batch = torch.stack(tiles[batch_start:batch_start + BATCH_SIZE])
                feats = extractor(batch).cpu().numpy()
                all_feats.append(feats)
            features = np.concatenate(all_feats, axis=0)

            # Score
            scores = score_tiles(features, memory_bank)
            score_grid = scores.reshape(n_tiles_y, n_tiles_x)
            max_score = float(scores.max())
            mean_score = float(scores.mean())
            is_anomaly = max_score > threshold

            # Heatmap
            if mirror:
                score_grid = np.fliplr(score_grid)
            score_tensor = torch.from_numpy(score_grid).unsqueeze(0).unsqueeze(0).float()
            heatmap_full = F.interpolate(
                score_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                mode='bilinear', align_corners=False
            ).squeeze().numpy()
            heatmap_full = gaussian_filter(heatmap_full, sigma=4)

            # Draw
            heatmap_path = heatmap_dir / f"heatmap_{img_path.stem}.png"
            _draw_heatmap(img_display, heatmap_full, max_score, threshold, heatmap_path, img_path.name)

            results.append({
                "file": img_path.name, "cam": cam_id,
                "max_score": max_score, "mean_score": mean_score,
                "anomaly": is_anomaly
            })

    n_anom = sum(1 for r in results if r["anomaly"])
    print(f"    {len(results)} images, {n_anom} anomaly")

    # Save results
    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _draw_heatmap(img_display, heatmap, max_score, threshold, output_path, title):
    img_arr = np.array(img_display)
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    axes[0].imshow(img_arr)
    axes[0].set_title(f"Original: {title}", fontsize=14)
    axes[0].axis("off")

    vmax = max(threshold * 2, np.percentile(heatmap, 99.5))
    im1 = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Score (max={max_score:.2f})", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(img_arr)
    overlay_min = threshold * 0.7
    masked = np.ma.masked_where(heatmap < overlay_min, heatmap)
    im2 = axes[2].imshow(masked, cmap='jet', alpha=0.65,
                          vmin=overlay_min, vmax=max(vmax, threshold * 2))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    if max_score > threshold:
        axes[2].set_title(f"ANOMALY (score={max_score:.2f})", fontsize=14, color='red', fontweight='bold')
    else:
        axes[2].set_title(f"NORMAL (score={max_score:.2f})", fontsize=14, color='green', fontweight='bold')
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v6-TILE | {TARGET_SPEC}/group_{GROUP_ID} | tile={TILE_SIZE} | {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


# ===== MAIN =====
def main():
    t0 = time.time()
    print(f"PatchCore v6 — Tile {TILE_SIZE}x{TILE_SIZE}, stride {TILE_STRIDE}, full-res {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Group {GROUP_ID}: cam {CAM_IDS}, mirror cam {MIRROR_CAM}")

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles per image: {len(positions)} ({n_tiles_x}x{n_tiles_y})")

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backbone
    print("\n[0] Loading backbone...")
    extractor = TileFeatureExtractor("cuda")

    # Discover data
    print("\n[1] Scanning NAS...")
    date_folders, defect_folder = discover_spec_folders(TARGET_SPEC)
    print(f"  Dates: {sorted(date_folders.keys())}")
    for d, fs in sorted(date_folders.items()):
        print(f"    {d}: {len(fs)} folders")
    if defect_folder:
        print(f"  Defect folder: {defect_folder.name}")

    # Tile masks (sample from each folder)
    print("\n[2] Computing tile masks...")
    tile_masks = {}
    sample_folders = []
    for date_str, folders in sorted(date_folders.items()):
        sample_folders.extend(folders[:3])
    for folder in sample_folders[:5]:
        mask = compute_tile_mask(folder, CAM_IDS, positions)
        tile_masks[folder.name] = mask
        n_valid = mask.sum()
        print(f"    {folder.name}: {n_valid}/{len(positions)} valid tiles")

    # Feature extraction
    print("\n[3] Extracting tile features (date-balanced)...")
    features, keys, total_images = extract_tile_features(
        date_folders, CAM_IDS, MIRROR_CAM, extractor, positions, tile_masks)

    # Self-validation (1 round)
    print("\n[4] Self-validation...")
    features_clean, keys_clean, sv_scores, sv_threshold = self_validation(
        features, keys, extractor, SELF_VAL_MAD_K)

    # Save self-val histogram
    plt.figure(figsize=(12, 5))
    plt.hist(sv_scores, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(sv_threshold, color='red', linestyle='--', label=f'Threshold: {sv_threshold:.4f}')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title(f"Self-Validation Score Distribution (Tile {TILE_SIZE}x{TILE_SIZE})")
    plt.legend()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100)
    plt.close()

    # Final coreset
    print("\n[5] Final coreset selection...")
    memory_bank, _ = greedy_coreset_selection(features_clean)
    print(f"    Final memory bank: {memory_bank.shape}")
    np.save(output_dir / "memory_bank.npy", memory_bank)

    # Compute threshold from clean features
    clean_scores = score_tiles(features_clean, memory_bank)
    median = np.median(clean_scores)
    mad = np.median(np.abs(clean_scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    print(f"    Threshold (MAD k={SELF_VAL_MAD_K}): {threshold:.4f}")

    # Save metadata
    meta = {
        "version": "v6-tile",
        "group_id": GROUP_ID,
        "cam_ids": CAM_IDS,
        "mirror_cam": MIRROR_CAM,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "image_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "tiles_per_image": len(positions),
        "tiles_grid": [n_tiles_x, n_tiles_y],
        "total_images": total_images,
        "total_tiles_extracted": len(features),
        "tiles_after_selfval": len(features_clean),
        "memory_bank_shape": list(memory_bank.shape),
        "coreset_ratio": CORESET_RATIO,
        "self_val_rounds": SELF_VAL_ROUNDS,
        "self_val_mad_k": SELF_VAL_MAD_K,
        "threshold_mad": threshold,
        "score_median": float(median),
        "score_mad_std": float(mad_std),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Inference on defect folder
    if defect_folder:
        print("\n[6] Defect folder inference...")
        results = run_defect_inference(defect_folder, memory_bank, extractor, positions, output_dir, threshold)

        all_scores = [r["max_score"] for r in results]
        n_anom = sum(1 for r in results if r["anomaly"])
        print(f"    Scores: {min(all_scores):.2f} ~ {max(all_scores):.2f}")
        print(f"    Anomaly: {n_anom}/{len(results)}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed / 60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
