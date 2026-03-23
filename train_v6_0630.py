#!/usr/bin/env python3
"""PatchCore v6 — 0630 only training.

0630 비결함 29개 폴더로만 학습, 결함 폴더(160852) 1개 제외.
Tile 128x128, stride 128, full-res 1920x1200
Group 1 only (cam 1, 10)
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
TRAIN_DATE = "20250630"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 128
TILE_STRIDE = 128

TRIM_HEAD = 100
TRIM_TAIL = 100

DEFECT_FOLDER_PREFIX = "20250630160852"
SUBSAMPLE_STEP = 1  # No subsampling — use all images

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
def discover_0630_folders(spec_pattern):
    """Find all 0630 folders, separate defect from normal."""
    date_dir = NAS_ROOT / TRAIN_DATE
    normal_folders = []
    defect_folder = None

    if not date_dir.is_dir():
        print(f"ERROR: {date_dir} not found!")
        return [], None

    for sub in sorted(date_dir.iterdir()):
        if sub.is_dir() and spec_pattern in sub.name:
            if (sub / "camera_1").is_dir():
                if DEFECT_FOLDER_PREFIX in sub.name:
                    defect_folder = sub
                else:
                    normal_folders.append(sub)

    return normal_folders, defect_folder


# ===== FEATURE EXTRACTION =====
def extract_tile_features(folders, cam_ids, mirror_cam, extractor, positions, tile_masks):
    """Extract tile features from 0630 normal folders."""
    all_features = []
    all_keys = []
    total_images = 0
    total_tiles = 0

    print(f"\n    Training folders: {len(folders)}, subsample={SUBSAMPLE_STEP}")

    for folder in folders:
        mask = tile_masks.get(folder.name)
        if mask is None:
            mask = np.ones(len(positions), dtype=bool)
        valid_indices = np.where(mask)[0]
        folder_images = 0

        for cam_id in cam_ids:
            images = get_image_paths(folder, cam_id, SUBSAMPLE_STEP)
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

                tiles = []
                tile_idxs = []
                for t_idx in valid_indices:
                    tx, ty = positions[t_idx]
                    tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                    tiles.append(TRANSFORM(tile))
                    tile_idxs.append(t_idx)

                if not tiles:
                    continue

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
                folder_images += 1

        print(f"    {folder.name}: {folder_images} images")

    features = np.concatenate(all_features, axis=0)
    print(f"\n    Total: {total_images} images, {features.shape[0]} tiles ({features.shape[1]}-dim)")
    return features, all_keys, total_images


# ===== CORESET =====
def greedy_coreset_selection(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)

    print(f"    Coreset: {n} -> {target} (ratio={ratio})")
    device = "cuda"

    # Random projection for speed
    proj = torch.randn(d, proj_dim, device=device) / (proj_dim ** 0.5)
    feat_tensor = torch.from_numpy(features).to(device)
    feat_proj = feat_tensor @ proj

    # Initialize with random point
    selected = [np.random.randint(n)]
    min_dists = torch.full((n,), float('inf'), device=device)

    for i in range(1, target):
        last = feat_proj[selected[-1]].unsqueeze(0)
        dists = torch.cdist(feat_proj, last).squeeze(1)
        min_dists = torch.minimum(min_dists, dists)
        next_idx = torch.argmax(min_dists).item()
        selected.append(next_idx)

        if (i + 1) % 5000 == 0:
            print(f"      {i + 1}/{target}")

    selected = np.array(selected)
    return features[selected], selected


def score_tiles(tile_features, memory_bank):
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    scores = []
    bs = 256
    for i in range(0, len(tf), bs):
        dists = torch.cdist(tf[i:i+bs], mb)
        min_dists, _ = dists.min(dim=1)
        scores.append(min_dists.cpu().numpy())
    return np.concatenate(scores)


# ===== SELF-VALIDATION =====
def self_validation(features, keys, extractor, mad_k=3.5):
    n = len(features)
    print(f"    Features: {n}, computing coreset for scoring...")
    coreset, _ = greedy_coreset_selection(features)
    scores = score_tiles(features, coreset)

    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_std = 1.4826 * mad
    threshold = median + mad_k * mad_std
    n_reject = (scores > threshold).sum()
    max_reject = int(n * SELF_VAL_MAX_REJECT_PCT / 100)

    print(f"    Median={median:.4f}, MAD_std={mad_std:.4f}, threshold={threshold:.4f}")
    print(f"    Reject: {n_reject}/{n} (max allowed: {max_reject})")

    if n_reject > max_reject:
        print(f"    WARNING: too many rejects, capping at {max_reject}")
        sorted_idx = np.argsort(scores)
        keep_idx = sorted_idx[:n - max_reject]
    else:
        keep_idx = np.where(scores <= threshold)[0]

    features_clean = features[keep_idx]
    keys_clean = [keys[i] for i in keep_idx]
    print(f"    After self-val: {len(features_clean)} tiles")
    return features_clean, keys_clean, scores, threshold


# ===== DEFECT INFERENCE =====
def run_defect_inference(defect_folder, memory_bank, extractor, positions, output_dir, threshold):
    results = []
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))

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
            if mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            tiles = []
            tile_idxs = []
            for t_idx, (tx, ty) in enumerate(positions):
                tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                arr = np.array(tile)
                if arr.mean() < MASK_BRIGHTNESS_THRESHOLD:
                    continue
                tiles.append(TRANSFORM(tile))
                tile_idxs.append(t_idx)

            if not tiles:
                continue

            all_feats = []
            for bs_start in range(0, len(tiles), BATCH_SIZE):
                batch = torch.stack(tiles[bs_start:bs_start + BATCH_SIZE])
                feats = extractor(batch).cpu().numpy()
                all_feats.append(feats)
            feats = np.concatenate(all_feats, axis=0)
            scores = score_tiles(feats, memory_bank)

            max_score = float(scores.max())
            results.append({
                "file": img_path.name,
                "cam": cam_id,
                "max_score": max_score,
                "mean_score": float(scores.mean()),
                "anomaly": max_score > threshold,
            })

    return results


# ===== MAIN =====
def main():
    t0 = time.time()
    print(f"PatchCore v6 — 0630 ONLY Training")
    print(f"Tile {TILE_SIZE}x{TILE_SIZE}, stride {TILE_STRIDE}, full-res {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Group {GROUP_ID}: cam {CAM_IDS}, mirror cam {MIRROR_CAM}")

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles per image: {len(positions)} ({n_tiles_x}x{n_tiles_y})")

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}_0630only"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backbone
    print("\n[0] Loading backbone...")
    extractor = TileFeatureExtractor("cuda")

    # Discover 0630 data
    print("\n[1] Scanning 0630 folders...")
    normal_folders, defect_folder = discover_0630_folders(TARGET_SPEC)
    print(f"  Normal folders: {len(normal_folders)}")
    for f in normal_folders:
        print(f"    {f.name}")
    if defect_folder:
        print(f"  Defect folder (EXCLUDED): {defect_folder.name}")

    # Tile masks
    print("\n[2] Computing tile masks...")
    tile_masks = {}
    for folder in normal_folders[:5]:
        mask = compute_tile_mask(folder, CAM_IDS, positions)
        tile_masks[folder.name] = mask
        n_valid = mask.sum()
        print(f"    {folder.name}: {n_valid}/{len(positions)} valid tiles")

    # Feature extraction
    print("\n[3] Extracting tile features (0630 only, no subsampling)...")
    features, keys, total_images = extract_tile_features(
        normal_folders, CAM_IDS, MIRROR_CAM, extractor, positions, tile_masks)

    # Self-validation
    print("\n[4] Self-validation...")
    features_clean, keys_clean, sv_scores, sv_threshold = self_validation(
        features, keys, extractor, SELF_VAL_MAD_K)

    # Save self-val histogram
    plt.figure(figsize=(12, 5))
    plt.hist(sv_scores, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(sv_threshold, color='red', linestyle='--', label=f'Threshold: {sv_threshold:.4f}')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title(f"Self-Validation Score Distribution (0630 only)")
    plt.legend()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100)
    plt.close()

    # Final coreset
    print("\n[5] Final coreset selection...")
    memory_bank, _ = greedy_coreset_selection(features_clean)
    print(f"    Final memory bank: {memory_bank.shape}")
    np.save(output_dir / "memory_bank.npy", memory_bank)

    # Compute threshold
    clean_scores = score_tiles(features_clean, memory_bank)
    median = np.median(clean_scores)
    mad = np.median(np.abs(clean_scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    print(f"    Threshold (MAD k={SELF_VAL_MAD_K}): {threshold:.4f}")

    # Count images per folder for metadata
    folder_image_counts = {}
    for folder in normal_folders:
        count = 0
        for cam_id in CAM_IDS:
            count += len(get_image_paths(folder, cam_id, SUBSAMPLE_STEP))
        folder_image_counts[folder.name] = count

    # Save metadata
    meta = {
        "version": "v6-tile-0630only",
        "group_id": GROUP_ID,
        "cam_ids": CAM_IDS,
        "mirror_cam": MIRROR_CAM,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "image_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "tiles_per_image": len(positions),
        "tiles_grid": [n_tiles_x, n_tiles_y],
        "training_date": TRAIN_DATE,
        "training_folders": [f.name for f in normal_folders],
        "excluded_defect_folder": defect_folder.name if defect_folder else None,
        "folder_image_counts": folder_image_counts,
        "total_images": total_images,
        "total_tiles_extracted": len(features),
        "tiles_after_selfval": len(features_clean),
        "memory_bank_shape": list(memory_bank.shape),
        "coreset_ratio": CORESET_RATIO,
        "subsample_step": SUBSAMPLE_STEP,
        "trim_head": TRIM_HEAD,
        "trim_tail": TRIM_TAIL,
        "self_val_rounds": SELF_VAL_ROUNDS,
        "self_val_mad_k": SELF_VAL_MAD_K,
        "threshold_mad": float(threshold),
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
        print(f"    Defect images: {len(results)}")
        print(f"    Scores: {min(all_scores):.4f} ~ {max(all_scores):.4f} (median={np.median(all_scores):.4f})")
        print(f"    Anomaly: {n_anom}/{len(results)} ({100*n_anom/max(len(results),1):.1f}%)")

        with open(output_dir / "defect_results.json", "w") as f:
            json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed / 60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
