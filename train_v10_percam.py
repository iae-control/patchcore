#!/usr/bin/env python3
"""PatchCore v9 — Per-camera models with 192x300 tiling.

Tile 192x300, 10 cols x 4 rows = 40 tiles/image
Training: top 2 rows only (20 tiles/image)
Inference: all 4 rows (40 tiles/image)
One model per camera (10 models), no mirror flipping
Date-balanced sampling, trim 100 head/tail, exclude defect folder only (include 0630 normal)
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
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v10_percam")
TARGET_SPEC = "596x199"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_W = 192
TILE_H = 300
N_COLS = IMAGE_WIDTH // TILE_W   # 10
N_ROWS = IMAGE_HEIGHT // TILE_H  # 4
TRAIN_ROWS = 2  # top 2 rows only for training
TILES_PER_IMAGE_TRAIN = N_COLS * TRAIN_ROWS  # 20
TILES_PER_IMAGE_INFER = N_COLS * N_ROWS      # 40

TRIM_HEAD = 100
TRIM_TAIL = 100

# Date-balanced sampling
DEFECT_DATES = {"20250630"}
DEFECT_FOLDER_PREFIX = "20250630160852"
SUBSAMPLE_NORMAL = 3  # use every 3rd image for normal dates

# Coreset
CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128

# Self-validation
SELF_VAL_MAD_K = 3.5
SELF_VAL_MAX_REJECT_PCT = 5.0

# Tile mask (skip very dark tiles)
MASK_SAMPLE_COUNT = 20
MASK_BRIGHTNESS_THRESHOLD = 30

# GPU
BATCH_SIZE = 64  # smaller batch because tiles are larger (192x300)

# Which cameras to train (all 10)
CAMERA_IDS = list(range(1, 11))

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== UTILS =====
def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def tile_positions(n_cols, n_rows, tile_w, tile_h):
    """Returns list of (x, y, col, row) for each tile."""
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tile_w
            y = row * tile_h
            positions.append((x, y, col, row))
    return positions


def get_image_paths(folder, cam_id, subsample_step=1):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')],
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
            print("  Downloading WideResNet50...")
            from torchvision.models import Wide_ResNet50_2_Weights
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        """Input: (B, 3, H, W) -> Output: (B, 1536)
        Works with any input size thanks to adaptive_avg_pool2d.
        For 192x300 input: layer2 output ~24x38, layer3 ~12x19.
        """
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = self.layer1(x)
            f2 = self.layer2(h)
            f3 = self.layer3(f2)
            f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
            features = torch.cat([f2, f3_up], dim=1)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features.float()


# ===== DATA DISCOVERY =====
def discover_spec_folders(spec_pattern):
    """Find all folders for a spec, separated by date."""
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
                            elif entry.name in DEFECT_DATES:
                                # Include non-defect folders from defect dates
                                date_folders[entry.name].append(sub)
                            else:
                                date_folders[entry.name].append(sub)
            except PermissionError:
                continue

    return date_folders, defect_folder


# ===== TILE MASK =====
def compute_tile_mask(sample_folders, cam_id, positions):
    """Compute brightness-based mask for tiles of a specific camera."""
    num_tiles = len(positions)
    brightness_accum = np.zeros(num_tiles, dtype=np.float64)
    count = 0

    for folder in sample_folders:
        images = get_image_paths(folder, cam_id)[:MASK_SAMPLE_COUNT]
        for img_path in images:
            try:
                img = Image.open(img_path).convert("L")
            except:
                continue
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                continue
            img_arr = np.array(img)
            for t_idx, (tx, ty, col, row) in enumerate(positions):
                tile = img_arr[ty:ty + TILE_H, tx:tx + TILE_W]
                brightness_accum[t_idx] += tile.mean()
            count += 1

    if count == 0:
        return np.ones(num_tiles, dtype=bool)

    avg_brightness = brightness_accum / count
    return avg_brightness >= MASK_BRIGHTNESS_THRESHOLD


# ===== FEATURE EXTRACTION =====
def extract_features_for_camera(date_folders, cam_id, extractor, train_positions, tile_mask):
    """Extract tile features for a single camera from training data."""
    valid_indices = np.where(tile_mask[:len(train_positions)])[0]
    if len(valid_indices) == 0:
        print(f"    WARNING: no valid tiles for camera {cam_id}")
        return np.array([]), [], 0

    all_features = []
    all_keys = []
    total_images = 0

    for date_str, folders in sorted(date_folders.items()):
        subsample = SUBSAMPLE_NORMAL
        date_images = 0

        for folder in folders:
            images = get_image_paths(folder, cam_id, subsample)

            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                except:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue

                # Extract valid training tiles (top 2 rows only)
                tiles = []
                tile_idxs = []
                for t_idx in valid_indices:
                    tx, ty, col, row = train_positions[t_idx]
                    tile = img.crop((tx, ty, tx + TILE_W, ty + TILE_H))
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
                date_images += 1

        if date_images > 0:
            print(f"      {date_str}: {date_images} images")

    if not all_features:
        return np.array([]), [], 0

    features = np.concatenate(all_features, axis=0)
    print(f"      Total: {total_images} images, {features.shape[0]} tiles")
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

    for i in range(target - 1):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, dists)
        next_idx = torch.argmax(min_distances).item()
        selected.append(next_idx)
        if (i + 1) % 1000 == 0:
            print(f"        coreset {i+1}/{target}")

    indices = np.array(selected)
    return features[indices], indices


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


# ===== SELF-VALIDATION =====
def self_validation(features, keys, mad_k=SELF_VAL_MAD_K):
    n = len(features)
    print(f"      Self-val: {n} tiles")
    coreset, _ = greedy_coreset_selection(features)
    scores = score_tiles(features, coreset)

    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_std = 1.4826 * mad
    threshold = median + mad_k * mad_std

    n_reject = (scores > threshold).sum()
    max_reject = int(n * SELF_VAL_MAX_REJECT_PCT / 100)

    if n_reject > max_reject:
        sorted_idx = np.argsort(scores)
        keep_idx = sorted_idx[:n - max_reject]
        n_reject = max_reject
    else:
        keep_idx = np.where(scores <= threshold)[0]

    print(f"      Rejected: {n_reject}/{n} ({100*n_reject/n:.1f}%)")
    features_clean = features[keep_idx]
    keys_clean = [keys[i] for i in keep_idx]
    return features_clean, keys_clean, scores, threshold


# ===== POSITION STATS =====
def compute_position_stats(date_folders, cam_id, memory_bank, extractor, all_positions):
    """Compute per-position mean/std of anomaly scores for z-score normalization.
    Uses ALL positions (all 4 rows) for calibration.
    """
    n_positions = len(all_positions)
    pos_scores = defaultdict(list)

    # Sample some calibration images
    cal_images = []
    for date_str, folders in sorted(date_folders.items()):
        for folder in folders:
            imgs = get_image_paths(folder, cam_id, subsample_step=5)
            cal_images.extend(imgs)

    # Cap at ~300 images for speed
    if len(cal_images) > 300:
        step = len(cal_images) // 300
        cal_images = cal_images[::step][:300]

    print(f"      Calibrating pos_stats: {len(cal_images)} images")

    for idx, img_path in enumerate(cal_images):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            continue

        tiles = []
        tile_idxs = []
        for t_idx, (tx, ty, col, row) in enumerate(all_positions):
            tile = img.crop((tx, ty, tx + TILE_W, ty + TILE_H))
            tiles.append(TRANSFORM(tile))
            tile_idxs.append(t_idx)

        if not tiles:
            continue

        all_feats = []
        for bs in range(0, len(tiles), BATCH_SIZE):
            batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
            feats = extractor(batch).cpu().numpy()
            all_feats.append(feats)
        feats = np.concatenate(all_feats, axis=0)
        scores = score_tiles(feats, memory_bank)

        for i, t_idx in enumerate(tile_idxs):
            pos_scores[t_idx].append(scores[i])

        if (idx + 1) % 100 == 0:
            print(f"        pos_stats: {idx+1}/{len(cal_images)}")

    # Compute mean/std per position
    pos_mean = np.zeros(n_positions, dtype=np.float32)
    pos_std = np.ones(n_positions, dtype=np.float32)  # default std=1 to avoid div by 0
    for t_idx in range(n_positions):
        if t_idx in pos_scores and len(pos_scores[t_idx]) >= 10:
            arr = np.array(pos_scores[t_idx])
            pos_mean[t_idx] = arr.mean()
            pos_std[t_idx] = max(arr.std(), 0.01)  # floor at 0.01
        elif t_idx in pos_scores:
            arr = np.array(pos_scores[t_idx])
            pos_mean[t_idx] = arr.mean()
            pos_std[t_idx] = max(arr.std(), 0.1)

    return pos_mean, pos_std


# ===== DEFECT INFERENCE =====
def run_defect_inference(defect_folder, cam_id, memory_bank, extractor,
                         all_positions, pos_mean, pos_std, output_dir):
    """Score defect folder for a single camera using z-score."""
    images = get_image_paths(defect_folder, cam_id)
    if not images:
        return []

    results = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            continue

        # Extract all tiles (4 rows)
        tiles = []
        tile_idxs = []
        for t_idx, (tx, ty, col, row) in enumerate(all_positions):
            tile = img.crop((tx, ty, tx + TILE_W, ty + TILE_H))
            tiles.append(TRANSFORM(tile))
            tile_idxs.append(t_idx)

        if not tiles:
            continue

        all_feats = []
        for bs in range(0, len(tiles), BATCH_SIZE):
            batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
            feats = extractor(batch).cpu().numpy()
            all_feats.append(feats)
        feats = np.concatenate(all_feats, axis=0)
        raw_scores = score_tiles(feats, memory_bank)

        # Z-score normalization
        z_scores = (raw_scores - pos_mean[tile_idxs]) / pos_std[tile_idxs]

        results.append({
            "file": img_path.name,
            "cam": cam_id,
            "max_raw": float(raw_scores.max()),
            "mean_raw": float(raw_scores.mean()),
            "max_z": float(z_scores.max()),
            "mean_z": float(z_scores.mean()),
            "anomaly_count": int((z_scores > 2.5).sum()),
            "n_tiles": len(tile_idxs),
        })

    return results


# ===== MAIN =====
def main():
    t0 = time.time()
    print(f"PatchCore v9 — Per-Camera 192x300 Tiling")
    print(f"Tile {TILE_W}x{TILE_H}, {N_COLS} cols x {N_ROWS} rows = {TILES_PER_IMAGE_INFER} tiles/image")
    print(f"Training: top {TRAIN_ROWS} rows = {TILES_PER_IMAGE_TRAIN} tiles/image")
    print(f"Cameras: {CAMERA_IDS}")
    print(f"Target spec: {TARGET_SPEC}")

    # All tile positions (4 rows)
    all_positions = tile_positions(N_COLS, N_ROWS, TILE_W, TILE_H)
    # Training positions (top 2 rows only)
    train_positions = [p for p in all_positions if p[3] < TRAIN_ROWS]

    print(f"All positions: {len(all_positions)}, Train positions: {len(train_positions)}")

    # Backbone (shared across all cameras)
    print("\n[0] Loading backbone...")
    extractor = TileFeatureExtractor("cuda")

    # Discover data
    print("\n[1] Scanning NAS...")
    date_folders, defect_folder = discover_spec_folders(TARGET_SPEC)
    print(f"  Training dates: {sorted(date_folders.keys())}")
    total_folders = sum(len(fs) for fs in date_folders.values())
    print(f"  Total training folders: {total_folders}")
    if defect_folder:
        print(f"  Defect folder: {defect_folder.name}")

    # Sample folders for tile mask computation
    sample_folders = []
    for date_str, folders in sorted(date_folders.items()):
        sample_folders.extend(folders[:2])
    sample_folders = sample_folders[:10]

    # Train each camera
    all_results = {}
    for cam_id in CAMERA_IDS:
        cam_t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  CAMERA {cam_id}")
        print(f"{'='*70}")

        cam_dir = OUTPUT_DIR / TARGET_SPEC / f"camera_{cam_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Tile mask for this camera
        print(f"    [2] Computing tile mask...")
        tile_mask = compute_tile_mask(sample_folders, cam_id, all_positions)
        n_valid_all = tile_mask.sum()
        n_valid_train = tile_mask[:len(train_positions)].sum()
        print(f"      Valid tiles: {n_valid_train}/{len(train_positions)} train, {n_valid_all}/{len(all_positions)} total")

        # Feature extraction
        print(f"    [3] Extracting features (train rows only)...")
        features, keys, n_images = extract_features_for_camera(
            date_folders, cam_id, extractor, train_positions, tile_mask)

        if len(features) == 0:
            print(f"    SKIPPED: no features extracted")
            continue

        # Self-validation
        print(f"    [4] Self-validation...")
        features_clean, keys_clean, sv_scores, sv_threshold = self_validation(features, keys)

        # Save self-val histogram
        plt.figure(figsize=(10, 4))
        plt.hist(sv_scores, bins=80, edgecolor='black', alpha=0.7)
        plt.axvline(sv_threshold, color='red', linestyle='--', label=f'Threshold: {sv_threshold:.4f}')
        plt.title(f"Camera {cam_id} — Self-Validation")
        plt.legend()
        plt.savefig(cam_dir / "self_val_hist.png", dpi=100)
        plt.close()

        # Final coreset
        print(f"    [5] Final coreset...")
        memory_bank, _ = greedy_coreset_selection(features_clean)
        print(f"      Memory bank: {memory_bank.shape}")
        np.save(cam_dir / "memory_bank.npy", memory_bank)

        # Position stats (all 4 rows)
        print(f"    [6] Computing position stats (all rows)...")
        pos_mean, pos_std = compute_position_stats(
            date_folders, cam_id, memory_bank, extractor, all_positions)
        np.save(cam_dir / "pos_mean.npy", pos_mean)
        np.save(cam_dir / "pos_std.npy", pos_std)

        # Compute threshold from clean scores
        clean_scores = score_tiles(features_clean, memory_bank)
        median = float(np.median(clean_scores))
        mad = float(np.median(np.abs(clean_scores - median)))
        mad_std = 1.4826 * mad
        threshold = median + SELF_VAL_MAD_K * mad_std

        # Save metadata
        meta = {
            "version": "v9-percam",
            "camera_id": cam_id,
            "spec": TARGET_SPEC,
            "tile_size": [TILE_W, TILE_H],
            "grid": [N_COLS, N_ROWS],
            "train_rows": TRAIN_ROWS,
            "tiles_per_image_train": TILES_PER_IMAGE_TRAIN,
            "tiles_per_image_infer": TILES_PER_IMAGE_INFER,
            "image_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "n_training_images": n_images,
            "n_tiles_extracted": len(features),
            "n_tiles_after_selfval": len(features_clean),
            "memory_bank_shape": list(memory_bank.shape),
            "coreset_ratio": CORESET_RATIO,
            "self_val_mad_k": SELF_VAL_MAD_K,
            "threshold_mad": threshold,
            "score_median": median,
            "score_mad_std": mad_std,
            "training_dates": sorted(date_folders.keys()),
            "defect_folders_excluded": [DEFECT_FOLDER_PREFIX],
            "trim_head": TRIM_HEAD,
            "trim_tail": TRIM_TAIL,
            "subsample_normal": SUBSAMPLE_NORMAL,
            "valid_tiles_train": int(n_valid_train),
            "valid_tiles_all": int(n_valid_all),
        }
        with open(cam_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Defect inference
        if defect_folder:
            print(f"    [7] Defect inference...")
            results = run_defect_inference(
                defect_folder, cam_id, memory_bank, extractor,
                all_positions, pos_mean, pos_std, cam_dir)

            if results:
                max_zs = [r["max_z"] for r in results]
                anom_counts = [r["anomaly_count"] for r in results]
                n_detected = sum(1 for r in results if r["max_z"] > 2.5)
                print(f"      Defect images: {len(results)}")
                print(f"      max_z: {min(max_zs):.2f} ~ {max(max_zs):.2f} (median={np.median(max_zs):.2f})")
                print(f"      Detected (z>2.5): {n_detected}/{len(results)} ({100*n_detected/len(results):.1f}%)")

                with open(cam_dir / "defect_results.json", "w") as f:
                    json.dump(results, f, indent=2)

                all_results[cam_id] = {
                    "n_defect": len(results),
                    "n_detected": n_detected,
                    "detect_rate": n_detected / len(results) if results else 0,
                    "median_max_z": float(np.median(max_zs)),
                    "memory_bank_size": memory_bank.shape[0],
                }

        cam_elapsed = time.time() - cam_t0
        print(f"    Camera {cam_id} done in {cam_elapsed/60:.1f} min")

        # Free GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for cam_id, res in sorted(all_results.items()):
        print(f"  Camera {cam_id}: detect {res['n_detected']}/{res['n_defect']} "
              f"({100*res['detect_rate']:.1f}%), median_max_z={res['median_max_z']:.2f}, "
              f"bank={res['memory_bank_size']}")

    with open(OUTPUT_DIR / TARGET_SPEC / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed / 60:.1f} min")
    print(f"Output: {OUTPUT_DIR / TARGET_SPEC}")


if __name__ == "__main__":
    main()
