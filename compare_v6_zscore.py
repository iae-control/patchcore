"""v6 tile-based + per-position z-score normalization.

0630-only 모델 사용.
정상 이미지에서 타일 위치별 mean/std 계산 → z-score 정규화.
결함 히트맵에서 진짜 이상 영역만 부각.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json, time, re
from collections import defaultdict

# ===== Config =====
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v6/596x199/group_1_0630only")
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

CAM_IDS = [1, 10]
MIRROR_CAM = 10
TILE_SIZE = 128
TILE_STRIDE = 128
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
BRIGHTNESS_THRESHOLD = 30
TRIM_HEAD = 100
TRIM_TAIL = 100

# Per-position stats: how many normal images to sample
N_NORMAL_SAMPLES = 300
# Heatmaps to generate
MAX_HEATMAPS_DEFECT = 20
MAX_HEATMAPS_NORMAL = 10

from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def tile_positions(w, h, size, stride):
    positions = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            positions.append((x, y))
    return positions


class TileFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        from torchvision.models import wide_resnet50_2
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        local_weights = cache_dir / "wide_resnet50_2-95faca4d.pth"
        if local_weights.exists():
            backbone = wide_resnet50_2(weights=None)
            state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        else:
            from torchvision.models import Wide_ResNet50_2_Weights
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = self.layer1(x)
            f2 = self.layer2(h)
            f3 = self.layer3(f2)
            f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
            features = torch.cat([f2, f3_up], dim=1)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features.float()


def get_image_paths(folder, cam_id, subsample=1):
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
    if subsample > 1:
        images = images[::subsample]
    return images


def score_tiles(tile_features, memory_bank):
    if len(tile_features) == 0:
        return np.array([])
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    scores = []
    bs = 256
    for i in range(0, len(tf), bs):
        dists = torch.cdist(tf[i:i+bs], mb)
        min_dists, _ = dists.min(dim=1)
        scores.append(min_dists.cpu().numpy())
    return np.concatenate(scores)


def extract_and_score_image(img, extractor, positions, memory_bank):
    """Extract tile features and score, return per-position scores (NaN for masked)."""
    n_positions = len(positions)
    pos_scores = np.full(n_positions, np.nan)

    tiles = []
    valid_idx = []
    for i, (x, y) in enumerate(positions):
        tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        arr = np.array(tile)
        if arr.mean() < BRIGHTNESS_THRESHOLD:
            continue
        tiles.append(TRANSFORM(tile))
        valid_idx.append(i)

    if not tiles:
        return pos_scores, []

    all_feats = []
    for bs_start in range(0, len(tiles), 256):
        batch = torch.stack(tiles[bs_start:bs_start + 256])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    feats = np.concatenate(all_feats, axis=0)
    scores = score_tiles(feats, memory_bank)

    for k, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[k]

    return pos_scores, valid_idx


def compute_position_stats(normal_folders, extractor, positions, memory_bank, n_samples=300):
    """Compute per-position mean and std from normal images."""
    n_positions = len(positions)
    # Collect scores per position
    all_pos_scores = []  # list of arrays (n_positions,)

    count = 0
    for folder in normal_folders:
        if count >= n_samples:
            break
        for cam_id in CAM_IDS:
            if count >= n_samples:
                break
            images = get_image_paths(folder, cam_id, subsample=3)  # subsample for speed
            for img_path in images:
                if count >= n_samples:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                except:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue
                if cam_id == MIRROR_CAM:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                pos_scores, _ = extract_and_score_image(img, extractor, positions, memory_bank)
                all_pos_scores.append(pos_scores)
                count += 1

                if count % 50 == 0:
                    print(f"    Position stats: {count}/{n_samples} images")

    print(f"    Position stats computed from {count} images")

    # Stack: (n_images, n_positions)
    score_matrix = np.array(all_pos_scores)

    # Per-position mean and std (ignoring NaN)
    pos_mean = np.nanmean(score_matrix, axis=0)
    pos_std = np.nanstd(score_matrix, axis=0)
    pos_std = np.maximum(pos_std, 1e-6)  # avoid division by zero

    return pos_mean, pos_std


def make_zscore_heatmap(img, pos_scores, pos_mean, pos_std, positions,
                        n_tiles_x, n_tiles_y, title, output_path, z_threshold=3.0):
    """Draw z-score normalized heatmap."""
    # Z-score normalize
    z_scores = (pos_scores - pos_mean) / pos_std

    # Build grids
    raw_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
    z_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
    for i in range(len(positions)):
        row = i // n_tiles_x
        col = i % n_tiles_x
        raw_grid[row, col] = pos_scores[i]
        z_grid[row, col] = z_scores[i]

    max_raw = float(np.nanmax(pos_scores))
    max_z = float(np.nanmax(z_scores))
    status = "ANOMALY" if max_z > z_threshold else "NORMAL"
    color = "red" if max_z > z_threshold else "green"

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # 1. Original
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    # 2. Raw score heatmap
    axes[1].imshow(img)
    raw_vmax = max(0.8, np.nanmax(raw_grid)) if not np.all(np.isnan(raw_grid)) else 1
    hm1 = axes[1].imshow(raw_grid, cmap='jet', alpha=0.5,
                          extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                          interpolation='nearest', vmin=0, vmax=raw_vmax)
    plt.colorbar(hm1, ax=axes[1], fraction=0.03, pad=0.02)
    axes[1].set_title(f"Raw Score (max={max_raw:.3f})", fontsize=13)
    axes[1].axis("off")

    # 3. Z-score heatmap overlay
    axes[2].imshow(img)
    z_vmax = max(z_threshold * 2, np.nanmax(z_grid)) if not np.all(np.isnan(z_grid)) else 5
    hm2 = axes[2].imshow(z_grid, cmap='jet', alpha=0.5,
                          extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                          interpolation='nearest', vmin=-2, vmax=z_vmax)
    plt.colorbar(hm2, ax=axes[2], fraction=0.03, pad=0.02)
    axes[2].set_title(f"Z-Score Heatmap (max={max_z:.2f})", fontsize=13)
    axes[2].axis("off")

    # 4. Z-score thresholded overlay (only show z > threshold)
    axes[3].imshow(img)
    z_thresh_grid = np.where(z_grid > z_threshold, z_grid, np.nan)
    masked = np.ma.masked_invalid(z_thresh_grid)
    if masked.count() > 0:
        hm3 = axes[3].imshow(masked, cmap='hot', alpha=0.7,
                              extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                              interpolation='nearest', vmin=z_threshold, vmax=z_vmax)
        plt.colorbar(hm3, ax=axes[3], fraction=0.03, pad=0.02)
    axes[3].set_title(f"Anomaly Only (z>{z_threshold:.1f})", fontsize=13, color=color, fontweight='bold')
    axes[3].axis("off")

    fig.suptitle(f"{title} | raw_max={max_raw:.3f} | z_max={max_z:.2f} | {status}",
                 fontsize=14, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    return max_raw, max_z


def main():
    t0 = time.time()
    print("=" * 60)
    print("PatchCore v6 — Z-Score Normalized Comparison (0630 only)")
    print("=" * 60)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})")

    # Load model
    print("\n[0] Loading model...")
    extractor = TileFeatureExtractor("cuda")
    memory_bank = np.load(MODEL_DIR / "memory_bank.npy")
    with open(MODEL_DIR / "training_meta.json") as f:
        meta = json.load(f)
    raw_threshold = meta["threshold_mad"]
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Raw threshold: {raw_threshold:.4f}")

    # Discover 0630 folders
    print("\n[1] Discovering 0630 folders...")
    date_dir = NAS_ROOT / TRAIN_DATE
    normal_folders = []
    defect_folder = None
    for sub in sorted(date_dir.iterdir()):
        if sub.is_dir() and TARGET_SPEC in sub.name:
            if (sub / "camera_1").is_dir():
                if DEFECT_FOLDER_PREFIX in sub.name:
                    defect_folder = sub
                else:
                    normal_folders.append(sub)
    print(f"  Normal folders: {len(normal_folders)}")
    print(f"  Defect folder: {defect_folder.name if defect_folder else 'NOT FOUND'}")

    # Compute per-position stats from normal images
    print(f"\n[2] Computing per-position stats from {N_NORMAL_SAMPLES} normal images...")
    pos_mean, pos_std = compute_position_stats(
        normal_folders, extractor, positions, memory_bank, n_samples=N_NORMAL_SAMPLES)
    np.save(MODEL_DIR / "pos_mean.npy", pos_mean)
    np.save(MODEL_DIR / "pos_std.npy", pos_std)
    print(f"  pos_mean range: {np.nanmin(pos_mean):.4f} ~ {np.nanmax(pos_mean):.4f}")
    print(f"  pos_std range: {np.nanmin(pos_std):.4f} ~ {np.nanmax(pos_std):.4f}")

    # Z-score threshold
    Z_THRESHOLD = 3.5

    # Output dirs
    output_dir = MODEL_DIR / "zscore_compare"
    heatmap_defect_dir = output_dir / "heatmaps_defect"
    heatmap_normal_dir = output_dir / "heatmaps_normal"
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_defect_dir.mkdir(parents=True, exist_ok=True)
    heatmap_normal_dir.mkdir(parents=True, exist_ok=True)

    # ===== Score defect images =====
    print(f"\n[3] Scoring defect images (z-score)...")
    defect_results = []
    heatmap_count = 0

    if defect_folder:
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

                pos_scores, valid_idx = extract_and_score_image(
                    img, extractor, positions, memory_bank)
                z_scores = (pos_scores - pos_mean) / pos_std
                max_raw = float(np.nanmax(pos_scores))
                max_z = float(np.nanmax(z_scores))

                defect_results.append({
                    "file": img_path.name,
                    "cam": cam_id,
                    "max_raw": max_raw,
                    "max_z": max_z,
                    "mean_z": float(np.nanmean(z_scores)),
                    "anomaly": bool(max_z > Z_THRESHOLD),
                })

                # Heatmaps
                if heatmap_count < MAX_HEATMAPS_DEFECT:
                    make_zscore_heatmap(
                        img, pos_scores, pos_mean, pos_std, positions,
                        n_tiles_x, n_tiles_y,
                        f"DEFECT cam{cam_id} {img_path.name}",
                        heatmap_defect_dir / f"defect_{cam_id}_{heatmap_count:03d}.png",
                        z_threshold=Z_THRESHOLD)
                    heatmap_count += 1

        n_anom = sum(1 for r in defect_results if r["anomaly"])
        z_vals = [r["max_z"] for r in defect_results]
        print(f"  Defect images: {len(defect_results)}")
        print(f"  Z-score range: {min(z_vals):.2f} ~ {max(z_vals):.2f} (median={np.median(z_vals):.2f})")
        print(f"  Anomaly (z>{Z_THRESHOLD}): {n_anom}/{len(defect_results)} ({100*n_anom/max(len(defect_results),1):.1f}%)")

    # ===== Score normal images =====
    print(f"\n[4] Scoring normal images (z-score)...")
    normal_results = []
    heatmap_count = 0

    for folder in normal_folders[:5]:
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, subsample=5)
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

                pos_scores, valid_idx = extract_and_score_image(
                    img, extractor, positions, memory_bank)
                z_scores = (pos_scores - pos_mean) / pos_std
                max_raw = float(np.nanmax(pos_scores))
                max_z = float(np.nanmax(z_scores))

                normal_results.append({
                    "file": img_path.name,
                    "folder": folder.name,
                    "cam": cam_id,
                    "max_raw": max_raw,
                    "max_z": max_z,
                    "mean_z": float(np.nanmean(z_scores)),
                    "anomaly": bool(max_z > Z_THRESHOLD),
                })

                if heatmap_count < MAX_HEATMAPS_NORMAL:
                    make_zscore_heatmap(
                        img, pos_scores, pos_mean, pos_std, positions,
                        n_tiles_x, n_tiles_y,
                        f"NORMAL {folder.name} cam{cam_id}",
                        heatmap_normal_dir / f"normal_{cam_id}_{heatmap_count:03d}.png",
                        z_threshold=Z_THRESHOLD)
                    heatmap_count += 1

    n_fp = sum(1 for r in normal_results if r["anomaly"])
    z_vals_n = [r["max_z"] for r in normal_results]
    print(f"  Normal images: {len(normal_results)}")
    if z_vals_n:
        print(f"  Z-score range: {min(z_vals_n):.2f} ~ {max(z_vals_n):.2f} (median={np.median(z_vals_n):.2f})")
        print(f"  False positive (z>{Z_THRESHOLD}): {n_fp}/{len(normal_results)} ({100*n_fp/max(len(normal_results),1):.1f}%)")

    # ===== Histogram =====
    print(f"\n[5] Generating histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw score histogram
    defect_raw = [r["max_raw"] for r in defect_results]
    normal_raw = [r["max_raw"] for r in normal_results]
    axes[0].hist(normal_raw, bins=80, alpha=0.6, label=f"Normal (n={len(normal_raw)})", color='blue', edgecolor='black')
    axes[0].hist(defect_raw, bins=80, alpha=0.6, label=f"Defect (n={len(defect_raw)})", color='red', edgecolor='black')
    axes[0].axvline(raw_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={raw_threshold:.3f}')
    axes[0].set_xlabel("Raw Max Score (L2 distance)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw Score Distribution")
    axes[0].legend()

    # Z-score histogram
    defect_z = [r["max_z"] for r in defect_results]
    normal_z = [r["max_z"] for r in normal_results]
    axes[1].hist(normal_z, bins=80, alpha=0.6, label=f"Normal (n={len(normal_z)})", color='blue', edgecolor='black')
    axes[1].hist(defect_z, bins=80, alpha=0.6, label=f"Defect (n={len(defect_z)})", color='red', edgecolor='black')
    axes[1].axvline(Z_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Z-threshold={Z_THRESHOLD}')
    axes[1].set_xlabel("Max Z-Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Z-Score Distribution (per-position normalized)")
    axes[1].legend()

    fig.suptitle(f"PatchCore v6 — 0630 Only | Z-Score Normalized | Group 1", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "zscore_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "z_threshold": Z_THRESHOLD,
            "raw_threshold": raw_threshold,
            "n_normal_samples_for_stats": N_NORMAL_SAMPLES,
            "defect_count": len(defect_results),
            "defect_anomaly": sum(1 for r in defect_results if r["anomaly"]),
            "normal_count": len(normal_results),
            "normal_fp": n_fp,
            "defect_results": defect_results,
            "normal_results": normal_results,
        }, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
