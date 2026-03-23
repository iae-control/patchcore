#!/usr/bin/env python3
"""Experiment 3b: 32x32 Tiles + 0630 Only + Z-Score.

32x32 타일로 결함 해상도 극대화.
60x37=2220개/이미지. 세로 균열이 타일 내 비중이 더 커짐.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp3b_tile32")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 32
TILE_STRIDE = 32  # no overlap

TRIM_HEAD = 100
TRIM_TAIL = 100
SUBSAMPLE_TRAIN = 5

CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128
BRIGHTNESS_THRESHOLD = 30

# Z-score
NORM_SAMPLE_IMAGES = 300
ZSCORE_THRESHOLD = 3.5

GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10
BATCH_SIZE = 1024  # 32x32 tiles are tiny, can fit more

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TileFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
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


def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def tile_positions(w, h, size, stride):
    positions = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
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


def score_tiles(tile_features, memory_bank):
    if len(tile_features) == 0:
        return np.array([])
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    scores = []
    for i in range(0, len(tf), 256):
        dists = torch.cdist(tf[i:i+256], mb)
        min_dists, _ = dists.min(dim=1)
        scores.append(min_dists.cpu().numpy())
    return np.concatenate(scores)


def greedy_coreset_selection(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)
    print(f"    Coreset: {n} -> {target}")
    device = "cuda"
    proj = torch.randn(d, proj_dim, device=device) / (proj_dim ** 0.5)
    feat_tensor = torch.from_numpy(features).to(device)
    feat_proj = feat_tensor @ proj
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
    return features[np.array(selected)], np.array(selected)


def discover_0630_folders():
    date_dir = NAS_ROOT / TRAIN_DATE
    normal_folders, defect_folder = [], None
    for sub in sorted(date_dir.iterdir()):
        if sub.is_dir() and TARGET_SPEC in sub.name and (sub / "camera_1").is_dir():
            if DEFECT_FOLDER_PREFIX in sub.name:
                defect_folder = sub
            else:
                normal_folders.append(sub)
    return normal_folders, defect_folder


def extract_and_score_image(img, extractor, positions, memory_bank):
    n_positions = len(positions)
    pos_scores = np.full(n_positions, np.nan)
    tiles, valid_idx = [], []
    for i, (x, y) in enumerate(positions):
        tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        arr = np.array(tile)
        if arr.mean() < BRIGHTNESS_THRESHOLD:
            continue
        tiles.append(TRANSFORM(tile))
        valid_idx.append(i)
    if not tiles:
        return pos_scores, valid_idx
    all_feats = []
    for bs in range(0, len(tiles), BATCH_SIZE):
        batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    feats = np.concatenate(all_feats, axis=0)
    scores = score_tiles(feats, memory_bank)
    for k, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[k]
    return pos_scores, valid_idx


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment 3b: 32x32 Tiles + 0630 Only + Z-Score")
    print("=" * 60)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})")

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[0] Loading backbone...")
    extractor = TileFeatureExtractor("cuda")

    print("\n[1] Discovering 0630...")
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)}, Defect: {defect_folder.name if defect_folder else 'N/A'}")

    # Feature extraction
    print(f"\n[2] Extracting 32x32 tile features (subsample={SUBSAMPLE_TRAIN})...")
    all_features = []
    total_images = 0
    for folder in normal_folders:
        folder_imgs = 0
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, SUBSAMPLE_TRAIN)
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
                tiles, valid_idx = [], []
                for i, (x, y) in enumerate(positions):
                    tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
                    arr = np.array(tile)
                    if arr.mean() < BRIGHTNESS_THRESHOLD:
                        continue
                    tiles.append(TRANSFORM(tile))
                    valid_idx.append(i)
                if not tiles:
                    continue
                for bs in range(0, len(tiles), BATCH_SIZE):
                    batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
                    feats = extractor(batch).cpu().numpy()
                    all_features.append(feats)
                total_images += 1
                folder_imgs += 1
        print(f"    {folder.name}: {folder_imgs} images")

    features = np.concatenate(all_features, axis=0)
    print(f"  Total: {total_images} images, {features.shape[0]} tiles")
    del all_features; gc.collect()

    # Coreset
    print("\n[3] Coreset...")
    memory_bank, _ = greedy_coreset_selection(features)
    print(f"  Memory bank: {memory_bank.shape}")
    np.save(output_dir / "memory_bank.npy", memory_bank)
    del features; gc.collect(); torch.cuda.empty_cache()

    # Position stats
    print(f"\n[4] Position stats ({NORM_SAMPLE_IMAGES} images)...")
    all_pos_scores = []
    count = 0
    for folder in normal_folders:
        if count >= NORM_SAMPLE_IMAGES:
            break
        for cam_id in CAM_IDS:
            if count >= NORM_SAMPLE_IMAGES:
                break
            images = get_image_paths(folder, cam_id, 3)
            for img_path in images:
                if count >= NORM_SAMPLE_IMAGES:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                except:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue
                if cam_id == MIRROR_CAM:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                ps, _ = extract_and_score_image(img, extractor, positions, memory_bank)
                all_pos_scores.append(ps)
                count += 1
                if count % 50 == 0:
                    print(f"    {count}/{NORM_SAMPLE_IMAGES}")

    score_matrix = np.array(all_pos_scores)
    pos_mean = np.nanmean(score_matrix, axis=0)
    pos_std = np.nanstd(score_matrix, axis=0)
    pos_std = np.maximum(pos_std, 1e-6)
    np.save(output_dir / "pos_mean.npy", pos_mean)
    np.save(output_dir / "pos_std.npy", pos_std)
    print(f"  Mean: {np.nanmin(pos_mean):.4f}~{np.nanmax(pos_mean):.4f}")
    print(f"  Std: {np.nanmin(pos_std):.4f}~{np.nanmax(pos_std):.4f}")

    # Defect inference
    print("\n[5] Defect inference...")
    defect_results = []
    heatmap_dir = output_dir / "heatmaps_defect"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    hm_count = 0

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
                ps, vi = extract_and_score_image(img, extractor, positions, memory_bank)
                z = (ps - pos_mean) / pos_std
                max_z = float(np.nanmax(z))
                max_raw = float(np.nanmax(ps))
                defect_results.append({
                    "file": img_path.name, "cam": cam_id,
                    "max_raw": max_raw, "max_z": max_z,
                    "anomaly": bool(max_z > ZSCORE_THRESHOLD),
                })
                if hm_count < 20:
                    # Heatmap
                    z_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
                    raw_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
                    for i in range(len(positions)):
                        row, col = i // n_tiles_x, i % n_tiles_x
                        z_grid[row, col] = z[i]
                        raw_grid[row, col] = ps[i]

                    if mirror:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    fig, axes = plt.subplots(1, 4, figsize=(32, 7))
                    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")

                    hm1 = axes[1].imshow(raw_grid, cmap='hot', interpolation='nearest',
                                          extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0])
                    plt.colorbar(hm1, ax=axes[1], fraction=0.03)
                    axes[1].imshow(img, alpha=0.3); axes[1].set_title(f"Raw (max={max_raw:.3f})"); axes[1].axis("off")

                    z_vmax = max(ZSCORE_THRESHOLD * 2, np.nanmax(z_grid))
                    hm2 = axes[2].imshow(z_grid, cmap='hot', interpolation='nearest',
                                          extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0], vmin=-2, vmax=z_vmax)
                    plt.colorbar(hm2, ax=axes[2], fraction=0.03)
                    axes[2].imshow(img, alpha=0.3); axes[2].set_title(f"Z-Score (max={max_z:.2f})"); axes[2].axis("off")

                    axes[3].imshow(img)
                    z_thresh = np.where(z_grid > ZSCORE_THRESHOLD, z_grid, np.nan)
                    masked = np.ma.masked_invalid(z_thresh)
                    if masked.count() > 0:
                        hm3 = axes[3].imshow(masked, cmap='hot', alpha=0.7,
                                              extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                                              interpolation='nearest', vmin=ZSCORE_THRESHOLD, vmax=z_vmax)
                        plt.colorbar(hm3, ax=axes[3], fraction=0.03)
                    status = "ANOMALY" if max_z > ZSCORE_THRESHOLD else "NORMAL"
                    clr = 'red' if max_z > ZSCORE_THRESHOLD else 'green'
                    axes[3].set_title(f"{status} (z={max_z:.2f})", color=clr, fontweight='bold')
                    axes[3].axis("off")

                    fig.suptitle(f"Exp3b 32x32 | cam{cam_id} {img_path.name}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(heatmap_dir / f"defect_{cam_id}_{hm_count:03d}.png", dpi=120, bbox_inches='tight')
                    plt.close()
                    hm_count += 1

        n_anom = sum(1 for r in defect_results if r["anomaly"])
        zv = [r["max_z"] for r in defect_results]
        print(f"  Defect: {len(defect_results)}, z={min(zv):.2f}~{max(zv):.2f} (med={np.median(zv):.2f})")
        print(f"  Anomaly: {n_anom}/{len(defect_results)} ({100*n_anom/max(len(defect_results),1):.1f}%)")

    # Normal inference
    print("\n[6] Normal inference...")
    normal_results = []
    for folder in normal_folders[:5]:
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, 10)
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
                ps, _ = extract_and_score_image(img, extractor, positions, memory_bank)
                z = (ps - pos_mean) / pos_std
                normal_results.append({
                    "file": img_path.name, "folder": folder.name, "cam": cam_id,
                    "max_z": float(np.nanmax(z)), "anomaly": bool(np.nanmax(z) > ZSCORE_THRESHOLD),
                })

    n_fp = sum(1 for r in normal_results if r["anomaly"])
    zn = [r["max_z"] for r in normal_results]
    print(f"  Normal: {len(normal_results)}, z={min(zn):.2f}~{max(zn):.2f} (med={np.median(zn):.2f})")
    print(f"  FP: {n_fp}/{len(normal_results)} ({100*n_fp/max(len(normal_results),1):.1f}%)")

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(zn, bins=80, alpha=0.6, label=f"Normal (n={len(zn)})", color='blue', edgecolor='black')
    ax.hist([r["max_z"] for r in defect_results], bins=80, alpha=0.6, label=f"Defect (n={len(defect_results)})", color='red', edgecolor='black')
    ax.axvline(ZSCORE_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'z={ZSCORE_THRESHOLD}')
    ax.set_xlabel("Max Z-Score"); ax.set_ylabel("Count")
    ax.set_title(f"Exp3: 64x64 Tiles + 0630 + Z-Score | Group {GROUP_ID}")
    ax.legend(); plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=150); plt.close()

    # Meta
    meta = {
        "experiment": "exp3b_tile32_0630", "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
        "tiles_per_image": len(positions), "grid": [n_tiles_x, n_tiles_y],
        "training_date": TRAIN_DATE, "training_folders": [f.name for f in normal_folders],
        "total_train_images": total_images, "memory_bank_shape": list(memory_bank.shape),
        "coreset_ratio": CORESET_RATIO, "zscore_threshold": ZSCORE_THRESHOLD,
        "defect_count": len(defect_results),
        "defect_anomaly": sum(1 for r in defect_results if r["anomaly"]),
        "normal_count": len(normal_results), "normal_fp": n_fp,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
