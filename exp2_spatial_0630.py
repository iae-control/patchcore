#!/usr/bin/env python3
"""Experiment 2: v5b Spatial + 0630 Only + Z-Score Normalization.

0630 비결함 29개 폴더에서 spatial memory bank 구축 → per-position z-score → 결함 추론.
Spatial: AvgPool(k=3,s=3) → 50×80 position별 1536-dim 벡터 (global avg pool 없음).
과거 v5b에서 히트맵 localization이 잘 되었던 방식.
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
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp2_spatial_0630")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

TRIM_HEAD = 100
TRIM_TAIL = 100
SUBSAMPLE_TRAIN = 5  # Every 5th image for training (speed)

# Coreset
CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128

# Z-score
NORM_SAMPLE_IMAGES = 300
NORM_SUBSAMPLE_STEP = 5
ZSCORE_THRESHOLD = 3.0
GAUSSIAN_SIGMA = 4

# Camera
GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== BACKBONE =====
class SpatialFeatureExtractor(nn.Module):
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
            from torchvision.models import Wide_ResNet50_2_Weights
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.spatial_pool = nn.AvgPool2d(kernel_size=SPATIAL_POOL_K, stride=SPATIAL_POOL_S)
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
            features = self.spatial_pool(features)
        return features.float()

    def extract_spatial(self, x):
        feat_map = self.forward(x)
        B, C, H, W = feat_map.shape
        features = feat_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return features.cpu().numpy(), (H, W)


# ===== UTILS =====
def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


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


def score_spatial(features, memory_bank, batch_size=4096):
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(memory_bank).cuda()
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i + batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        min_d, _ = dists.min(dim=1)
        scores[i:i + batch_size] = min_d.cpu().numpy()
    return scores


# ===== CORESET =====
def greedy_coreset_selection(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)

    print(f"    Coreset: {n} -> {target} (ratio={ratio})")
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

    selected = np.array(selected)
    return features[selected], selected


# ===== DATA DISCOVERY =====
def discover_0630_folders():
    date_dir = NAS_ROOT / TRAIN_DATE
    normal_folders = []
    defect_folder = None
    if not date_dir.is_dir():
        print(f"ERROR: {date_dir} not found!")
        return [], None
    for sub in sorted(date_dir.iterdir()):
        if sub.is_dir() and TARGET_SPEC in sub.name:
            if (sub / "camera_1").is_dir():
                if DEFECT_FOLDER_PREFIX in sub.name:
                    defect_folder = sub
                else:
                    normal_folders.append(sub)
    return normal_folders, defect_folder


# ===== FEATURE EXTRACTION =====
def extract_spatial_features(folders, extractor, cam_ids, mirror_cam, subsample=5,
                              max_vectors_per_image=200):
    """Extract spatial features with per-image subsampling to control memory.

    4000 positions/image × 8000+ images = too much for RAM.
    Randomly sample max_vectors_per_image positions per image.
    """
    all_features = []
    total_images = 0
    spatial_size = None

    for folder in folders:
        folder_images = 0
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

                tensor = TRANSFORM(img).unsqueeze(0)
                feats, spatial_size = extractor.extract_spatial(tensor)

                # Subsample positions to control memory
                if len(feats) > max_vectors_per_image:
                    idx = np.random.choice(len(feats), max_vectors_per_image, replace=False)
                    feats = feats[idx]

                all_features.append(feats)
                total_images += 1
                folder_images += 1

                # Periodic memory check
                if total_images % 500 == 0:
                    current = np.concatenate(all_features, axis=0)
                    print(f"      Progress: {total_images} images, {current.shape[0]} vectors, "
                          f"~{current.nbytes / 1e9:.1f} GB")
                    del current

        print(f"    {folder.name}: {folder_images} images")

    features = np.concatenate(all_features, axis=0)
    del all_features
    gc.collect()
    print(f"    Total: {total_images} images, {features.shape[0]} spatial vectors ({features.shape[1]}-dim)")
    print(f"    Memory: ~{features.nbytes / 1e9:.1f} GB")
    return features, total_images, spatial_size


# ===== POSITION STATS =====
def build_position_stats(extractor, memory_bank, spatial_size, normal_folders, n_target=300):
    Hp, Wp = spatial_size
    all_score_maps = []
    count = 0

    for folder in normal_folders:
        if count >= n_target:
            break
        for cam_id in CAM_IDS:
            if count >= n_target:
                break
            images = get_image_paths(folder, cam_id, NORM_SUBSAMPLE_STEP)
            for img_path in images:
                if count >= n_target:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                except:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue
                if cam_id == MIRROR_CAM:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                tensor = TRANSFORM(img).unsqueeze(0)
                feats, _ = extractor.extract_spatial(tensor)
                scores = score_spatial(feats, memory_bank)
                score_map = scores.reshape(Hp, Wp)
                all_score_maps.append(score_map)
                count += 1
                if count % 50 == 0:
                    print(f"    Position stats: {count}/{n_target}")

    all_maps = np.stack(all_score_maps, axis=0)
    pos_mean = np.mean(all_maps, axis=0)
    pos_std = np.std(all_maps, axis=0)
    pos_std = np.maximum(pos_std, 0.01)

    print(f"    Stats from {count} images")
    print(f"    Mean range: {pos_mean.min():.4f} ~ {pos_mean.max():.4f}")
    print(f"    Std range: {pos_std.min():.4f} ~ {pos_std.max():.4f}")
    return pos_mean, pos_std


# ===== HEATMAP =====
def generate_heatmap(img, extractor, memory_bank, spatial_size, pos_mean, pos_std,
                     output_path, title, mirror=False):
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = TRANSFORM(img).unsqueeze(0)
    feats, _ = extractor.extract_spatial(tensor)
    scores = score_spatial(feats, memory_bank)

    Hp, Wp = spatial_size
    score_map = scores.reshape(Hp, Wp)
    zscore_map = (score_map - pos_mean) / pos_std

    # Upsample to full resolution
    z_tensor = torch.from_numpy(zscore_map).unsqueeze(0).unsqueeze(0).float()
    zscore_full = F.interpolate(z_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                 mode='bilinear', align_corners=False).squeeze().numpy()
    raw_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    raw_full = F.interpolate(raw_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                              mode='bilinear', align_corners=False).squeeze().numpy()

    zscore_full = gaussian_filter(zscore_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)
    raw_full = gaussian_filter(raw_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)

    if mirror:
        zscore_full = np.fliplr(zscore_full)
        raw_full = np.fliplr(raw_full)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    max_z = float(zscore_full.max())
    max_raw = float(raw_full.max())

    img_arr = np.array(img)
    fig, axes = plt.subplots(1, 4, figsize=(36, 8))

    # Original
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    # Raw heatmap
    vmin_r = np.percentile(raw_full, 5)
    vmax_r = np.percentile(raw_full, 99.5)
    im1 = axes[1].imshow(raw_full, cmap='hot', vmin=vmin_r, vmax=vmax_r)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Raw Score (max={max_raw:.3f})", fontsize=13)
    axes[1].axis("off")

    # Z-score heatmap
    im2 = axes[2].imshow(zscore_full, cmap='hot',
                          vmin=0, vmax=max(ZSCORE_THRESHOLD * 1.5, zscore_full.max() * 0.9))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_title(f"Z-Score (max={max_z:.2f})", fontsize=13)
    axes[2].axis("off")

    # Overlay
    axes[3].imshow(img_arr)
    masked_z = np.ma.masked_where(zscore_full < ZSCORE_THRESHOLD * 0.7, zscore_full)
    if masked_z.count() > 0:
        im3 = axes[3].imshow(masked_z, cmap='jet', alpha=0.6,
                              vmin=ZSCORE_THRESHOLD * 0.7,
                              vmax=max(ZSCORE_THRESHOLD * 2, zscore_full.max()))
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

    status = "ANOMALY" if max_z > ZSCORE_THRESHOLD else "NORMAL"
    color = 'red' if max_z > ZSCORE_THRESHOLD else 'green'
    axes[3].set_title(f"{status} (z={max_z:.2f})", fontsize=14, color=color, fontweight='bold')
    axes[3].axis("off")

    fig.suptitle(f"Exp2 Spatial+ZScore | {title} | raw={max_raw:.3f} z={max_z:.2f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return max_raw, max_z


# ===== MAIN =====
def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment 2: Spatial Features + 0630 Only + Z-Score")
    print("=" * 60)

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backbone
    print("\n[0] Loading backbone...")
    extractor = SpatialFeatureExtractor("cuda")

    # Get spatial size from a test image
    print("\n[1] Discovering 0630 folders...")
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)} folders")
    print(f"  Defect: {defect_folder.name if defect_folder else 'NOT FOUND'}")

    # Feature extraction
    print(f"\n[2] Extracting spatial features (subsample={SUBSAMPLE_TRAIN})...")
    features, total_images, spatial_size = extract_spatial_features(
        normal_folders, extractor, CAM_IDS, MIRROR_CAM, subsample=SUBSAMPLE_TRAIN)
    print(f"  Spatial size: {spatial_size}")

    # Coreset
    print("\n[3] Coreset selection...")
    memory_bank, _ = greedy_coreset_selection(features)
    print(f"  Memory bank: {memory_bank.shape}")
    np.save(output_dir / "memory_bank.npy", memory_bank)
    np.save(output_dir / "spatial_size.npy", np.array(spatial_size))

    # Free memory
    del features
    gc.collect()
    torch.cuda.empty_cache()

    # Position stats
    print(f"\n[4] Building per-position stats ({NORM_SAMPLE_IMAGES} images)...")
    pos_mean, pos_std = build_position_stats(
        extractor, memory_bank, spatial_size, normal_folders, NORM_SAMPLE_IMAGES)
    np.save(output_dir / "pos_mean.npy", pos_mean)
    np.save(output_dir / "pos_std.npy", pos_std)

    # Defect inference
    print("\n[5] Defect inference with z-score...")
    heatmap_defect_dir = output_dir / "heatmaps_defect"
    heatmap_normal_dir = output_dir / "heatmaps_normal"
    heatmap_defect_dir.mkdir(parents=True, exist_ok=True)
    heatmap_normal_dir.mkdir(parents=True, exist_ok=True)

    defect_results = []
    heatmap_count = 0
    MAX_HEATMAPS = 20

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

                tensor = TRANSFORM(img if not mirror else img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
                feats, _ = extractor.extract_spatial(tensor)
                scores = score_spatial(feats, memory_bank)
                Hp, Wp = spatial_size
                score_map = scores.reshape(Hp, Wp)
                zscore_map = (score_map - pos_mean) / pos_std
                max_raw = float(scores.max())
                max_z = float(zscore_map.max())

                defect_results.append({
                    "file": img_path.name,
                    "cam": cam_id,
                    "max_raw": max_raw,
                    "max_z": max_z,
                    "mean_z": float(zscore_map.mean()),
                    "anomaly": bool(max_z > ZSCORE_THRESHOLD),
                })

                if heatmap_count < MAX_HEATMAPS:
                    generate_heatmap(img, extractor, memory_bank, spatial_size, pos_mean, pos_std,
                                     heatmap_defect_dir / f"defect_{cam_id}_{heatmap_count:03d}.png",
                                     f"DEFECT cam{cam_id} {img_path.name}", mirror=mirror)
                    heatmap_count += 1

        n_anom = sum(1 for r in defect_results if r["anomaly"])
        z_vals = [r["max_z"] for r in defect_results]
        print(f"  Defect: {len(defect_results)} images")
        print(f"  Z-score: {min(z_vals):.2f} ~ {max(z_vals):.2f} (median={np.median(z_vals):.2f})")
        print(f"  Anomaly (z>{ZSCORE_THRESHOLD}): {n_anom}/{len(defect_results)} ({100*n_anom/max(len(defect_results),1):.1f}%)")

    # Normal inference (sample)
    print("\n[6] Normal inference (sample)...")
    normal_results = []
    heatmap_count = 0

    for folder in normal_folders[:5]:
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, subsample_step=10)
            mirror = (cam_id == MIRROR_CAM)
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                except:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue

                tensor = TRANSFORM(img if not mirror else img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
                feats, _ = extractor.extract_spatial(tensor)
                scores = score_spatial(feats, memory_bank)
                Hp, Wp = spatial_size
                score_map = scores.reshape(Hp, Wp)
                zscore_map = (score_map - pos_mean) / pos_std
                max_raw = float(scores.max())
                max_z = float(zscore_map.max())

                normal_results.append({
                    "file": img_path.name,
                    "folder": folder.name,
                    "cam": cam_id,
                    "max_raw": max_raw,
                    "max_z": max_z,
                    "anomaly": bool(max_z > ZSCORE_THRESHOLD),
                })

                if heatmap_count < 10:
                    generate_heatmap(img, extractor, memory_bank, spatial_size, pos_mean, pos_std,
                                     heatmap_normal_dir / f"normal_{cam_id}_{heatmap_count:03d}.png",
                                     f"NORMAL {folder.name} cam{cam_id}", mirror=mirror)
                    heatmap_count += 1

    n_fp = sum(1 for r in normal_results if r["anomaly"])
    z_vals_n = [r["max_z"] for r in normal_results]
    print(f"  Normal: {len(normal_results)} images")
    if z_vals_n:
        print(f"  Z-score: {min(z_vals_n):.2f} ~ {max(z_vals_n):.2f} (median={np.median(z_vals_n):.2f})")
        print(f"  FP (z>{ZSCORE_THRESHOLD}): {n_fp}/{len(normal_results)} ({100*n_fp/max(len(normal_results),1):.1f}%)")

    # Histogram
    print("\n[7] Generating histogram...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    defect_z = [r["max_z"] for r in defect_results]
    normal_z = [r["max_z"] for r in normal_results]
    ax.hist(normal_z, bins=80, alpha=0.6, label=f"Normal (n={len(normal_z)})", color='blue', edgecolor='black')
    ax.hist(defect_z, bins=80, alpha=0.6, label=f"Defect (n={len(defect_z)})", color='red', edgecolor='black')
    ax.axvline(ZSCORE_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Z-threshold={ZSCORE_THRESHOLD}')
    ax.set_xlabel("Max Z-Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Exp2: Spatial + 0630 Only + Z-Score | Group {GROUP_ID}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save meta
    meta = {
        "experiment": "exp2_spatial_0630",
        "method": "spatial_features_no_pooling",
        "training_date": TRAIN_DATE,
        "training_folders": [f.name for f in normal_folders],
        "excluded_defect": defect_folder.name if defect_folder else None,
        "spatial_pool": f"AvgPool(k={SPATIAL_POOL_K}, s={SPATIAL_POOL_S})",
        "spatial_size": list(spatial_size),
        "total_train_images": total_images,
        "memory_bank_shape": list(memory_bank.shape),
        "coreset_ratio": CORESET_RATIO,
        "subsample_train": SUBSAMPLE_TRAIN,
        "zscore_threshold": ZSCORE_THRESHOLD,
        "norm_sample_images": NORM_SAMPLE_IMAGES,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "defect_count": len(defect_results),
        "defect_anomaly": sum(1 for r in defect_results if r["anomaly"]),
        "normal_count": len(normal_results),
        "normal_fp": n_fp,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "results.json", "w") as f:
        json.dump({"defect": defect_results, "normal": normal_results}, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
