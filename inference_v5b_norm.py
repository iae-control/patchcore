#!/usr/bin/env python3
"""v5b inference with per-position score normalization.

Key idea: each position in the 50x80 spatial grid has its own normal
score distribution (mean, std). High-variability zones (flange-web junction)
get higher std -> z-score stays low -> no false alarm.
Low-variability zones (web center) get low std -> real defect causes
high z-score -> proper detection.

Steps:
1. Load saved memory bank
2. Score a batch of NORMAL training images to build per-position stats
3. Inference on defect folder using z-score normalization
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5b")
TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3
TRIM_HEAD = 100
TRIM_TAIL = 100

GAUSSIAN_SIGMA = 4
NORM_SAMPLE_IMAGES = 300  # number of normal images to build per-position stats
NORM_SUBSAMPLE_STEP = 5   # sample every 5th image from normal folders
ZSCORE_THRESHOLD = 3.0    # z-score threshold for anomaly


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.spatial_pool = nn.AvgPool2d(kernel_size=SPATIAL_POOL_K, stride=SPATIAL_POOL_S)
        self.to(device)
        self.eval()

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


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_paths(folder, cam_id, subsample_step=1):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.exists():
        return []
    images = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
    if TRIM_HEAD + TRIM_TAIL < len(images):
        images = images[TRIM_HEAD: len(images) - TRIM_TAIL]
    return images[::subsample_step]


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


def build_position_stats(extractor, memory_bank, spatial_size, train_folders,
                         cam_ids, mirror_cam, n_target=NORM_SAMPLE_IMAGES):
    """Score normal training images and compute per-position mean/std."""
    Hp, Wp = spatial_size
    all_score_maps = []
    count = 0

    print(f"  Building per-position stats from {n_target} normal images...")
    sys.stdout.flush()

    for folder in train_folders:
        if count >= n_target:
            break
        for cam_id in cam_ids:
            if count >= n_target:
                break
            images = get_image_paths(folder, cam_id, subsample_step=NORM_SUBSAMPLE_STEP)
            for img_path in images:
                if count >= n_target:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue
                if cam_id == mirror_cam:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                tensor = TRANSFORM(img).unsqueeze(0)
                features, _ = extractor.extract_spatial(tensor)
                scores = score_spatial(features, memory_bank)
                score_map = scores.reshape(Hp, Wp)
                all_score_maps.append(score_map)
                count += 1

                if count % 50 == 0:
                    print(f"    {count}/{n_target} images scored")
                    sys.stdout.flush()

    all_maps = np.stack(all_score_maps, axis=0)  # (N, Hp, Wp)
    pos_mean = np.mean(all_maps, axis=0)  # (Hp, Wp)
    pos_std = np.std(all_maps, axis=0)    # (Hp, Wp)
    # Avoid division by zero: set minimum std
    pos_std = np.maximum(pos_std, 0.01)

    print(f"  Stats computed from {count} images")
    print(f"  Mean score range: {pos_mean.min():.4f} ~ {pos_mean.max():.4f}")
    print(f"  Std range: {pos_std.min():.4f} ~ {pos_std.max():.4f}")
    sys.stdout.flush()

    return pos_mean, pos_std


def generate_heatmap_normalized(img_path, memory_bank, extractor, spatial_size,
                                 pos_mean, pos_std, output_path, mirror=False):
    """Generate z-score normalized anomaly heatmap."""
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        return None, None, None

    img_display = img.copy()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = TRANSFORM(img).unsqueeze(0)
    features, _ = extractor.extract_spatial(tensor)
    scores = score_spatial(features, memory_bank)

    Hp, Wp = spatial_size
    score_map = scores.reshape(Hp, Wp)

    # === PER-POSITION Z-SCORE NORMALIZATION ===
    zscore_map = (score_map - pos_mean) / pos_std

    # Upsample z-score map to full resolution
    z_tensor = torch.from_numpy(zscore_map).unsqueeze(0).unsqueeze(0).float()
    zscore_full = F.interpolate(
        z_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Also upsample raw score map for comparison
    raw_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    raw_full = F.interpolate(
        raw_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Gaussian smoothing
    zscore_full = gaussian_filter(zscore_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)
    raw_full = gaussian_filter(raw_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)

    if mirror:
        zscore_full = np.fliplr(zscore_full)
        raw_full = np.fliplr(raw_full)

    max_zscore = float(zscore_full.max())
    max_raw = float(raw_full.max())

    # === VISUALIZATION: 4 panels ===
    img_arr = np.array(img_display)
    fig, axes = plt.subplots(1, 4, figsize=(48, 10))

    # Panel 1: Original
    axes[0].imshow(img_arr)
    axes[0].set_title(f"Original: {img_path.name}", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Raw heatmap (for comparison)
    vmin_r = np.percentile(raw_full, 5)
    vmax_r = np.percentile(raw_full, 99.5)
    im1 = axes[1].imshow(raw_full, cmap='hot', vmin=vmin_r, vmax=vmax_r)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Raw Score (max={max_raw:.3f})", fontsize=13)
    axes[1].axis("off")

    # Panel 3: Z-score heatmap
    im2 = axes[2].imshow(zscore_full, cmap='hot',
                          vmin=0, vmax=max(ZSCORE_THRESHOLD * 1.5, zscore_full.max() * 0.9))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_title(f"Z-Score (max={max_zscore:.2f})", fontsize=13)
    axes[2].axis("off")

    # Panel 4: Overlay - only show z-score > threshold
    axes[3].imshow(img_arr)
    masked_z = np.ma.masked_where(zscore_full < ZSCORE_THRESHOLD * 0.7, zscore_full)
    if masked_z.count() > 0:
        im3 = axes[3].imshow(masked_z, cmap='jet', alpha=0.6,
                              vmin=ZSCORE_THRESHOLD * 0.7,
                              vmax=max(ZSCORE_THRESHOLD * 2, zscore_full.max()))
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

    if max_zscore > ZSCORE_THRESHOLD:
        status = f"ANOMALY (z={max_zscore:.2f})"
        color = 'red'
    else:
        status = f"NORMAL (z={max_zscore:.2f})"
        color = 'green'
    axes[3].set_title(status, fontsize=14, color=color, fontweight='bold')
    axes[3].axis("off")

    plt.suptitle(f"PatchCore v5b | {TARGET_SPEC}/group_1 | {img_path.name} | "
                 f"Per-position Z-score (thr={ZSCORE_THRESHOLD})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return max_raw, max_zscore, max_zscore > ZSCORE_THRESHOLD


def main():
    print("=" * 60)
    print("PatchCore v5b - PER-POSITION NORMALIZED INFERENCE")
    print("=" * 60)

    # 1. Load saved model
    model_dir = OUTPUT_DIR / TARGET_SPEC / "group_1"
    memory_bank = np.load(model_dir / "memory_bank.npy")
    spatial_size = tuple(np.load(model_dir / "spatial_size.npy"))
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Spatial size: {spatial_size}")
    sys.stdout.flush()

    # 2. Load extractor
    print("\nLoading WideResNet50...")
    extractor = SpatialFeatureExtractor(device="cuda")
    sys.stdout.flush()

    # 3. Find training folders (excluding defect folder)
    spec_pattern = re.compile(r"_(\d+x\d+)$")
    train_folders = []
    for date_dir in sorted(NAS_ROOT.iterdir()):
        if not date_dir.is_dir():
            continue
        for folder in sorted(date_dir.iterdir()):
            if not folder.is_dir():
                continue
            m = spec_pattern.search(folder.name)
            if m and m.group(1) == TARGET_SPEC and DEFECT_FOLDER_PREFIX not in folder.name:
                train_folders.append(folder)

    print(f"  Found {len(train_folders)} normal training folders")

    # Camera config for group 1
    CAMERA_CAMS = [1, 10]
    MIRROR_CAM = 10

    # 4. Build per-position normalization stats
    print("\n[1/2] Building per-position normalization stats...")
    t0 = time.time()
    pos_mean, pos_std = build_position_stats(
        extractor, memory_bank, spatial_size,
        train_folders, CAMERA_CAMS, MIRROR_CAM, n_target=NORM_SAMPLE_IMAGES
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Save stats
    np.save(model_dir / "pos_mean.npy", pos_mean)
    np.save(model_dir / "pos_std.npy", pos_std)
    print(f"  Saved pos_mean.npy, pos_std.npy")

    # Visualize the std map (shows where variability is high)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    im0 = axes[0].imshow(pos_mean, cmap='hot')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Per-position Mean Score\n(range: {pos_mean.min():.3f}~{pos_mean.max():.3f})")

    im1 = axes[1].imshow(pos_std, cmap='hot')
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"Per-position Std (variability)\n(range: {pos_std.min():.4f}~{pos_std.max():.4f})")
    plt.suptitle("Normal Score Statistics (50x80 grid)", fontsize=14)
    plt.tight_layout()
    plt.savefig(model_dir / "position_stats.png", dpi=120)
    plt.close()
    sys.stdout.flush()

    # 5. Inference on defect folder
    print("\n[2/2] Inference with per-position normalization...")
    defect_folder = None
    for d in sorted(NAS_ROOT.glob("20250630/*")):
        if d.is_dir() and DEFECT_FOLDER_PREFIX in d.name:
            defect_folder = d
            break

    if not defect_folder:
        print("ERROR: defect folder not found!")
        return

    print(f"  Defect folder: {defect_folder.name}")

    cam1_images = get_image_paths(defect_folder, cam_id=1, subsample_step=1)
    print(f"  Camera 1 images: {len(cam1_images)}")

    heatmap_dir = model_dir / "heatmaps_v3" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Dense sampling for better evaluation
    n = len(cam1_images)
    sample_idx = set()
    sample_idx.update(range(min(5, n)))
    sample_idx.update(range(max(0, n - 5), n))
    sample_idx.update(range(0, n, 50))
    sample_idx.update(range(n // 4, min(n // 4 + 3, n)))
    sample_idx.update(range(n // 2, min(n // 2 + 3, n)))
    sample_idx.update(range(3 * n // 4, min(3 * n // 4 + 3, n)))
    sample_indices = sorted(sample_idx)

    print(f"  Generating z-score heatmaps for {len(sample_indices)} images...")
    sys.stdout.flush()

    results = []
    for idx in tqdm(sample_indices, desc="Z-Score Heatmaps"):
        img_path = cam1_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_raw, max_z, is_anomaly = generate_heatmap_normalized(
            img_path, memory_bank, extractor, spatial_size,
            pos_mean, pos_std, out_path
        )
        if max_raw is not None:
            results.append({
                "idx": idx, "file": img_path.name,
                "max_raw_score": max_raw, "max_zscore": max_z,
                "anomaly": is_anomaly,
            })

    with open(model_dir / "inference_results_v3.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_anomaly = sum(1 for r in results if r["anomaly"])
    all_z = [r["max_zscore"] for r in results]
    all_raw = [r["max_raw_score"] for r in results]
    print(f"\n{'='*60}")
    print(f"  Images: {len(results)}")
    print(f"  Anomaly (z>{ZSCORE_THRESHOLD}): {n_anomaly}/{len(results)}")
    print(f"  Z-score range: {min(all_z):.2f} ~ {max(all_z):.2f}")
    print(f"  Raw score range: {min(all_raw):.3f} ~ {max(all_raw):.3f}")
    print(f"  Heatmaps: {heatmap_dir}")
    print(f"{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
