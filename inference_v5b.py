#!/usr/bin/env python3
"""v5b inference-only: load saved memory bank, generate improved heatmaps.
Fixes: Gaussian smoothing, threshold-based normalization, cleaner overlay.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re
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
import matplotlib.colors as mcolors

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5b")
TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

# Must match training config
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3
TRIM_HEAD = 100
TRIM_TAIL = 100

# Heatmap improvements
GAUSSIAN_SIGMA = 4  # PatchCore paper standard
SCORE_PERCENTILE_CAP = 99.5  # cap extreme values


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


def generate_heatmap_v2(img_path, memory_bank, extractor, spatial_size, output_path,
                        threshold, mirror=False):
    """Improved heatmap: Gaussian smoothing + threshold-based visualization."""
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        return None, None

    img_display = img.copy()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = TRANSFORM(img).unsqueeze(0)
    features, _ = extractor.extract_spatial(tensor)
    scores = score_spatial(features, memory_bank)

    Hp, Wp = spatial_size
    score_map = scores.reshape(Hp, Wp)

    # Upsample raw score map to full resolution FIRST
    score_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    heatmap_full = F.interpolate(
        score_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # === Gaussian smoothing AFTER upsampling (PatchCore paper) ===
    heatmap_full = gaussian_filter(heatmap_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)

    if mirror:
        heatmap_full = np.fliplr(heatmap_full)

    max_score = float(heatmap_full.max())
    mean_score = float(np.mean(scores))

    # === VISUALIZATION ===
    img_arr = np.array(img_display)
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # Panel 1: Original
    axes[0].imshow(img_arr)
    axes[0].set_title(f"Original: {img_path.name}", fontsize=14)
    axes[0].axis("off")

    # Panel 2: Raw heatmap with Gaussian smoothing
    vmin = np.percentile(heatmap_full, 5)
    vmax = np.percentile(heatmap_full, SCORE_PERCENTILE_CAP)
    im1 = axes[1].imshow(heatmap_full, cmap='hot', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Heatmap (max={max_score:.3f}, thr={threshold:.3f})", fontsize=14)
    axes[1].axis("off")

    # Panel 3: Overlay - threshold-based
    # Only show anomaly regions (above threshold) with clear coloring
    axes[2].imshow(img_arr)

    # Create masked heatmap: only show scores above a visible threshold
    overlay_min = threshold * 0.85  # slightly below threshold for gradient
    overlay_max = np.percentile(heatmap_full, SCORE_PERCENTILE_CAP)
    if overlay_max <= overlay_min:
        overlay_max = overlay_min + 0.1

    # Mask below overlay_min to transparent
    masked_heatmap = np.ma.masked_where(heatmap_full < overlay_min, heatmap_full)

    im2 = axes[2].imshow(masked_heatmap, cmap='jet', alpha=0.65,
                          vmin=overlay_min, vmax=overlay_max)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Label: above/below threshold
    if max_score > threshold:
        status = f"ANOMALY (max={max_score:.3f})"
        color = 'red'
    else:
        status = f"NORMAL (max={max_score:.3f})"
        color = 'green'
    axes[2].set_title(f"Overlay: {status}", fontsize=14, color=color, fontweight='bold')
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v5b SPATIAL | {TARGET_SPEC}/group_1 | {img_path.name} | "
                 f"Gaussian σ={GAUSSIAN_SIGMA}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return max_score, mean_score


def get_image_paths(folder, cam_id, subsample_step=1):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.exists():
        return []
    images = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
    if TRIM_HEAD + TRIM_TAIL < len(images):
        images = images[TRIM_HEAD: len(images) - TRIM_TAIL]
    return images[::subsample_step]


def main():
    print("=" * 60)
    print("PatchCore v5b - IMPROVED HEATMAP INFERENCE")
    print("=" * 60)

    # Load saved model
    model_dir = OUTPUT_DIR / TARGET_SPEC / "group_1"
    bank_path = model_dir / "memory_bank.npy"
    size_path = model_dir / "spatial_size.npy"

    if not bank_path.exists():
        print(f"ERROR: {bank_path} not found!")
        return

    memory_bank = np.load(bank_path)
    spatial_size = tuple(np.load(size_path))
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Spatial size: {spatial_size}")
    sys.stdout.flush()

    # Load threshold from previous training
    # Recalculate from bank stats or use saved value
    # From training: median=1.8581, threshold=2.3199
    threshold = 2.3199
    print(f"  Threshold: {threshold}")

    # Load extractor
    print("\nLoading WideResNet50...")
    extractor = SpatialFeatureExtractor(device="cuda")
    sys.stdout.flush()

    # Find defect folder
    defect_folder = None
    for d in sorted(NAS_ROOT.glob("20250630/*")):
        if d.is_dir() and DEFECT_FOLDER_PREFIX in d.name:
            defect_folder = d
            break

    if not defect_folder:
        print(f"ERROR: defect folder {DEFECT_FOLDER_PREFIX} not found!")
        return

    print(f"\nInference: {defect_folder.name}")

    cam1_images = get_image_paths(defect_folder, cam_id=1, subsample_step=1)
    print(f"  Camera 1 images: {len(cam1_images)}")

    heatmap_dir = model_dir / "heatmaps_v2" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Sample: first 5, last 5, every 50th, quarter points
    n = len(cam1_images)
    sample_idx = set()
    sample_idx.update(range(min(5, n)))
    sample_idx.update(range(max(0, n - 5), n))
    sample_idx.update(range(0, n, 50))  # denser sampling
    sample_idx.update(range(n // 4, min(n // 4 + 3, n)))
    sample_idx.update(range(n // 2, min(n // 2 + 3, n)))
    sample_idx.update(range(3 * n // 4, min(3 * n // 4 + 3, n)))
    sample_indices = sorted(sample_idx)

    print(f"  Generating improved heatmaps for {len(sample_indices)} images...")
    print(f"  Gaussian sigma: {GAUSSIAN_SIGMA}")
    sys.stdout.flush()

    results = []
    for idx in tqdm(sample_indices, desc="Heatmaps v2"):
        img_path = cam1_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_score, mean_score = generate_heatmap_v2(
            img_path, memory_bank, extractor, spatial_size, out_path, threshold
        )
        if max_score is not None:
            results.append({
                "idx": idx,
                "file": img_path.name,
                "max_score": max_score,
                "mean_score": mean_score,
                "anomaly": max_score > threshold,
            })

    with open(model_dir / "inference_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    all_max = [r["max_score"] for r in results]
    n_anomaly = sum(1 for r in results if r["anomaly"])
    print(f"\n{'='*60}")
    print(f"  Images: {len(results)}")
    print(f"  Anomaly: {n_anomaly}/{len(results)} (threshold={threshold:.4f})")
    print(f"  Score range: {min(all_max):.4f} ~ {max(all_max):.4f}")
    print(f"  Heatmaps saved to: {heatmap_dir}")
    print(f"{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
