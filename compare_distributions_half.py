#!/usr/bin/env python3
"""
Compare z-score distributions: DEFECT date vs NORMAL date
Full-resolution model (output_v5b_half), group 1
Outputs: overlapping histogram + stats
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, time, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import wide_resnet50_2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v5b_half/596x199/group_1")
OUTPUT_PATH = Path("/home/dk-sdd/patchcore/output_v5b_half/596x199/group_1/dist_compare_half.png")

SPEC = "596x199"
CAM_IDS = [1, 10]
MIRROR_CAMS = {10}
TRIM_HEAD = 100
TRIM_TAIL = 100

IMAGE_WIDTH_ORIG = 1920
IMAGE_HEIGHT_ORIG = 1200
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 600
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

DEFECT_DATE = "20250630"
NORMAL_DATE = "20250831"
DEFECT_PREFIX = "20250630160852"

# Sample settings - enough for good histogram
MAX_NORMAL_FOLDERS = 10
NORMAL_SUBSAMPLE = 5  # every 5th image
DEFECT_SUBSAMPLE = 1  # all defect images (already have results but re-score for consistency)

TRANSFORM = transforms.Compose([
    transforms.Resize((600, 960), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        local_weights = cache_dir / "wide_resnet50_2-95faca4d.pth"
        backbone = wide_resnet50_2(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict)

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


def score_spatial(features, memory_bank, batch_size=4096):
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(memory_bank).cuda()
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i+batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        min_d, _ = dists.min(dim=1)
        scores[i:i+batch_size] = min_d.cpu().numpy()
    return scores


def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def get_max_z_scores(folders, cam_ids, extractor, memory_bank, spatial_size,
                     pos_mean, pos_std, subsample_step=1, label=""):
    """Score images, return list of per-image max z-scores."""
    Hp, Wp = spatial_size
    z_scores_list = []
    img_count = 0

    for fi, folder in enumerate(folders):
        folder_count = 0
        for cam_id in cam_ids:
            cam_dir = folder / f"camera_{cam_id}"
            if not cam_dir.is_dir():
                continue
            images = sorted(
                [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                key=natural_sort_key
            )
            if len(images) <= TRIM_HEAD + TRIM_TAIL:
                continue
            images = images[TRIM_HEAD:len(images) - TRIM_TAIL]
            if subsample_step > 1:
                images = images[::subsample_step]

            mirror = cam_id in MIRROR_CAMS
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    continue
                if img.size != (IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG):
                    continue
                if mirror:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                tensor = TRANSFORM(img).unsqueeze(0)
                features, _ = extractor.extract_spatial(tensor)
                scores = score_spatial(features, memory_bank)
                z = (scores - pos_mean) / (pos_std + 1e-6)
                max_z = float(z.max())
                z_scores_list.append(max_z)
                folder_count += 1
                img_count += 1

        if (fi + 1) % 5 == 0 or fi == len(folders) - 1:
            print(f"  [{label}] {fi+1}/{len(folders)} folders, {img_count} images")

    return z_scores_list


def main():
    device = "cuda:0"
    print("Loading model...")
    memory_bank = np.load(MODEL_DIR / "memory_bank.npy")
    spatial_size = np.load(MODEL_DIR / "spatial_size.npy").tolist()
    pos_mean = np.load(MODEL_DIR / "pos_mean.npy")
    pos_std = np.load(MODEL_DIR / "pos_std.npy")
    with open(MODEL_DIR / "training_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold_mad"]
    print(f"Bank: {memory_bank.shape}, Spatial: {spatial_size}, Threshold: {threshold:.4f}")

    extractor = SpatialFeatureExtractor(device)
    print("Backbone loaded\n")

    # === DEFECT date ===
    print(f"=== DEFECT DATE ({DEFECT_DATE}) ===")
    # Use existing results if available
    results_path = MODEL_DIR / "inference_results_full.json"
    if results_path.exists():
        with open(results_path) as f:
            defect_results = json.load(f)
        defect_z = [r["max_z_score"] for r in defect_results]
        print(f"  Loaded {len(defect_z)} existing results")
    else:
        defect_dir = NAS_ROOT / DEFECT_DATE
        defect_folders = sorted([d for d in defect_dir.iterdir()
                                  if d.is_dir() and SPEC in d.name and DEFECT_PREFIX in d.name])
        print(f"  {len(defect_folders)} defect folders")
        defect_z = get_max_z_scores(defect_folders, CAM_IDS, extractor, memory_bank,
                                     spatial_size, pos_mean, pos_std, DEFECT_SUBSAMPLE, "DEFECT")

    # === NORMAL date ===
    print(f"\n=== NORMAL DATE ({NORMAL_DATE}) ===")
    normal_dir = NAS_ROOT / NORMAL_DATE
    normal_folders = sorted([d for d in normal_dir.iterdir()
                              if d.is_dir() and SPEC in d.name])[:MAX_NORMAL_FOLDERS]
    print(f"  {len(normal_folders)} normal folders, subsample={NORMAL_SUBSAMPLE}")
    normal_z = get_max_z_scores(normal_folders, CAM_IDS, extractor, memory_bank,
                                 spatial_size, pos_mean, pos_std, NORMAL_SUBSAMPLE, "NORMAL")

    # === PLOT ===
    print(f"\n=== RESULTS ===")
    print(f"Defect: {len(defect_z)} images, z: {np.min(defect_z):.2f} ~ {np.max(defect_z):.2f}, "
          f"mean={np.mean(defect_z):.2f}, median={np.median(defect_z):.2f}")
    print(f"Normal: {len(normal_z)} images, z: {np.min(normal_z):.2f} ~ {np.max(normal_z):.2f}, "
          f"mean={np.mean(normal_z):.2f}, median={np.median(normal_z):.2f}")
    print(f"Threshold: {threshold:.4f}")

    # Overlap analysis
    defect_below = sum(1 for z in defect_z if z <= threshold)
    normal_above = sum(1 for z in normal_z if z > threshold)
    print(f"\nDefect below threshold (missed): {defect_below}/{len(defect_z)} ({100*defect_below/len(defect_z):.1f}%)")
    print(f"Normal above threshold (false alarm): {normal_above}/{len(normal_z)} ({100*normal_above/len(normal_z):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Left: overlapping histogram
    ax = axes[0]
    bins = np.linspace(0, max(max(defect_z), max(normal_z)) * 1.05, 80)
    ax.hist(normal_z, bins=bins, alpha=0.6, color='#2196F3', label=f'Normal ({NORMAL_DATE}) n={len(normal_z)}', density=True)
    ax.hist(defect_z, bins=bins, alpha=0.6, color='#F44336', label=f'Defect ({DEFECT_DATE}) n={len(defect_z)}', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax.set_xlabel('Max Z-Score (per image)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Z-Score Distribution: Defect vs Normal', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: box plot
    ax2 = axes[1]
    bp = ax2.boxplot([normal_z, defect_z], labels=[f'Normal\n({NORMAL_DATE})', f'Defect\n({DEFECT_DATE})'],
                      patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#F44336')
    bp['boxes'][1].set_alpha(0.6)
    ax2.axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax2.set_ylabel('Max Z-Score', fontsize=13)
    ax2.set_title('Z-Score Box Plot', fontsize=15)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'PatchCore v5b-HALF (960x600) | 596x199/group_1 | Defect vs Normal', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")

    # Also save as JSON for reference
    json_path = MODEL_DIR / "dist_compare_half.json"
    with open(json_path, "w") as f:
        json.dump({
            "defect_date": DEFECT_DATE, "normal_date": NORMAL_DATE,
            "threshold": threshold,
            "defect_z": defect_z, "normal_z": normal_z,
            "defect_count": len(defect_z), "normal_count": len(normal_z),
        }, f)
    print(f"Saved: {json_path}")
    print("DONE")


if __name__ == "__main__":
    main()
