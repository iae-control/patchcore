#!/usr/bin/env python3
"""v9 FP test — run hold-out normal images through each camera model.
Training used every 3rd image (indices 0,3,6,...).
Validation uses every 3rd image offset by 1 (indices 1,4,7,...).
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
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from collections import defaultdict

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v9_percam/596x199")
TARGET_SPEC = "596x199"
TRAIN_DATES = ["20250831", "20251027"]
DEFECT_DATES = {"20250630"}
DEFECT_FOLDER_PREFIX = "20250630160852"

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1200
TILE_W, TILE_H = 192, 300
N_COLS, N_ROWS = 10, 4
TRIM_HEAD, TRIM_TAIL = 100, 100
BATCH_SIZE = 64
Z_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
SUBSAMPLE_STEP = 3  # same as training
VAL_OFFSET = 1       # offset so we skip training images

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]

def get_tile_positions():
    positions = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            tx = col * TILE_W
            ty = row * TILE_H
            positions.append((tx, ty, col, row))
    return positions

def get_holdout_images(folder, cam_id):
    """Get images NOT used in training (offset by 1 from subsample)."""
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
    # Training used indices 0, 3, 6, 9, ...
    # Validation uses indices 1, 4, 7, 10, ...
    holdout = images[VAL_OFFSET::SUBSAMPLE_STEP]
    return holdout

# ===== BACKBONE =====
class TileFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        weights_path = Path("/home/dk-sdd/patchcore/models/wide_resnet50_2.pth")
        if weights_path.exists():
            print(f"  Loading WideResNet50 from local cache")
            model = wide_resnet50_2(weights=None)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            model = wide_resnet50_2(weights="IMAGENET1K_V1")
        model.eval()
        self.layer2 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                     model.layer1, model.layer2)
        self.layer3 = nn.Sequential(model.layer3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to(device)

    @torch.no_grad()
    def __call__(self, batch):
        batch = batch.to(self.device)
        f2 = self.layer2(batch)
        f3 = self.layer3(f2)
        f2_pooled = self.pool(f2).flatten(1)
        f3_pooled = self.pool(f3).flatten(1)
        return torch.cat([f2_pooled, f3_pooled], dim=1)

def score_tiles(features, memory_bank):
    dists = np.linalg.norm(features[:, None, :] - memory_bank[None, :, :], axis=2)
    return dists.min(axis=1)


# ===== DISCOVER FOLDERS =====
def discover_val_folders():
    """Find training-date folders with camera images for hold-out validation."""
    folders = []
    for date_str in TRAIN_DATES:
        date_dir = NAS_ROOT / date_str
        if not date_dir.is_dir():
            continue
        for sub in sorted(date_dir.iterdir()):
            if sub.is_dir() and TARGET_SPEC in sub.name:
                if DEFECT_FOLDER_PREFIX in sub.name:
                    continue  # skip defect folder
                if (sub / "camera_1").is_dir():
                    folders.append((date_str, sub))
    return folders


# ===== MAIN =====
def main():
    print("=" * 60)
    print("v9 FP Test — Hold-out Normal Image False Positive Rate")
    print(f"Using training dates with offset={VAL_OFFSET}, step={SUBSAMPLE_STEP}")
    print("=" * 60)

    all_positions = get_tile_positions()

    # Discover validation folders
    val_folders = discover_val_folders()
    print(f"\nValidation folders: {len(val_folders)}")
    for d, f in val_folders[:5]:
        print(f"  {d}/{f.name}")
    if len(val_folders) > 5:
        print(f"  ... and {len(val_folders)-5} more")

    # Load backbone
    print("\nLoading backbone...")
    extractor = TileFeatureExtractor()

    # Per-camera FP test
    cameras = list(range(1, 11))
    all_cam_results = {}

    for cam_id in cameras:
        cam_dir = MODEL_DIR / f"camera_{cam_id}"
        if not cam_dir.exists():
            print(f"\nCamera {cam_id}: no model, skip")
            continue

        print(f"\n{'='*60}")
        print(f"  CAMERA {cam_id}")
        print(f"{'='*60}")

        # Load model
        memory_bank = np.load(cam_dir / "memory_bank.npy")
        pos_mean = np.load(cam_dir / "pos_mean.npy")
        pos_std = np.load(cam_dir / "pos_std.npy")
        print(f"  Memory bank: {memory_bank.shape}")

        # Collect hold-out images
        all_images = []
        for date_str, folder in val_folders:
            imgs = get_holdout_images(folder, cam_id)
            all_images.extend(imgs)

        print(f"  Hold-out normal images: {len(all_images)}")
        if not all_images:
            continue

        # Subsample if too many (max 5000 for speed)
        if len(all_images) > 5000:
            step = len(all_images) // 5000
            all_images = all_images[::step]
            print(f"  Subsampled to: {len(all_images)}")

        # Run inference
        max_zs = []
        mean_zs = []
        detected_counts = {th: 0 for th in Z_THRESHOLDS}
        per_image_results = []

        t0 = time.time()
        for i, img_path in enumerate(all_images):
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
            raw_scores = score_tiles(feats, memory_bank)

            z_scores = (raw_scores - pos_mean[tile_idxs]) / (pos_std[tile_idxs] + 1e-8)

            max_z = float(np.max(z_scores))
            mean_z = float(np.mean(z_scores))
            max_zs.append(max_z)
            mean_zs.append(mean_z)

            for th in Z_THRESHOLDS:
                if max_z > th:
                    detected_counts[th] += 1

            # Track worst offenders
            if max_z > 2.5:
                per_image_results.append({
                    "file": str(img_path),
                    "max_z": round(max_z, 3),
                    "mean_z": round(mean_z, 3),
                })

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{len(all_images)} done ({elapsed:.0f}s)")

        n_total = len(max_zs)
        elapsed = time.time() - t0
        print(f"  Processed: {n_total} images in {elapsed:.0f}s")
        print(f"  max_z stats: min={min(max_zs):.2f}, median={np.median(max_zs):.2f}, "
              f"p95={np.percentile(max_zs, 95):.2f}, p99={np.percentile(max_zs, 99):.2f}, max={max(max_zs):.2f}")

        print(f"  FP rates (image flagged if max_z > threshold):")
        cam_fp = {}
        for th in Z_THRESHOLDS:
            fp_rate = detected_counts[th] / n_total * 100
            print(f"    z>{th}: {detected_counts[th]}/{n_total} ({fp_rate:.1f}%)")
            cam_fp[str(th)] = {"fp_count": detected_counts[th], "fp_rate": round(fp_rate, 2)}

        if per_image_results:
            per_image_results.sort(key=lambda x: x["max_z"], reverse=True)
            print(f"  Top 5 worst FP (z>2.5):")
            for r in per_image_results[:5]:
                print(f"    {Path(r['file']).name}: max_z={r['max_z']}")

        all_cam_results[cam_id] = {
            "n_images": n_total,
            "max_z_min": round(float(min(max_zs)), 3),
            "max_z_median": round(float(np.median(max_zs)), 3),
            "max_z_p95": round(float(np.percentile(max_zs, 95)), 3),
            "max_z_p99": round(float(np.percentile(max_zs, 99)), 3),
            "max_z_max": round(float(max(max_zs)), 3),
            "fp_by_threshold": cam_fp,
            "worst_fp": per_image_results[:20],
        }

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — FP Rate per Camera (hold-out normal)")
    print(f"{'='*60}")
    print(f"{'Cam':>4} {'N':>6} {'med_z':>7} {'p95_z':>7} {'p99_z':>7} {'max_z':>7}", end="")
    for th in Z_THRESHOLDS:
        print(f" {'FP@'+str(th):>7}", end="")
    print()

    for cam_id in cameras:
        if cam_id not in all_cam_results:
            continue
        r = all_cam_results[cam_id]
        print(f"{cam_id:>4} {r['n_images']:>6} {r['max_z_median']:>7.2f} {r['max_z_p95']:>7.2f} "
              f"{r['max_z_p99']:>7.2f} {r['max_z_max']:>7.2f}", end="")
        for th in Z_THRESHOLDS:
            fp = r['fp_by_threshold'][str(th)]
            print(f" {fp['fp_rate']:>6.1f}%", end="")
        print()

    # Save
    out_path = MODEL_DIR / "fp_test_results.json"
    with open(out_path, "w") as f:
        json.dump(all_cam_results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
