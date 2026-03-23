#!/usr/bin/env python3
"""v9 FP test — RAW score (no z-score normalization).
Uses original PatchCore approach: raw NN distance as anomaly score.
Tests both normal (hold-out) and defect images, computes AUROC + F1.
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
SUBSAMPLE_STEP = 3
VAL_OFFSET = 1

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
            positions.append((col * TILE_W, row * TILE_H, col, row))
    return positions

def get_holdout_images(folder, cam_id):
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
    return images[VAL_OFFSET::SUBSAMPLE_STEP]

def get_image_paths(folder, cam_id):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')],
        key=natural_sort_key
    )
    if len(images) <= TRIM_HEAD + TRIM_TAIL:
        return []
    return images[TRIM_HEAD:len(images) - TRIM_TAIL]

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

def score_tiles_gpu(features, memory_bank, device="cuda"):
    """GPU-accelerated NN distance computation."""
    feat_t = torch.from_numpy(features).float().to(device)
    bank_t = torch.from_numpy(memory_bank).float().to(device)
    # Chunked to avoid OOM
    scores = []
    chunk = 256
    for i in range(0, len(feat_t), chunk):
        d = torch.cdist(feat_t[i:i+chunk], bank_t)
        scores.append(d.min(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def score_image(img, extractor, memory_bank, all_positions, device="cuda"):
    """Score a single image, return per-tile raw scores."""
    tiles = []
    for tx, ty, col, row in all_positions:
        tile = img.crop((tx, ty, tx + TILE_W, ty + TILE_H))
        tiles.append(TRANSFORM(tile))

    all_feats = []
    for bs in range(0, len(tiles), BATCH_SIZE):
        batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    feats = np.concatenate(all_feats, axis=0)
    raw_scores = score_tiles_gpu(feats, memory_bank, device)
    return raw_scores


def process_images(image_list, extractor, memory_bank, all_positions, label, max_images=3000):
    """Process a list of images, return (max_score, mean_score, label) per image."""
    if len(image_list) > max_images:
        step = len(image_list) // max_images
        image_list = image_list[::step][:max_images]

    results = []
    t0 = time.time()
    for i, img_path in enumerate(image_list):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            continue

        raw_scores = score_image(img, extractor, memory_bank, all_positions)
        results.append({
            "max_score": float(np.max(raw_scores)),
            "mean_score": float(np.mean(raw_scores)),
            "top3_mean": float(np.mean(np.sort(raw_scores)[-3:])),
            "label": label,
        })

        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(image_list)} ({time.time()-t0:.0f}s)")

    return results


def compute_metrics(normal_results, defect_results):
    """Compute AUROC, F1, FP rates at various thresholds."""
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

    all_results = normal_results + defect_results
    labels = np.array([r["label"] for r in all_results])

    metrics = {}
    for score_key in ["max_score", "mean_score", "top3_mean"]:
        scores = np.array([r[score_key] for r in all_results])

        # AUROC
        try:
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = 0.0

        # Find best F1
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1s)
        best_f1 = float(f1s[best_idx])
        best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0

        # Normal stats
        normal_scores = np.array([r[score_key] for r in normal_results])
        defect_scores = np.array([r[score_key] for r in defect_results])

        # FP rates at various percentile thresholds
        fp_results = {}
        for pct in [95, 99, 99.5]:
            th = np.percentile(normal_scores, pct)
            fp_count = int(np.sum(normal_scores > th))
            det_count = int(np.sum(defect_scores > th))
            fp_results[f"p{pct}"] = {
                "threshold": round(float(th), 4),
                "fp_rate": round(fp_count / len(normal_scores) * 100, 2),
                "det_rate": round(det_count / len(defect_scores) * 100, 2),
            }

        metrics[score_key] = {
            "auroc": round(auroc, 4),
            "best_f1": round(best_f1, 4),
            "best_f1_threshold": round(best_th, 4),
            "normal_min": round(float(normal_scores.min()), 4),
            "normal_median": round(float(np.median(normal_scores)), 4),
            "normal_p95": round(float(np.percentile(normal_scores, 95)), 4),
            "normal_max": round(float(normal_scores.max()), 4),
            "defect_min": round(float(defect_scores.min()), 4),
            "defect_median": round(float(np.median(defect_scores)), 4),
            "defect_max": round(float(defect_scores.max()), 4),
            "fp_at_percentile": fp_results,
        }

    return metrics


# ===== MAIN =====
def main():
    print("=" * 60)
    print("v9 FP Test — RAW Score (no z-score)")
    print("AUROC + F1 + FP rate evaluation")
    print("=" * 60)

    all_positions = get_tile_positions()

    # Discover folders
    val_folders = []
    defect_folder = None
    for date_str in TRAIN_DATES:
        date_dir = NAS_ROOT / date_str
        if not date_dir.is_dir():
            continue
        for sub in sorted(date_dir.iterdir()):
            if sub.is_dir() and TARGET_SPEC in sub.name:
                if DEFECT_FOLDER_PREFIX in sub.name:
                    continue
                if (sub / "camera_1").is_dir():
                    val_folders.append(sub)

    # Find defect folder
    defect_date_dir = NAS_ROOT / "20250630"
    if defect_date_dir.is_dir():
        for sub in sorted(defect_date_dir.iterdir()):
            if sub.is_dir() and DEFECT_FOLDER_PREFIX in sub.name:
                defect_folder = sub
                break

    print(f"Validation folders: {len(val_folders)}")
    print(f"Defect folder: {defect_folder.name if defect_folder else 'None'}")

    # Load backbone
    print("\nLoading backbone...")
    extractor = TileFeatureExtractor()

    cameras = list(range(1, 11))
    all_cam_metrics = {}

    for cam_id in cameras:
        cam_dir = MODEL_DIR / f"camera_{cam_id}"
        if not cam_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  CAMERA {cam_id}")
        print(f"{'='*60}")

        memory_bank = np.load(cam_dir / "memory_bank.npy")
        print(f"  Memory bank: {memory_bank.shape}")

        # Collect hold-out normal images
        normal_images = []
        for folder in val_folders:
            normal_images.extend(get_holdout_images(folder, cam_id))
        print(f"  Normal images: {len(normal_images)}")

        # Collect defect images
        defect_images = []
        if defect_folder:
            defect_images = get_image_paths(defect_folder, cam_id)
        print(f"  Defect images: {len(defect_images)}")

        # Process normal
        print(f"  Scoring normal...")
        normal_results = process_images(normal_images, extractor, memory_bank, all_positions, label=0, max_images=3000)

        # Process defect
        print(f"  Scoring defect...")
        defect_results = process_images(defect_images, extractor, memory_bank, all_positions, label=1, max_images=3000)

        print(f"  Normal scored: {len(normal_results)}, Defect scored: {len(defect_results)}")

        if normal_results and defect_results:
            metrics = compute_metrics(normal_results, defect_results)

            for key in ["max_score", "mean_score", "top3_mean"]:
                m = metrics[key]
                print(f"\n  [{key}]")
                print(f"    AUROC: {m['auroc']}")
                print(f"    Best F1: {m['best_f1']} (th={m['best_f1_threshold']})")
                print(f"    Normal:  min={m['normal_min']}, median={m['normal_median']}, p95={m['normal_p95']}, max={m['normal_max']}")
                print(f"    Defect:  min={m['defect_min']}, median={m['defect_median']}, max={m['defect_max']}")
                for pk, pv in m['fp_at_percentile'].items():
                    print(f"    @{pk}: th={pv['threshold']}, FP={pv['fp_rate']}%, Det={pv['det_rate']}%")

            all_cam_metrics[cam_id] = metrics
        else:
            print(f"  Skipped (no data)")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Cam':>4} {'AUROC_max':>10} {'AUROC_mean':>11} {'AUROC_top3':>11} {'F1_max':>7} {'F1_mean':>8} {'F1_top3':>8}")
    for cam_id in cameras:
        if cam_id not in all_cam_metrics:
            continue
        m = all_cam_metrics[cam_id]
        print(f"{cam_id:>4} {m['max_score']['auroc']:>10.4f} {m['mean_score']['auroc']:>11.4f} "
              f"{m['top3_mean']['auroc']:>11.4f} {m['max_score']['best_f1']:>7.4f} "
              f"{m['mean_score']['best_f1']:>8.4f} {m['top3_mean']['best_f1']:>8.4f}")

    out_path = MODEL_DIR / "fp_test_raw_results.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_cam_metrics.items()}, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
