#!/usr/bin/env python3
"""Experiment 8: Spatial Model + Ens-MAX (compare with tile-based exp5b).

Spatial (v5b) backbone: full image -> WideResNet50 layer2+layer3 -> AvgPool(3,3) -> 50x80 grid
Per-position z-score + 11 metrics Ens-MAX.

Compares spatial vs tile approach under same Ens-MAX framework.
Tests k=1 and k=5 for k-NN scoring.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys, json, re, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from scipy import ndimage, stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

SPATIAL_DIR = Path("/home/dk-sdd/patchcore/output_exp2_spatial_0630/596x199/group_1")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp8_spatial_ensmax")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

TRIM_HEAD = 100
TRIM_TAIL = 100
BRIGHTNESS_THRESHOLD = 30
NORM_SAMPLE_IMAGES = 300

GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10
BATCH_SIZE = 8  # Full images, not tiles

# Z-score threshold for anomaly count
ZSCORE_THRESHOLDS = [2.5, 3.0, 3.5]
K_VALUES = [1, 5]

METRIC_NAMES = [
    "tile_max_raw", "tile_mean_raw", "tile_p95_raw",
    "tile_max_z", "tile_mean_z", "tile_p95_z",
    "anomaly_count", "cluster_score",
    "max_vertical_run", "z_skewness", "z_kurtosis",
]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
        return features.float()  # (B, 1536, H, W)


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


def load_and_preprocess(img_path, cam_id):
    """Load image, convert grayscale->RGB, mirror if cam_10."""
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    if cam_id == MIRROR_CAM:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(img)
    if arr.mean() < BRIGHTNESS_THRESHOLD:
        return None
    return TRANSFORM(img)


def score_spatial_knn(features, memory_bank, k=1):
    """Score spatial features against memory bank with k-NN.
    features: (N_positions, 1536)
    memory_bank: (M, 1536) coreset
    Returns: (N_positions,) distances
    """
    bank_t = torch.from_numpy(memory_bank).cuda()
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    batch_size = 4096

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i + batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        if k == 1:
            min_d, _ = dists.min(dim=1)
            scores[i:i + batch_size] = min_d.cpu().numpy()
        else:
            topk_d, _ = dists.topk(k, dim=1, largest=False)
            scores[i:i + batch_size] = topk_d.mean(dim=1).cpu().numpy()

    return scores


def compute_cluster_score(z_grid, threshold):
    """Spatial clustering of anomalous positions."""
    binary = (z_grid > threshold).astype(np.int32)
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters == 0:
        return 0.0
    sizes = ndimage.sum(binary, labeled, range(1, n_clusters + 1))
    return float(max(sizes)) if len(sizes) > 0 else 0.0


def extract_11_metrics(pos_scores, pos_mean, pos_std, n_h, n_w, zscore_th):
    """Extract 11 metrics from spatial position scores."""
    valid_mask = ~np.isnan(pos_scores)
    raw = pos_scores.copy()
    raw[~valid_mask] = 0.0

    safe_std = np.where(pos_std > 1e-6, pos_std, 1.0)
    z = np.where(valid_mask, (pos_scores - pos_mean) / safe_std, 0.0)

    valid_raw = raw[valid_mask]
    valid_z = z[valid_mask]

    if len(valid_raw) == 0:
        return np.zeros(11, dtype=np.float32)

    metrics = np.zeros(11, dtype=np.float32)

    # 1-3: Raw distance metrics
    metrics[0] = float(np.max(valid_raw))        # max_raw
    metrics[1] = float(np.mean(valid_raw))        # mean_raw
    metrics[2] = float(np.percentile(valid_raw, 95))  # p95_raw

    # 4-6: Z-score metrics
    metrics[3] = float(np.max(valid_z))           # max_z
    metrics[4] = float(np.mean(valid_z))          # mean_z
    metrics[5] = float(np.percentile(valid_z, 95))  # p95_z

    # 7: Anomaly count (z > threshold)
    metrics[6] = float(np.sum(valid_z > zscore_th))

    # 8: Cluster score
    z_grid = z.reshape(n_h, n_w)
    metrics[7] = compute_cluster_score(z_grid, zscore_th)

    # 9: Max vertical run
    binary_grid = (z_grid > zscore_th).astype(np.int32)
    max_vrun = 0
    for col in range(binary_grid.shape[1]):
        run = 0
        for row in range(binary_grid.shape[0]):
            if binary_grid[row, col]:
                run += 1
                max_vrun = max(max_vrun, run)
            else:
                run = 0
    metrics[8] = float(max_vrun)

    # 10-11: Z-score distribution shape
    if len(valid_z) > 3:
        metrics[9] = float(scipy_stats.skew(valid_z))     # z_skewness
        metrics[10] = float(scipy_stats.kurtosis(valid_z))  # z_kurtosis

    return metrics


def extract_spatial_features_and_score(img_tensor, extractor, memory_bank, k=1):
    """Extract spatial features from single image, score against memory bank."""
    feat_map = extractor(img_tensor.unsqueeze(0))  # (1, 1536, H, W)
    B, C, H, W = feat_map.shape
    features = feat_map.permute(0, 2, 3, 1).reshape(H * W, C)
    features_np = features.cpu().numpy()

    pos_scores = score_spatial_knn(features_np, memory_bank, k=k)
    return pos_scores, (H, W)


def main():
    print("=" * 60)
    print("Exp8: Spatial Model + Ens-MAX")
    print("=" * 60)

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spatial memory bank
    print("\n[1] Loading spatial memory bank...")
    mb_path = SPATIAL_DIR / "memory_bank.npy"
    memory_bank = np.load(mb_path)
    print(f"  Memory bank: {memory_bank.shape}")

    # Initialize extractor
    print("\n[2] Initializing spatial feature extractor...")
    extractor = SpatialFeatureExtractor(device="cuda")

    # Discover folders
    print("\n[3] Discovering 0630 folders...")
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)}, Defect: {defect_folder.name if defect_folder else 'N/A'}")

    # ========== COMPUTE POSITION STATS ==========
    print(f"\n[4] Computing per-position stats from {NORM_SAMPLE_IMAGES} normal images...")

    # First pass: determine spatial grid size
    test_paths = []
    for folder in normal_folders:
        for cam_id in CAM_IDS:
            paths = get_image_paths(folder, cam_id, 10)
            if paths:
                test_paths.extend(paths[:2])
                break
        if test_paths:
            break

    test_tensor = load_and_preprocess(test_paths[0], CAM_IDS[0])
    feat_map = extractor(test_tensor.unsqueeze(0))
    _, _, sp_H, sp_W = feat_map.shape
    n_positions = sp_H * sp_W
    print(f"  Spatial grid: {sp_W}x{sp_H} = {n_positions} positions")

    # Collect normal image paths
    all_normal_paths = []
    for folder in normal_folders:
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, 3)
            all_normal_paths.extend([(p, cam_id) for p in images])
    np.random.seed(42)
    np.random.shuffle(all_normal_paths)

    # Compute position stats for each k value
    results_all = {}

    for k_val in K_VALUES:
        print(f"\n{'='*50}")
        print(f"  k-NN k={k_val}")
        print(f"{'='*50}")

        score_accumulator = []
        count = 0

        for img_path, cam_id in all_normal_paths:
            if count >= NORM_SAMPLE_IMAGES:
                break
            tensor = load_and_preprocess(img_path, cam_id)
            if tensor is None:
                continue
            pos_scores, (h, w) = extract_spatial_features_and_score(
                tensor, extractor, memory_bank, k=k_val)
            score_accumulator.append(pos_scores)
            count += 1
            if count % 100 == 0:
                print(f"    pos_stats: {count}/{NORM_SAMPLE_IMAGES}")

        score_matrix = np.array(score_accumulator)  # (N_images, N_positions)
        pos_mean = score_matrix.mean(axis=0)
        pos_std = score_matrix.std(axis=0)
        pos_std = np.where(pos_std < 1e-6, 1.0, pos_std)
        print(f"  pos_mean range: [{pos_mean.min():.3f}, {pos_mean.max():.3f}]")
        print(f"  pos_std range: [{pos_std.min():.3f}, {pos_std.max():.3f}]")

        # ========== COMPUTE NORMAL CALIBRATION METRICS ==========
        print(f"\n  Computing metrics for normal cal images (k={k_val})...")
        cal_normal_paths = all_normal_paths[:200]

        for zscore_th in ZSCORE_THRESHOLDS:
            config_key = f"k{k_val}_th{zscore_th}"
            print(f"\n  --- {config_key} ---")

            normal_cal_metrics = []
            for img_path, cam_id in cal_normal_paths:
                tensor = load_and_preprocess(img_path, cam_id)
                if tensor is None:
                    continue
                pos_scores, (h, w) = extract_spatial_features_and_score(
                    tensor, extractor, memory_bank, k=k_val)
                m = extract_11_metrics(pos_scores, pos_mean, pos_std, h, w, zscore_th)
                normal_cal_metrics.append(m)

            normal_cal = np.array(normal_cal_metrics)
            ens_mean = normal_cal.mean(axis=0)
            ens_std = normal_cal.std(axis=0)
            ens_std = np.where(ens_std < 1e-8, 1.0, ens_std)

            # ========== SCORE DEFECT IMAGES ==========
            print(f"  Scoring defect images...")
            defect_results = []
            if defect_folder:
                for cam_id in CAM_IDS:
                    images = get_image_paths(defect_folder, cam_id)
                    for img_path in images:
                        tensor = load_and_preprocess(img_path, cam_id)
                        if tensor is None:
                            continue
                        pos_scores, (h, w) = extract_spatial_features_and_score(
                            tensor, extractor, memory_bank, k=k_val)
                        m = extract_11_metrics(pos_scores, pos_mean, pos_std, h, w, zscore_th)
                        z_metrics = (m - ens_mean) / ens_std
                        ens_score = float(np.max(z_metrics))
                        defect_results.append({
                            "file": img_path.name, "cam": cam_id,
                            "ens_score": ens_score, "metrics": m.tolist(),
                        })
                        if len(defect_results) % 200 == 0:
                            print(f"    defect: {len(defect_results)}")
            print(f"  Defect: {len(defect_results)} images")

            # ========== SCORE NORMAL TEST IMAGES ==========
            print(f"  Scoring normal test images...")
            normal_results = []
            test_normal_paths = all_normal_paths[200:]
            for folder in normal_folders[:5]:
                for cam_id in CAM_IDS:
                    images = get_image_paths(folder, cam_id, 10)
                    for img_path in images:
                        if (img_path, cam_id) in cal_normal_paths:
                            continue
                        tensor = load_and_preprocess(img_path, cam_id)
                        if tensor is None:
                            continue
                        pos_scores, (h, w) = extract_spatial_features_and_score(
                            tensor, extractor, memory_bank, k=k_val)
                        m = extract_11_metrics(pos_scores, pos_mean, pos_std, h, w, zscore_th)
                        z_metrics = (m - ens_mean) / ens_std
                        ens_score = float(np.max(z_metrics))
                        normal_results.append({
                            "file": img_path.name, "cam": cam_id,
                            "ens_score": ens_score, "metrics": m.tolist(),
                        })
            print(f"  Normal test: {len(normal_results)} images")

            # ========== EVALUATE ==========
            defect_scores = np.array([r["ens_score"] for r in defect_results])
            normal_scores = np.array([r["ens_score"] for r in normal_results])

            print(f"\n  === {config_key} Results ===")
            print(f"  Defect scores: median={np.median(defect_scores):.3f}, "
                  f"mean={np.mean(defect_scores):.3f}, "
                  f"p25={np.percentile(defect_scores, 25):.3f}")
            print(f"  Normal scores: median={np.median(normal_scores):.3f}, "
                  f"mean={np.mean(normal_scores):.3f}, "
                  f"p75={np.percentile(normal_scores, 75):.3f}")
            print(f"  Separation: {np.median(defect_scores) - np.median(normal_scores):.3f}")

            # Threshold sweep
            best_f1, best_th = 0, 0
            best_at_fp5 = {"detect": 0, "fp": 0, "th": 0}

            for th in np.arange(0.5, 8.0, 0.25):
                tp = np.sum(defect_scores >= th)
                fn = np.sum(defect_scores < th)
                fp = np.sum(normal_scores >= th)
                tn = np.sum(normal_scores < th)

                detect_rate = tp / max(tp + fn, 1)
                fp_rate = fp / max(fp + tn, 1)
                precision = tp / max(tp + fp, 1)
                f1 = 2 * precision * detect_rate / max(precision + detect_rate, 1e-8)

                if f1 > best_f1:
                    best_f1, best_th = f1, th

                if fp_rate <= 0.05 and detect_rate > best_at_fp5["detect"]:
                    best_at_fp5 = {"detect": detect_rate, "fp": fp_rate, "th": th}

            # Print best results
            th = best_th
            tp = np.sum(defect_scores >= th)
            fp = np.sum(normal_scores >= th)
            fn = np.sum(defect_scores < th)
            tn = np.sum(normal_scores < th)
            detect_rate = tp / max(tp + fn, 1)
            fp_rate = fp / max(fp + tn, 1)

            print(f"\n  Best F1={best_f1:.3f} at th={best_th:.2f}")
            print(f"    Detect={detect_rate*100:.1f}%, FP={fp_rate*100:.1f}%")
            print(f"  Best at FP<=5%: Detect={best_at_fp5['detect']*100:.1f}%, "
                  f"FP={best_at_fp5['fp']*100:.1f}%, th={best_at_fp5['th']:.2f}")

            # Dominant metrics analysis
            all_z_metrics = []
            for r in defect_results:
                z_m = (np.array(r["metrics"]) - ens_mean) / ens_std
                dominant = int(np.argmax(z_m))
                all_z_metrics.append(dominant)

            from collections import Counter
            dom_counts = Counter(all_z_metrics)
            total = len(all_z_metrics)
            print(f"\n  Dominant metrics (defect):")
            for idx, cnt in dom_counts.most_common(5):
                print(f"    {METRIC_NAMES[idx]}: {cnt}/{total} ({cnt/total*100:.1f}%)")

            results_all[config_key] = {
                "best_f1": best_f1, "best_th": best_th,
                "detect_at_best_f1": detect_rate, "fp_at_best_f1": fp_rate,
                "best_at_fp5": best_at_fp5,
                "defect_median": float(np.median(defect_scores)),
                "normal_median": float(np.median(normal_scores)),
                "separation": float(np.median(defect_scores) - np.median(normal_scores)),
                "n_defect": len(defect_results),
                "n_normal": len(normal_results),
            }

        del score_accumulator, score_matrix
        gc.collect()
        torch.cuda.empty_cache()

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: Spatial Ens-MAX")
    print("=" * 60)

    # Reference: exp5b tile-based results
    print("\nReference (exp5b tile-based):")
    print("  F1=0.779 at th=1.50 (68.4% detect, 14.3% FP)")
    print("  FP<=5%: 46.4% detect at th=3.0, FP=2.8%")

    print("\nSpatial Ens-MAX results:")
    for key, res in sorted(results_all.items()):
        print(f"\n  {key}:")
        print(f"    Best F1={res['best_f1']:.3f} at th={res['best_th']:.2f} "
              f"(Detect={res['detect_at_best_f1']*100:.1f}%, FP={res['fp_at_best_f1']*100:.1f}%)")
        fp5 = res['best_at_fp5']
        print(f"    FP<=5%: Detect={fp5['detect']*100:.1f}%, FP={fp5['fp']*100:.1f}%, th={fp5['th']:.2f}")
        print(f"    Separation: {res['separation']:.3f}")

    # Save results
    with open(output_dir / "exp8_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {output_dir}/exp8_results.json")


if __name__ == "__main__":
    main()
