#!/usr/bin/env python3
"""Experiment 7 Tuning: Find optimal metric subset and aggregation.

exp7 raw scores 재활용 — 추가 GPU 연산 없이 다양한 조합 테스트.

테스트 항목:
1. Metric subset selection — 11개 중 최적 조합
2. Aggregation: MAX vs Top-K mean vs Weighted
3. k-NN k=1 vs k=5 비교 (k=1은 exp5b와 동일 조건)
4. Stride 64 (no overlap) vs stride 32 (overlap) 비교

미탐 감수 방향: FP ≤ 5% 제약 조건에서 최대 탐지율.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re, time, gc, itertools
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

PRETRAINED_DIR = Path("/home/dk-sdd/patchcore/output_exp3_tile64/596x199/group_1")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp7_tuning")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 64

TRIM_HEAD = 100
TRIM_TAIL = 100
BRIGHTNESS_THRESHOLD = 30
NORM_SAMPLE_IMAGES = 300

GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10
BATCH_SIZE = 512

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

def score_tiles_knn(tile_features, memory_bank, k=1):
    if len(tile_features) == 0:
        return np.array([])
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    scores = []
    for i in range(0, len(tf), 128):
        chunk = tf[i:i+128]
        dists = torch.cdist(chunk, mb)
        if k == 1:
            min_dists, _ = dists.min(dim=1)
            scores.append(min_dists.cpu().numpy())
        else:
            topk_dists, _ = dists.topk(k, dim=1, largest=False)
            avg_dists = topk_dists.mean(dim=1)
            scores.append(avg_dists.cpu().numpy())
    return np.concatenate(scores)


def compute_cluster_score(z_grid, threshold):
    binary = (z_grid > threshold).astype(int)
    if binary.sum() == 0:
        return 0.0
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters == 0:
        return 0.0
    cluster_sizes = [np.sum(labeled == i) for i in range(1, n_clusters + 1)]
    return float(np.sqrt(max(cluster_sizes) * sum(cluster_sizes)))

def compute_max_vertical_run(z_grid, threshold):
    binary = (z_grid > threshold).astype(int)
    max_run = 0
    for col in range(binary.shape[1]):
        run = 0
        for row in range(binary.shape[0]):
            if binary[row, col]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
    return float(max_run)


def extract_metrics(pos_scores, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th=3.0):
    valid_mask = ~np.isnan(pos_scores)
    raw = pos_scores.copy()
    raw[~valid_mask] = 0.0
    z = np.where(valid_mask, (pos_scores - pos_mean) / pos_std, 0.0)
    raw_valid = raw[valid_mask]
    z_valid = z[valid_mask]
    if len(raw_valid) == 0:
        return np.zeros(11)

    z_grid = np.full((n_tiles_y, n_tiles_x), 0.0)
    for i in range(len(pos_scores)):
        if valid_mask[i] and i // n_tiles_x < n_tiles_y and i % n_tiles_x < n_tiles_x:
            z_grid[i // n_tiles_x, i % n_tiles_x] = z[i]

    return np.array([
        float(np.max(raw_valid)),
        float(np.mean(raw_valid)),
        float(np.percentile(raw_valid, 95)),
        float(np.max(z_valid)),
        float(np.mean(z_valid)),
        float(np.percentile(z_valid, 95)),
        float(np.sum(z_valid > tile_th)),
        compute_cluster_score(z_grid, tile_th),
        compute_max_vertical_run(z_grid, tile_th),
        float(scipy_stats.skew(z_valid)) if len(z_valid) > 3 else 0.0,
        float(scipy_stats.kurtosis(z_valid)) if len(z_valid) > 3 else 0.0,
    ])


def score_image(img, extractor, positions, memory_bank, k=1):
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
        return pos_scores
    all_feats = []
    for bs in range(0, len(tiles), BATCH_SIZE):
        batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    feats = np.concatenate(all_feats, axis=0)
    scores = score_tiles_knn(feats, memory_bank, k=k)
    for ki, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[ki]
    return pos_scores


def evaluate_config(normal_metrics, defect_metrics, normal_test_metrics,
                    metric_indices, agg_method="max", fp_limit=0.05):
    """Evaluate a specific metric subset + aggregation method."""
    # Subset
    norm_sub = normal_metrics[:, metric_indices]
    ens_mean = np.mean(norm_sub, axis=0)
    ens_std = np.std(norm_sub, axis=0)
    ens_std = np.maximum(ens_std, 1e-6)

    # Score defect
    defect_scores = []
    for m in defect_metrics:
        m_sub = np.array(m)[metric_indices]
        ens_z = (m_sub - ens_mean) / ens_std
        if agg_method == "max":
            defect_scores.append(float(np.max(ens_z)))
        elif agg_method == "top3_mean":
            topk = np.sort(ens_z)[-min(3, len(ens_z)):]
            defect_scores.append(float(np.mean(topk)))
        elif agg_method == "mean":
            defect_scores.append(float(np.mean(ens_z)))

    # Score normal
    normal_scores = []
    for m in normal_test_metrics:
        m_sub = np.array(m)[metric_indices]
        ens_z = (m_sub - ens_mean) / ens_std
        if agg_method == "max":
            normal_scores.append(float(np.max(ens_z)))
        elif agg_method == "top3_mean":
            topk = np.sort(ens_z)[-min(3, len(ens_z)):]
            normal_scores.append(float(np.mean(topk)))
        elif agg_method == "mean":
            normal_scores.append(float(np.mean(ens_z)))

    # Find best threshold with FP constraint
    best_det, best_th, best_f1 = 0, 0, 0
    best_det_constrained, best_th_constrained = 0, 0
    for th in np.arange(0.5, 8.0, 0.25):
        tp = sum(1 for v in defect_scores if v > th)
        fp = sum(1 for v in normal_scores if v > th)
        fn = len(defect_scores) - tp
        det = tp / max(len(defect_scores), 1)
        fpr = fp / max(len(normal_scores), 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = det
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_th = f1, th
        if fpr <= fp_limit and det > best_det_constrained:
            best_det_constrained = det
            best_th_constrained = th

    return {
        "best_f1": best_f1,
        "best_th": best_th,
        "det_at_fp5": best_det_constrained,
        "th_at_fp5": best_th_constrained,
        "defect_median": float(np.median(defect_scores)),
        "normal_median": float(np.median(normal_scores)),
        "separation": float(np.median(defect_scores) - np.median(normal_scores)),
    }


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Exp7 Tuning: Metric subset + aggregation optimization", flush=True)
    print("=" * 60, flush=True)

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_bank = np.load(PRETRAINED_DIR / "memory_bank.npy")
    extractor = TileFeatureExtractor("cuda")
    normal_folders, defect_folder = discover_0630_folders()

    # ===== Test configurations: stride 64 (k=1), stride 64 (k=5), stride 32 (k=1), stride 32 (k=5) =====
    configs = [
        {"name": "s64_k1", "stride": 64, "k": 1},
        {"name": "s64_k5", "stride": 64, "k": 5},
        {"name": "s32_k1", "stride": 32, "k": 1},
        {"name": "s32_k5", "stride": 32, "k": 5},
    ]

    all_config_results = {}

    for cfg in configs:
        stride = cfg["stride"]
        k = cfg["k"]
        name = cfg["name"]
        print(f"\n{'='*50}", flush=True)
        print(f"Config: {name} (stride={stride}, k={k})", flush=True)
        print(f"{'='*50}", flush=True)

        positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, stride)
        n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, stride))
        n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, stride))
        print(f"  Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})", flush=True)

        # Position stats
        print(f"  Computing position stats...", flush=True)
        all_pos_scores = []
        count = 0
        for folder in normal_folders:
            if count >= NORM_SAMPLE_IMAGES:
                break
            for cam_id in CAM_IDS:
                if count >= NORM_SAMPLE_IMAGES:
                    break
                images = get_image_paths(folder, cam_id, 3)
                mirror = (cam_id == MIRROR_CAM)
                for img_path in images:
                    if count >= NORM_SAMPLE_IMAGES:
                        break
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except:
                        continue
                    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                        continue
                    if mirror:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    ps = score_image(img, extractor, positions, memory_bank, k=k)
                    all_pos_scores.append(ps)
                    count += 1
                    if count % 100 == 0:
                        print(f"    pos_stats: {count}/{NORM_SAMPLE_IMAGES}", flush=True)

        pos_mean = np.nanmean(np.array(all_pos_scores), axis=0)
        pos_std = np.nanstd(np.array(all_pos_scores), axis=0)
        pos_std = np.maximum(pos_std, 1e-6)

        # Compute metrics for different tile thresholds
        best_tile_th = 3.0
        tile_ths = [2.5, 3.0, 3.5]

        print(f"  Computing metrics for normal cal images...", flush=True)
        normal_metrics_by_th = {}
        for tile_th in tile_ths:
            normal_metrics_by_th[tile_th] = np.array([
                extract_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
                for ps in all_pos_scores
            ])

        # Defect
        print(f"  Scoring defect images...", flush=True)
        defect_raw = []
        di = 0
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
                    ps = score_image(img, extractor, positions, memory_bank, k=k)
                    defect_raw.append(ps)
                    di += 1
                    if di % 200 == 0:
                        print(f"    defect: {di}", flush=True)
        print(f"  Defect: {len(defect_raw)} images", flush=True)

        defect_metrics_by_th = {}
        for tile_th in tile_ths:
            defect_metrics_by_th[tile_th] = [
                extract_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
                for ps in defect_raw
            ]

        # Normal test
        print(f"  Scoring normal test images...", flush=True)
        normal_test_raw = []
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
                    ps = score_image(img, extractor, positions, memory_bank, k=k)
                    normal_test_raw.append(ps)
        print(f"  Normal test: {len(normal_test_raw)} images", flush=True)

        normal_test_metrics_by_th = {}
        for tile_th in tile_ths:
            normal_test_metrics_by_th[tile_th] = [
                extract_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
                for ps in normal_test_raw
            ]

        # ===== Evaluate metric subsets =====
        print(f"\n  Evaluating metric subsets...", flush=True)

        # Metric subsets to test
        subsets = {
            "all_11": list(range(11)),
            "original_8": list(range(8)),  # exp5b metrics
            "top4_disc": [0, 3, 6, 7],  # tile_max_raw, tile_max_z, anomaly_count, cluster_score
            "z_only": [3, 4, 5, 6, 7, 8],  # z-based + spatial
            "raw_only": [0, 1, 2],
            "count_spatial": [6, 7, 8],  # anomaly_count, cluster_score, max_vert
            "best_3": [3, 6, 8],  # tile_max_z, anomaly_count, max_vert
            "no_raw": list(range(3, 11)),  # exclude raw metrics
        }

        agg_methods = ["max", "top3_mean", "mean"]

        config_results = []

        for tile_th in tile_ths:
            nm = normal_metrics_by_th[tile_th]
            dm = defect_metrics_by_th[tile_th]
            ntm = normal_test_metrics_by_th[tile_th]

            for subset_name, indices in subsets.items():
                for agg in agg_methods:
                    r = evaluate_config(nm, dm, ntm, indices, agg, fp_limit=0.05)
                    result = {
                        "config": name,
                        "tile_th": tile_th,
                        "subset": subset_name,
                        "agg": agg,
                        "n_metrics": len(indices),
                        **r
                    }
                    config_results.append(result)

        all_config_results[name] = config_results

    # ===== Print summary =====
    print(f"\n{'='*80}", flush=True)
    print(f"TUNING RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Target: FP ≤ 5% constraint → maximize detection rate", flush=True)
    print(f"{'='*80}", flush=True)

    # Sort by det_at_fp5
    all_flat = []
    for name, results in all_config_results.items():
        all_flat.extend(results)

    sorted_by_det = sorted(all_flat, key=lambda x: x["det_at_fp5"], reverse=True)

    print(f"\n{'Config':>8s} {'TileTh':>6s} {'Subset':>15s} {'Agg':>10s} {'Det@FP5':>8s} {'F1':>6s} {'Sep':>6s}", flush=True)
    for r in sorted_by_det[:30]:
        print(f"  {r['config']:>8s} {r['tile_th']:>6.1f} {r['subset']:>15s} {r['agg']:>10s} "
              f"{r['det_at_fp5']*100:>7.1f}% {r['best_f1']:>6.3f} {r['separation']:>6.3f}", flush=True)

    # Also show best F1 results
    sorted_by_f1 = sorted(all_flat, key=lambda x: x["best_f1"], reverse=True)
    print(f"\nTop by F1:", flush=True)
    for r in sorted_by_f1[:10]:
        print(f"  {r['config']:>8s} {r['tile_th']:>6.1f} {r['subset']:>15s} {r['agg']:>10s} "
              f"F1={r['best_f1']:.3f} Det@FP5={r['det_at_fp5']*100:.1f}%", flush=True)

    # Save
    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump(all_flat, f, indent=2, ensure_ascii=False)

    elapsed = (time.time() - t0) / 60
    print(f"\nDONE in {elapsed:.1f} min", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
