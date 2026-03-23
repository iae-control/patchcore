#!/usr/bin/env python3
"""Experiment 7: Enhanced Ens-MAX (k-NN + Overlap + Vertical Crack Metrics)

exp5b 대비 개선사항:
  A) k-NN scoring (k=5) — 1-NN 대신 top-5 평균 거리 → 노이즈 강건
  B) Overlap tiles (stride 32) — 결함 경계 커버리지 향상
  C) 추가 지표 — 수직 균열 특화 (max_vertical_run, z_skewness, z_kurtosis)
  D) Tile z-score threshold 스윕 — 최적 threshold 탐색

exp3 memory bank 재활용 (64×64 타일, no CLAHE).
11 Metrics → z-score → MAX (or weighted).
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
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp7_enhanced")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 64
TILE_STRIDE = 32  # B: overlap (was 64)
KNN_K = 5  # A: k-NN (was 1)

TRIM_HEAD = 100
TRIM_TAIL = 100
BRIGHTNESS_THRESHOLD = 30

NORM_SAMPLE_IMAGES = 300
TILE_ZSCORE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5]  # D: sweep

GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10
BATCH_SIZE = 512

METRIC_NAMES = [
    "tile_max_raw", "tile_mean_raw", "tile_p95_raw",
    "tile_max_z", "tile_mean_z", "tile_p95_z",
    "anomaly_count", "cluster_score",
    "max_vertical_run", "z_skewness", "z_kurtosis",  # C: new
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


def score_tiles_knn(tile_features, memory_bank, k=KNN_K):
    """k-NN scoring: average of top-k nearest distances."""
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


def score_image(img, extractor, positions, memory_bank):
    """k-NN scoring for all tiles in image."""
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
    scores = score_tiles_knn(feats, memory_bank)
    for k_idx, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[k_idx]
    return pos_scores


def compute_cluster_score(z_grid, threshold):
    binary = (z_grid > threshold).astype(int)
    if binary.sum() == 0:
        return 0.0
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters == 0:
        return 0.0
    cluster_sizes = [np.sum(labeled == i) for i in range(1, n_clusters + 1)]
    max_cluster = max(cluster_sizes)
    total_anomaly = sum(cluster_sizes)
    return float(np.sqrt(max_cluster * total_anomaly))


def compute_max_vertical_run(z_grid, threshold):
    """C: 세로 방향 연속 이상 타일 최대 길이."""
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


def extract_11_metrics(pos_scores, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th=3.0):
    """11개 지표 추출."""
    valid_mask = ~np.isnan(pos_scores)
    raw = pos_scores.copy()
    raw[~valid_mask] = 0.0
    z = np.where(valid_mask, (pos_scores - pos_mean) / pos_std, 0.0)
    raw_valid = raw[valid_mask]
    z_valid = z[valid_mask]

    if len(raw_valid) == 0:
        return np.zeros(11)

    # 1-3: Raw distance metrics
    tile_max_raw = float(np.max(raw_valid))
    tile_mean_raw = float(np.mean(raw_valid))
    tile_p95_raw = float(np.percentile(raw_valid, 95))

    # 4-6: Z-score metrics
    tile_max_z = float(np.max(z_valid))
    tile_mean_z = float(np.mean(z_valid))
    tile_p95_z = float(np.percentile(z_valid, 95))

    # 7: Anomaly count
    anomaly_count = float(np.sum(z_valid > tile_th))

    # 8-9: Spatial metrics (grid)
    z_grid = np.full((n_tiles_y, n_tiles_x), 0.0)
    for i in range(len(pos_scores)):
        if valid_mask[i]:
            row, col = i // n_tiles_x, i % n_tiles_x
            if row < n_tiles_y and col < n_tiles_x:
                z_grid[row, col] = z[i]

    cluster_score = compute_cluster_score(z_grid, tile_th)
    max_vertical_run = compute_max_vertical_run(z_grid, tile_th)

    # 10-11: Distribution shape (C: new)
    z_skewness = float(scipy_stats.skew(z_valid)) if len(z_valid) > 3 else 0.0
    z_kurtosis = float(scipy_stats.kurtosis(z_valid)) if len(z_valid) > 3 else 0.0

    return np.array([
        tile_max_raw, tile_mean_raw, tile_p95_raw,
        tile_max_z, tile_mean_z, tile_p95_z,
        anomaly_count, cluster_score,
        max_vertical_run, z_skewness, z_kurtosis,
    ])


def run_ensmax_evaluation(normal_metrics, defect_images_data, normal_images_data,
                          pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th, tag):
    """Run Ens-MAX evaluation for a specific tile threshold."""
    n_metrics = normal_metrics.shape[1]
    ens_mean = np.mean(normal_metrics, axis=0)
    ens_std = np.std(normal_metrics, axis=0)
    ens_std = np.maximum(ens_std, 1e-6)

    # Score defect
    defect_ens_scores = []
    defect_dominants = []
    for m in defect_images_data:
        ens_z = (m - ens_mean) / ens_std
        ens_max = float(np.max(ens_z))
        dominant = METRIC_NAMES[int(np.argmax(ens_z))]
        defect_ens_scores.append(ens_max)
        defect_dominants.append(dominant)

    # Score normal
    normal_ens_scores = []
    for m in normal_images_data:
        ens_z = (m - ens_mean) / ens_std
        ens_max = float(np.max(ens_z))
        normal_ens_scores.append(ens_max)

    # Threshold sweep
    best_f1, best_th = 0, 0
    results_table = []
    for th in np.arange(1.0, 6.0, 0.25):
        tp = sum(1 for v in defect_ens_scores if v > th)
        fp = sum(1 for v in normal_ens_scores if v > th)
        fn = len(defect_ens_scores) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        det = 100 * tp / max(len(defect_ens_scores), 1)
        fpr = 100 * fp / max(len(normal_ens_scores), 1)
        results_table.append((th, det, fpr, f1))
        if f1 > best_f1:
            best_f1, best_th = f1, th

    # Dominant metric distribution
    dom_counts = {}
    for d in defect_dominants:
        dom_counts[d] = dom_counts.get(d, 0) + 1

    return {
        "tag": tag,
        "tile_th": tile_th,
        "defect_count": len(defect_ens_scores),
        "normal_count": len(normal_ens_scores),
        "defect_median": float(np.median(defect_ens_scores)),
        "normal_median": float(np.median(normal_ens_scores)),
        "best_f1": best_f1,
        "best_threshold": best_th,
        "dominant_metrics": dom_counts,
        "results_table": results_table,
        "ens_mean": ens_mean.tolist(),
        "ens_std": ens_std.tolist(),
    }


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Experiment 7: Enhanced Ens-MAX", flush=True)
    print(f"  k-NN k={KNN_K}, stride={TILE_STRIDE}, 11 metrics", flush=True)
    print(f"  Tile z-score thresholds: {TILE_ZSCORE_THRESHOLDS}", flush=True)
    print("=" * 60, flush=True)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})", flush=True)

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained
    print("\n[0] Loading pretrained exp3 model...", flush=True)
    memory_bank = np.load(PRETRAINED_DIR / "memory_bank.npy")
    print(f"  Memory bank: {memory_bank.shape}", flush=True)

    extractor = TileFeatureExtractor("cuda")

    print("\n[1] Discovering 0630 folders...", flush=True)
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)}, Defect: {defect_folder.name if defect_folder else 'N/A'}", flush=True)

    # ========== Phase 1: Position stats with k-NN ==========
    print(f"\n[2] Position stats ({NORM_SAMPLE_IMAGES} images, k-NN k={KNN_K})...", flush=True)
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
                ps = score_image(img, extractor, positions, memory_bank)
                all_pos_scores.append(ps)
                count += 1
                if count % 50 == 0:
                    print(f"    {count}/{NORM_SAMPLE_IMAGES}", flush=True)

    score_matrix = np.array(all_pos_scores)
    pos_mean = np.nanmean(score_matrix, axis=0)
    pos_std = np.nanstd(score_matrix, axis=0)
    pos_std = np.maximum(pos_std, 1e-6)
    np.save(output_dir / "pos_mean.npy", pos_mean)
    np.save(output_dir / "pos_std.npy", pos_std)
    print(f"  Mean: {np.nanmin(pos_mean):.4f}~{np.nanmax(pos_mean):.4f}", flush=True)
    print(f"  Std: {np.nanmin(pos_std):.4f}~{np.nanmax(pos_std):.4f}", flush=True)

    # ========== Phase 2: Collect raw scores for all images ==========
    # We'll compute metrics for each tile_th afterward
    print(f"\n[3] Scoring {NORM_SAMPLE_IMAGES} normal images...", flush=True)
    normal_raw_scores = all_pos_scores  # already computed

    print(f"\n[4] Scoring defect images...", flush=True)
    defect_raw_scores = []
    defect_meta = []  # (file, cam)
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
                ps = score_image(img, extractor, positions, memory_bank)
                defect_raw_scores.append(ps)
                defect_meta.append({"file": img_path.name, "cam": cam_id})
        print(f"  Defect: {len(defect_raw_scores)} images", flush=True)

    print(f"\n[5] Scoring normal test images...", flush=True)
    normal_test_scores = []
    normal_test_meta = []
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
                ps = score_image(img, extractor, positions, memory_bank)
                normal_test_scores.append(ps)
                normal_test_meta.append({"file": img_path.name, "folder": folder.name, "cam": cam_id})
    print(f"  Normal test: {len(normal_test_scores)} images", flush=True)

    # ========== Phase 3: D) Sweep tile z-score thresholds ==========
    print(f"\n[6] Tile threshold sweep...", flush=True)
    all_results = []
    best_overall = {"f1": 0}

    for tile_th in TILE_ZSCORE_THRESHOLDS:
        print(f"\n  --- tile_th={tile_th} ---", flush=True)

        # Compute 11 metrics for normal calibration
        normal_metrics = np.array([
            extract_11_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
            for ps in normal_raw_scores
        ])

        # Compute 11 metrics for defect
        defect_metrics = [
            extract_11_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
            for ps in defect_raw_scores
        ]

        # Compute 11 metrics for normal test
        normal_test_metrics = [
            extract_11_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th)
            for ps in normal_test_scores
        ]

        result = run_ensmax_evaluation(
            normal_metrics, defect_metrics, normal_test_metrics,
            pos_mean, pos_std, n_tiles_x, n_tiles_y, tile_th,
            tag=f"tile_th={tile_th}"
        )
        all_results.append(result)

        # Print summary
        print(f"  Defect median Ens-MAX: {result['defect_median']:.3f}", flush=True)
        print(f"  Normal median Ens-MAX: {result['normal_median']:.3f}", flush=True)
        print(f"  Best F1={result['best_f1']:.3f} at threshold={result['best_threshold']:.2f}", flush=True)
        print(f"  Dominant: {result['dominant_metrics']}", flush=True)

        # Print top rows
        print(f"  {'Threshold':>10s} {'Detect':>8s} {'FP':>8s} {'F1':>8s}", flush=True)
        for th, det, fpr, f1 in result['results_table']:
            if det > 0 or fpr > 0:
                print(f"  {th:10.2f} {det:7.1f}% {fpr:7.1f}% {f1:7.3f}", flush=True)

        if result['best_f1'] > best_overall['f1']:
            best_overall = {
                'f1': result['best_f1'],
                'tile_th': tile_th,
                'ens_th': result['best_threshold'],
                'normal_metrics': normal_metrics,
                'defect_metrics': defect_metrics,
                'normal_test_metrics': normal_test_metrics,
                'ens_mean': result['ens_mean'],
                'ens_std': result['ens_std'],
            }

    # ========== Phase 4: Generate heatmaps for best config ==========
    print(f"\n[7] Generating heatmaps (best: tile_th={best_overall['tile_th']})...", flush=True)
    heatmap_dir = output_dir / "heatmaps_defect"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    best_tile_th = best_overall['tile_th']
    best_ens_mean = np.array(best_overall['ens_mean'])
    best_ens_std = np.array(best_overall['ens_std'])

    for hm_idx in range(min(30, len(defect_raw_scores))):
        ps = defect_raw_scores[hm_idx]
        meta = defect_meta[hm_idx]
        m = extract_11_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y, best_tile_th)
        ens_z = (m - best_ens_mean) / best_ens_std
        ens_max = float(np.max(ens_z))
        dominant = METRIC_NAMES[int(np.argmax(ens_z))]

        valid_mask = ~np.isnan(ps)
        z = np.where(valid_mask, (ps - pos_mean) / pos_std, 0.0)
        z_grid = z.reshape(n_tiles_y, n_tiles_x)

        # Load original image
        cam_id = meta['cam']
        mirror = (cam_id == MIRROR_CAM)
        img_path = None
        for c in CAM_IDS:
            if c == cam_id:
                cam_dir = defect_folder / f"camera_{c}"
                for p in cam_dir.iterdir():
                    if p.name == meta['file']:
                        img_path = p
                        break
        if img_path:
            orig = Image.open(img_path).convert("RGB")
        else:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(26, 7))
        axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")

        z_vmax = max(best_tile_th * 2, np.max(z_grid))
        hm = axes[1].imshow(z_grid, cmap='hot', interpolation='nearest',
                             extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0], vmin=-2, vmax=z_vmax)
        axes[1].imshow(orig, alpha=0.3)
        plt.colorbar(hm, ax=axes[1], fraction=0.03)
        axes[1].set_title(f"Z-Score (max_z={m[3]:.2f})", fontsize=12)
        axes[1].axis("off")

        colors = ['red' if ens_z[i] == ens_max else 'steelblue' for i in range(len(METRIC_NAMES))]
        axes[2].barh(range(len(METRIC_NAMES)), ens_z, color=colors)
        axes[2].set_yticks(range(len(METRIC_NAMES)))
        axes[2].set_yticklabels(METRIC_NAMES, fontsize=8)
        axes[2].axvline(best_overall['ens_th'], color='green', ls='--', lw=2)
        status = "ANOMALY" if ens_max > best_overall['ens_th'] else "NORMAL"
        clr = 'red' if ens_max > best_overall['ens_th'] else 'green'
        axes[2].set_title(f"Ens-MAX={ens_max:.2f} [{dominant}] → {status}",
                           color=clr, fontweight='bold', fontsize=12)

        fig.suptitle(f"Exp7 Enhanced | cam{cam_id} {meta['file']}", fontsize=12)
        plt.tight_layout()
        plt.savefig(heatmap_dir / f"defect_{cam_id}_{hm_idx:03d}.png", dpi=120, bbox_inches='tight')
        plt.close()

    # ========== Summary ==========
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':>20s} {'Best F1':>8s} {'Best Th':>8s} {'D_med':>8s} {'N_med':>8s}", flush=True)
    for r in all_results:
        print(f"  tile_th={r['tile_th']:<5.1f}   {r['best_f1']:>8.3f} {r['best_threshold']:>8.2f} "
              f"{r['defect_median']:>8.3f} {r['normal_median']:>8.3f}", flush=True)

    print(f"\n★ BEST: tile_th={best_overall['tile_th']}, "
          f"F1={best_overall['f1']:.3f}, ens_th={best_overall['ens_th']:.2f}", flush=True)

    # Compare with exp5b baseline
    print(f"\nvs exp5b baseline (F1=0.779, th=1.50):", flush=True)
    improvement = best_overall['f1'] - 0.779
    print(f"  F1 change: {improvement:+.3f}", flush=True)

    # Save
    meta_save = {
        "experiment": "exp7_enhanced_ensmax",
        "knn_k": KNN_K,
        "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
        "tiles_per_image": len(positions), "grid": [n_tiles_x, n_tiles_y],
        "metrics": METRIC_NAMES,
        "tile_thresholds_tested": TILE_ZSCORE_THRESHOLDS,
        "best_tile_threshold": best_overall['tile_th'],
        "best_ens_threshold": best_overall['ens_th'],
        "best_f1": best_overall['f1'],
        "all_results": [{k: v for k, v in r.items() if k != 'results_table'} for r in all_results],
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta_save, f, indent=2, ensure_ascii=False)

    # Save best config stats
    np.save(output_dir / "best_ens_mean.npy", np.array(best_overall['ens_mean']))
    np.save(output_dir / "best_ens_std.npy", np.array(best_overall['ens_std']))

    elapsed = (time.time() - t0) / 60
    print(f"\nDONE in {elapsed:.1f} min", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
