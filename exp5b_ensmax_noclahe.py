#!/usr/bin/env python3
"""Experiment 5: Ens-MAX — Multi-metric Ensemble with MAX Aggregation.

사업계획서 핵심 전략: PatchCore 코어에서 8개 지표 추출 → z-score 정규화 → MAX 집계.
단일 max_z 대신 여러 관점의 이상 지표를 종합하여 탐지 성능 향상.

기존 exp3 (64×64 타일) memory bank 재활용.
추가: CLAHE 전처리 옵션.

8 Metrics per image:
  1. tile_max_raw    — 타일별 raw distance 최대값
  2. tile_mean_raw   — 타일별 raw distance 평균
  3. tile_p95_raw    — 타일별 raw distance 95 percentile
  4. tile_max_z      — 타일별 z-score 최대값 (기존 방식)
  5. tile_mean_z     — 타일별 z-score 평균
  6. tile_p95_z      — 타일별 z-score 95 percentile
  7. anomaly_count   — z > threshold인 타일 개수
  8. cluster_score   — 이상 타일의 공간적 군집도 (connected component 기반)

각 지표를 정상 이미지 분포로 z-score 정규화 → MAX 취함 → 단일 threshold.
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
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

# exp3 memory bank 재활용
PRETRAINED_DIR = Path("/home/dk-sdd/patchcore/output_exp3_tile64/596x199/group_1")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp5b_ensmax_noclahe")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 64
TILE_STRIDE = 64

TRIM_HEAD = 100
TRIM_TAIL = 100

BRIGHTNESS_THRESHOLD = 30

# Z-score normalization
NORM_SAMPLE_IMAGES = 300
TILE_ZSCORE_THRESHOLD = 3.0  # 개별 타일 z > 3.0 이면 이상 타일
ENS_ZSCORE_THRESHOLD = 3.0   # 최종 앙상블 z > 3.0 이면 이상 이미지

USE_CLAHE = False  # CLAHE OFF — memory bank과 동일 조건

GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAM = 10
BATCH_SIZE = 512

METRIC_NAMES = [
    "tile_max_raw", "tile_mean_raw", "tile_p95_raw",
    "tile_max_z", "tile_mean_z", "tile_p95_z",
    "anomaly_count", "cluster_score",
]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_clahe(img):
    """CLAHE 히스토그램 균등화 — 조명/밝기 편차 보정."""
    import cv2
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


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


def compute_cluster_score(z_grid, threshold):
    """이상 타일의 공간적 군집도. 큰 클러스터 = 높은 점수."""
    binary = (z_grid > threshold).astype(int)
    if binary.sum() == 0:
        return 0.0
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters == 0:
        return 0.0
    # 최대 클러스터 크기 × 총 이상 타일 수의 기하평균
    cluster_sizes = [np.sum(labeled == i) for i in range(1, n_clusters + 1)]
    max_cluster = max(cluster_sizes)
    total_anomaly = sum(cluster_sizes)
    return float(np.sqrt(max_cluster * total_anomaly))


def extract_8_metrics(pos_scores, pos_mean, pos_std, n_tiles_x, n_tiles_y):
    """이미지 하나에서 8개 지표 추출."""
    valid_mask = ~np.isnan(pos_scores)
    raw = pos_scores.copy()
    raw[~valid_mask] = 0.0

    z = np.where(valid_mask, (pos_scores - pos_mean) / pos_std, 0.0)

    raw_valid = raw[valid_mask]
    z_valid = z[valid_mask]

    if len(raw_valid) == 0:
        return np.zeros(8)

    # 1-3: Raw distance metrics
    tile_max_raw = float(np.max(raw_valid))
    tile_mean_raw = float(np.mean(raw_valid))
    tile_p95_raw = float(np.percentile(raw_valid, 95))

    # 4-6: Z-score metrics
    tile_max_z = float(np.max(z_valid))
    tile_mean_z = float(np.mean(z_valid))
    tile_p95_z = float(np.percentile(z_valid, 95))

    # 7: Anomaly count
    anomaly_count = float(np.sum(z_valid > TILE_ZSCORE_THRESHOLD))

    # 8: Cluster score
    z_grid = np.full((n_tiles_y, n_tiles_x), 0.0)
    for i in range(len(pos_scores)):
        if valid_mask[i]:
            row, col = i // n_tiles_x, i % n_tiles_x
            z_grid[row, col] = z[i]
    cluster_score = compute_cluster_score(z_grid, TILE_ZSCORE_THRESHOLD)

    return np.array([
        tile_max_raw, tile_mean_raw, tile_p95_raw,
        tile_max_z, tile_mean_z, tile_p95_z,
        anomaly_count, cluster_score,
    ])


def extract_and_score_image(img, extractor, positions, memory_bank):
    """이미지에서 모든 타일 스코어링. Returns pos_scores array."""
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
    scores = score_tiles(feats, memory_bank)
    for k, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[k]
    return pos_scores


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment 5: Ens-MAX (8-metric Ensemble)")
    print(f"CLAHE: {'ON' if USE_CLAHE else 'OFF'}")
    print("=" * 60)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})")

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained memory bank and position stats
    print("\n[0] Loading pretrained exp3 model...")
    memory_bank = np.load(PRETRAINED_DIR / "memory_bank.npy")
    pos_mean = np.load(PRETRAINED_DIR / "pos_mean.npy")
    pos_std = np.load(PRETRAINED_DIR / "pos_std.npy")
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Position stats: mean [{pos_mean.min():.3f}~{pos_mean.max():.3f}], "
          f"std [{pos_std.min():.4f}~{pos_std.max():.4f}]")

    extractor = TileFeatureExtractor("cuda")

    print("\n[1] Discovering 0630 folders...")
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)}, Defect: {defect_folder.name if defect_folder else 'N/A'}")

    # ========== Phase 1: Collect 8 metrics for normal images ==========
    print(f"\n[2] Collecting 8 metrics from {NORM_SAMPLE_IMAGES} normal images...")
    normal_metrics = []
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
                if USE_CLAHE:
                    img = apply_clahe(img)
                ps = extract_and_score_image(img, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                normal_metrics.append(m)
                count += 1
                if count % 50 == 0:
                    print(f"    {count}/{NORM_SAMPLE_IMAGES}")

    normal_metrics = np.array(normal_metrics)  # (N, 8)
    print(f"  Collected: {normal_metrics.shape}")

    # Compute ensemble normalization stats (per-metric mean/std)
    ens_mean = np.mean(normal_metrics, axis=0)
    ens_std = np.std(normal_metrics, axis=0)
    ens_std = np.maximum(ens_std, 1e-6)

    print("\n  Metric stats (normal):")
    for i, name in enumerate(METRIC_NAMES):
        print(f"    {name:20s}: mean={ens_mean[i]:.4f}, std={ens_std[i]:.4f}, "
              f"range=[{normal_metrics[:, i].min():.4f}, {normal_metrics[:, i].max():.4f}]")

    # Save ensemble stats
    np.save(output_dir / "ens_mean.npy", ens_mean)
    np.save(output_dir / "ens_std.npy", ens_std)

    # ========== Phase 2: Score normal images (for calibration) ==========
    print(f"\n[3] Scoring normal images for ensemble calibration...")
    normal_ens_scores = []
    for m in normal_metrics:
        ens_z = (m - ens_mean) / ens_std
        normal_ens_scores.append(float(np.max(ens_z)))

    normal_ens_scores = np.array(normal_ens_scores)
    print(f"  Normal Ens-MAX: mean={normal_ens_scores.mean():.3f}, "
          f"std={normal_ens_scores.std():.3f}, "
          f"max={normal_ens_scores.max():.3f}, "
          f"p99={np.percentile(normal_ens_scores, 99):.3f}")

    # ========== Phase 3: Defect inference ==========
    print("\n[4] Defect inference...")
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
                if USE_CLAHE:
                    img_proc = apply_clahe(img)
                else:
                    img_proc = img

                ps = extract_and_score_image(img_proc, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                ens_z = (m - ens_mean) / ens_std
                ens_max = float(np.max(ens_z))
                dominant_metric = METRIC_NAMES[int(np.argmax(ens_z))]

                defect_results.append({
                    "file": img_path.name, "cam": cam_id,
                    "ens_max": ens_max,
                    "dominant_metric": dominant_metric,
                    "metrics": {name: float(m[i]) for i, name in enumerate(METRIC_NAMES)},
                    "metric_z": {name: float(ens_z[i]) for i, name in enumerate(METRIC_NAMES)},
                    "anomaly": bool(ens_max > ENS_ZSCORE_THRESHOLD),
                })

                # Save heatmaps for first 30
                if hm_count < 30:
                    z = np.where(~np.isnan(ps), (ps - pos_mean) / pos_std, 0.0)
                    z_grid = z.reshape(n_tiles_y, n_tiles_x)

                    if mirror:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    fig, axes = plt.subplots(1, 3, figsize=(26, 7))

                    # Original
                    axes[0].imshow(img)
                    axes[0].set_title("Original", fontsize=14)
                    axes[0].axis("off")

                    # Z-score heatmap
                    z_vmax = max(TILE_ZSCORE_THRESHOLD * 2, np.max(z_grid))
                    hm = axes[1].imshow(z_grid, cmap='hot', interpolation='nearest',
                                         extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                                         vmin=-2, vmax=z_vmax)
                    axes[1].imshow(img, alpha=0.3)
                    plt.colorbar(hm, ax=axes[1], fraction=0.03)
                    axes[1].set_title(f"Z-Score Heatmap (max_z={m[3]:.2f})", fontsize=12)
                    axes[1].axis("off")

                    # Ens-MAX bar chart
                    colors = ['red' if ens_z[i] == ens_max else 'steelblue' for i in range(8)]
                    bars = axes[2].barh(range(8), ens_z, color=colors)
                    axes[2].set_yticks(range(8))
                    axes[2].set_yticklabels(METRIC_NAMES, fontsize=9)
                    axes[2].axvline(ENS_ZSCORE_THRESHOLD, color='green', ls='--', lw=2,
                                     label=f'threshold={ENS_ZSCORE_THRESHOLD}')
                    axes[2].set_xlabel("Z-Score")
                    axes[2].legend(fontsize=9)
                    status = "ANOMALY" if ens_max > ENS_ZSCORE_THRESHOLD else "NORMAL"
                    clr = 'red' if ens_max > ENS_ZSCORE_THRESHOLD else 'green'
                    axes[2].set_title(f"Ens-MAX={ens_max:.2f} [{dominant_metric}] → {status}",
                                       color=clr, fontweight='bold', fontsize=12)

                    fig.suptitle(f"Exp5 Ens-MAX | cam{cam_id} {img_path.name}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(heatmap_dir / f"defect_{cam_id}_{hm_count:03d}.png",
                                dpi=120, bbox_inches='tight')
                    plt.close()
                    hm_count += 1

        n_anom = sum(1 for r in defect_results if r["anomaly"])
        ens_vals = [r["ens_max"] for r in defect_results]
        print(f"  Defect: {len(defect_results)} images")
        print(f"  Ens-MAX: min={min(ens_vals):.2f}, max={max(ens_vals):.2f}, "
              f"med={np.median(ens_vals):.2f}")
        print(f"  ★ Anomaly: {n_anom}/{len(defect_results)} "
              f"({100*n_anom/max(len(defect_results),1):.1f}%)")

        # Dominant metric distribution
        dom_counts = {}
        for r in defect_results:
            dm = r["dominant_metric"]
            dom_counts[dm] = dom_counts.get(dm, 0) + 1
        print(f"  Dominant metrics: {dom_counts}")

    # ========== Phase 4: Normal inference (FP rate) ==========
    print("\n[5] Normal inference (FP check)...")
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
                if USE_CLAHE:
                    img = apply_clahe(img)
                ps = extract_and_score_image(img, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                ens_z = (m - ens_mean) / ens_std
                ens_max = float(np.max(ens_z))
                normal_results.append({
                    "file": img_path.name, "folder": folder.name, "cam": cam_id,
                    "ens_max": ens_max,
                    "dominant_metric": METRIC_NAMES[int(np.argmax(ens_z))],
                    "anomaly": bool(ens_max > ENS_ZSCORE_THRESHOLD),
                })

    n_fp = sum(1 for r in normal_results if r["anomaly"])
    ens_n = [r["ens_max"] for r in normal_results]
    print(f"  Normal: {len(normal_results)} images")
    print(f"  Ens-MAX: min={min(ens_n):.2f}, max={max(ens_n):.2f}, med={np.median(ens_n):.2f}")
    print(f"  ★ FP: {n_fp}/{len(normal_results)} ({100*n_fp/max(len(normal_results),1):.1f}%)")

    # ========== Threshold sweep ==========
    print("\n[6] Threshold sweep...")
    all_defect_ens = [r["ens_max"] for r in defect_results]
    all_normal_ens = [r["ens_max"] for r in normal_results]

    print(f"  {'Threshold':>10s} {'Detect':>8s} {'FP':>8s} {'F1':>8s}")
    best_f1, best_th = 0, 0
    for th in np.arange(1.5, 6.0, 0.25):
        tp = sum(1 for v in all_defect_ens if v > th)
        fp = sum(1 for v in all_normal_ens if v > th)
        fn = len(all_defect_ens) - tp
        tn = len(all_normal_ens) - fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        det_rate = 100 * tp / max(len(all_defect_ens), 1)
        fp_rate = 100 * fp / max(len(all_normal_ens), 1)
        print(f"  {th:10.2f} {det_rate:7.1f}% {fp_rate:7.1f}% {f1:7.3f}")
        if f1 > best_f1:
            best_f1, best_th = f1, th

    print(f"\n  ★ Best F1={best_f1:.3f} at threshold={best_th:.2f}")

    # ========== Histogram ==========
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Ens-MAX distribution
    axes[0].hist(all_normal_ens, bins=60, alpha=0.6, label=f"Normal (n={len(all_normal_ens)})",
                  color='blue', edgecolor='black')
    axes[0].hist(all_defect_ens, bins=60, alpha=0.6, label=f"Defect (n={len(all_defect_ens)})",
                  color='red', edgecolor='black')
    axes[0].axvline(best_th, color='green', ls='--', lw=2, label=f'Best th={best_th:.2f} (F1={best_f1:.3f})')
    axes[0].set_xlabel("Ens-MAX Score"); axes[0].set_ylabel("Count")
    axes[0].set_title("Ens-MAX Distribution"); axes[0].legend()

    # Per-metric comparison (defect vs normal mean)
    defect_metric_means = np.mean([
        [(r["metrics"][name]) for name in METRIC_NAMES] for r in defect_results
    ], axis=0) if defect_results else np.zeros(8)
    normal_metric_means = np.mean([
        [r["ens_max"]] * 8 for r in normal_results  # placeholder
    ], axis=0) if normal_results else np.zeros(8)

    # Better: compute actual metric z-scores for defect
    if defect_results:
        defect_z_means = np.mean([
            [r["metric_z"][name] for name in METRIC_NAMES] for r in defect_results
        ], axis=0)
    else:
        defect_z_means = np.zeros(8)

    x_pos = np.arange(8)
    axes[1].bar(x_pos, defect_z_means, color='red', alpha=0.7)
    axes[1].axhline(ENS_ZSCORE_THRESHOLD, color='green', ls='--', lw=2, label=f'threshold={ENS_ZSCORE_THRESHOLD}')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(METRIC_NAMES, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel("Mean Z-Score (defect)")
    axes[1].set_title("Metric-wise Defect Z-Scores (mean)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "ensmax_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========== Save metadata ==========
    meta = {
        "experiment": "exp5_ensmax",
        "base_model": str(PRETRAINED_DIR),
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "use_clahe": USE_CLAHE,
        "metrics": METRIC_NAMES,
        "ens_mean": ens_mean.tolist(),
        "ens_std": ens_std.tolist(),
        "tile_zscore_threshold": TILE_ZSCORE_THRESHOLD,
        "ens_zscore_threshold": ENS_ZSCORE_THRESHOLD,
        "best_f1": float(best_f1),
        "best_threshold": float(best_th),
        "training_date": TRAIN_DATE,
        "training_folders": [f.name for f in normal_folders],
        "norm_sample_images": NORM_SAMPLE_IMAGES,
        "defect_count": len(defect_results),
        "defect_anomaly": sum(1 for r in defect_results if r["anomaly"]),
        "defect_detection_rate": 100 * sum(1 for r in defect_results if r["anomaly"]) / max(len(defect_results), 1),
        "normal_count": len(normal_results),
        "normal_fp": n_fp,
        "normal_fp_rate": 100 * n_fp / max(len(normal_results), 1),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Save detailed results
    with open(output_dir / "defect_results.json", "w") as f:
        json.dump(defect_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "normal_results.json", "w") as f:
        json.dump(normal_results, f, indent=2, ensure_ascii=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f} min")
    print(f"Output: {output_dir}")
    print(f"Best F1: {best_f1:.3f} at threshold {best_th:.2f}")
    print(f"Detection: {meta['defect_detection_rate']:.1f}%, FP: {meta['normal_fp_rate']:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
