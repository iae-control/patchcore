#!/usr/bin/env python3
"""Experiment 6: CLAHE + 64x64 Tiles + Ens-MAX (Full Pipeline)

학습과 추론 모두 CLAHE 전처리 적용.
exp5에서 CLAHE를 추론에만 적용했더니 역효과 → 학습부터 일관 적용.

Pipeline:
1. CLAHE 전처리 → 64×64 타일 피처 추출 → coreset memory bank
2. CLAHE 전처리 → per-position z-score 정규화 stats
3. CLAHE 전처리 → 8-metric Ens-MAX 추론
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
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_exp6_clahe_ensmax")
TARGET_SPEC = "596x199"
TRAIN_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 64
TILE_STRIDE = 64

TRIM_HEAD = 100
TRIM_TAIL = 100
SUBSAMPLE_TRAIN = 5

CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128
BRIGHTNESS_THRESHOLD = 30

NORM_SAMPLE_IMAGES = 300
TILE_ZSCORE_THRESHOLD = 3.0
ENS_ZSCORE_THRESHOLD = 3.0

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

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
    """CLAHE: L채널 히스토그램 균등화."""
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
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


def greedy_coreset_selection(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)
    print(f"    Coreset: {n} -> {target}", flush=True)
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
            print(f"      {i + 1}/{target}", flush=True)
    return features[np.array(selected)], np.array(selected)


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


def load_and_preprocess(img_path, mirror=False):
    """이미지 로드 + CLAHE + 미러."""
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return None
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        return None
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = apply_clahe(img)
    return img


def extract_tile_features(img, extractor, positions):
    """이미지에서 타일 피처 추출. Returns (features, valid_indices)."""
    tiles, valid_idx = [], []
    for i, (x, y) in enumerate(positions):
        tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        arr = np.array(tile)
        if arr.mean() < BRIGHTNESS_THRESHOLD:
            continue
        tiles.append(TRANSFORM(tile))
        valid_idx.append(i)
    if not tiles:
        return np.array([]).reshape(0, 1536), valid_idx
    all_feats = []
    for bs in range(0, len(tiles), BATCH_SIZE):
        batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0), valid_idx


def score_image(img, extractor, positions, memory_bank):
    """이미지 타일 스코어링. Returns pos_scores array."""
    n_positions = len(positions)
    pos_scores = np.full(n_positions, np.nan)
    feats, valid_idx = extract_tile_features(img, extractor, positions)
    if len(feats) == 0:
        return pos_scores
    scores = score_tiles(feats, memory_bank)
    for k, idx in enumerate(valid_idx):
        pos_scores[idx] = scores[k]
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


def extract_8_metrics(pos_scores, pos_mean, pos_std, n_tiles_x, n_tiles_y):
    valid_mask = ~np.isnan(pos_scores)
    raw = pos_scores.copy()
    raw[~valid_mask] = 0.0
    z = np.where(valid_mask, (pos_scores - pos_mean) / pos_std, 0.0)
    raw_valid = raw[valid_mask]
    z_valid = z[valid_mask]
    if len(raw_valid) == 0:
        return np.zeros(8)

    tile_max_raw = float(np.max(raw_valid))
    tile_mean_raw = float(np.mean(raw_valid))
    tile_p95_raw = float(np.percentile(raw_valid, 95))
    tile_max_z = float(np.max(z_valid))
    tile_mean_z = float(np.mean(z_valid))
    tile_p95_z = float(np.percentile(z_valid, 95))
    anomaly_count = float(np.sum(z_valid > TILE_ZSCORE_THRESHOLD))

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


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Experiment 6: CLAHE + 64x64 Tiles + Ens-MAX (Full Pipeline)", flush=True)
    print(f"CLAHE: clipLimit={CLAHE_CLIP_LIMIT}, tileGrid={CLAHE_TILE_GRID}", flush=True)
    print("=" * 60, flush=True)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))
    print(f"Tiles: {len(positions)} ({n_tiles_x}x{n_tiles_y})", flush=True)

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{GROUP_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[0] Loading backbone...", flush=True)
    extractor = TileFeatureExtractor("cuda")

    print("\n[1] Discovering 0630 folders...", flush=True)
    normal_folders, defect_folder = discover_0630_folders()
    print(f"  Normal: {len(normal_folders)}, Defect: {defect_folder.name if defect_folder else 'N/A'}", flush=True)

    # ========== Phase 1: Feature extraction with CLAHE ==========
    print(f"\n[2] Extracting 64x64 tile features WITH CLAHE (subsample={SUBSAMPLE_TRAIN})...", flush=True)
    all_features = []
    total_images = 0
    folder_counts = {}
    for folder in normal_folders:
        folder_imgs = 0
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, SUBSAMPLE_TRAIN)
            mirror = (cam_id == MIRROR_CAM)
            for img_path in images:
                img = load_and_preprocess(img_path, mirror)
                if img is None:
                    continue
                feats, valid_idx = extract_tile_features(img, extractor, positions)
                if len(feats) > 0:
                    all_features.append(feats)
                    total_images += 1
                    folder_imgs += 1
        folder_counts[folder.name] = folder_imgs
        print(f"    {folder.name}: {folder_imgs} images", flush=True)

    features = np.concatenate(all_features, axis=0)
    print(f"  Total: {total_images} images, {features.shape[0]} tiles", flush=True)
    del all_features; gc.collect()

    # ========== Phase 2: Coreset ==========
    print("\n[3] Coreset selection...", flush=True)
    memory_bank, _ = greedy_coreset_selection(features)
    print(f"  Memory bank: {memory_bank.shape}", flush=True)
    np.save(output_dir / "memory_bank.npy", memory_bank)
    del features; gc.collect(); torch.cuda.empty_cache()

    # ========== Phase 3: Position stats with CLAHE ==========
    print(f"\n[4] Position stats ({NORM_SAMPLE_IMAGES} images, CLAHE)...", flush=True)
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
                img = load_and_preprocess(img_path, mirror)
                if img is None:
                    continue
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

    # ========== Phase 4: Ensemble normalization stats ==========
    print(f"\n[5] Collecting 8 metrics from {NORM_SAMPLE_IMAGES} normal images...", flush=True)
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
                img = load_and_preprocess(img_path, mirror)
                if img is None:
                    continue
                ps = score_image(img, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                normal_metrics.append(m)
                count += 1
                if count % 50 == 0:
                    print(f"    {count}/{NORM_SAMPLE_IMAGES}", flush=True)

    normal_metrics = np.array(normal_metrics)
    ens_mean = np.mean(normal_metrics, axis=0)
    ens_std = np.std(normal_metrics, axis=0)
    ens_std = np.maximum(ens_std, 1e-6)
    np.save(output_dir / "ens_mean.npy", ens_mean)
    np.save(output_dir / "ens_std.npy", ens_std)

    print("\n  Metric stats (normal, with CLAHE):", flush=True)
    for i, name in enumerate(METRIC_NAMES):
        print(f"    {name:20s}: mean={ens_mean[i]:.4f}, std={ens_std[i]:.4f}", flush=True)

    # ========== Phase 5: Defect inference ==========
    print("\n[6] Defect inference...", flush=True)
    defect_results = []
    heatmap_dir = output_dir / "heatmaps_defect"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    hm_count = 0

    if defect_folder:
        for cam_id in CAM_IDS:
            images = get_image_paths(defect_folder, cam_id)
            mirror = (cam_id == MIRROR_CAM)
            for img_path in images:
                img = load_and_preprocess(img_path, mirror)
                if img is None:
                    continue
                ps = score_image(img, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                ens_z = (m - ens_mean) / ens_std
                ens_max = float(np.max(ens_z))
                dominant = METRIC_NAMES[int(np.argmax(ens_z))]

                defect_results.append({
                    "file": img_path.name, "cam": cam_id,
                    "ens_max": ens_max, "dominant_metric": dominant,
                    "metrics": {name: float(m[i]) for i, name in enumerate(METRIC_NAMES)},
                    "metric_z": {name: float(ens_z[i]) for i, name in enumerate(METRIC_NAMES)},
                    "anomaly": bool(ens_max > ENS_ZSCORE_THRESHOLD),
                })

                if hm_count < 30:
                    z = np.where(~np.isnan(ps), (ps - pos_mean) / pos_std, 0.0)
                    z_grid = z.reshape(n_tiles_y, n_tiles_x)

                    # Load original (un-mirrored) for display
                    orig = Image.open(img_path).convert("RGB")

                    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
                    axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")

                    z_vmax = max(TILE_ZSCORE_THRESHOLD * 2, np.max(z_grid))
                    hm = axes[1].imshow(z_grid, cmap='hot', interpolation='nearest',
                                         extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0], vmin=-2, vmax=z_vmax)
                    axes[1].imshow(orig, alpha=0.3)
                    plt.colorbar(hm, ax=axes[1], fraction=0.03)
                    axes[1].set_title(f"Z-Score (max_z={m[3]:.2f})", fontsize=12)
                    axes[1].axis("off")

                    colors = ['red' if ens_z[i] == ens_max else 'steelblue' for i in range(8)]
                    axes[2].barh(range(8), ens_z, color=colors)
                    axes[2].set_yticks(range(8))
                    axes[2].set_yticklabels(METRIC_NAMES, fontsize=9)
                    axes[2].axvline(ENS_ZSCORE_THRESHOLD, color='green', ls='--', lw=2)
                    status = "ANOMALY" if ens_max > ENS_ZSCORE_THRESHOLD else "NORMAL"
                    clr = 'red' if ens_max > ENS_ZSCORE_THRESHOLD else 'green'
                    axes[2].set_title(f"Ens-MAX={ens_max:.2f} [{dominant}] → {status}",
                                       color=clr, fontweight='bold', fontsize=12)

                    fig.suptitle(f"Exp6 CLAHE+Ens-MAX | cam{cam_id} {img_path.name}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(heatmap_dir / f"defect_{cam_id}_{hm_count:03d}.png", dpi=120, bbox_inches='tight')
                    plt.close()
                    hm_count += 1

        n_anom = sum(1 for r in defect_results if r["anomaly"])
        ens_vals = [r["ens_max"] for r in defect_results]
        print(f"  Defect: {len(defect_results)} images", flush=True)
        print(f"  Ens-MAX: min={min(ens_vals):.2f}, max={max(ens_vals):.2f}, med={np.median(ens_vals):.2f}", flush=True)
        print(f"  ★ Anomaly: {n_anom}/{len(defect_results)} ({100*n_anom/max(len(defect_results),1):.1f}%)", flush=True)
        dom_counts = {}
        for r in defect_results:
            dm = r["dominant_metric"]
            dom_counts[dm] = dom_counts.get(dm, 0) + 1
        print(f"  Dominant: {dom_counts}", flush=True)

    # ========== Phase 6: Normal inference ==========
    print("\n[7] Normal inference (FP check)...", flush=True)
    normal_results = []
    for folder in normal_folders[:5]:
        for cam_id in CAM_IDS:
            images = get_image_paths(folder, cam_id, 10)
            mirror = (cam_id == MIRROR_CAM)
            for img_path in images:
                img = load_and_preprocess(img_path, mirror)
                if img is None:
                    continue
                ps = score_image(img, extractor, positions, memory_bank)
                m = extract_8_metrics(ps, pos_mean, pos_std, n_tiles_x, n_tiles_y)
                ens_z = (m - ens_mean) / ens_std
                ens_max = float(np.max(ens_z))
                normal_results.append({
                    "file": img_path.name, "folder": folder.name, "cam": cam_id,
                    "ens_max": ens_max, "anomaly": bool(ens_max > ENS_ZSCORE_THRESHOLD),
                })

    n_fp = sum(1 for r in normal_results if r["anomaly"])
    ens_n = [r["ens_max"] for r in normal_results]
    print(f"  Normal: {len(normal_results)} images", flush=True)
    print(f"  Ens-MAX: min={min(ens_n):.2f}, max={max(ens_n):.2f}, med={np.median(ens_n):.2f}", flush=True)
    print(f"  ★ FP: {n_fp}/{len(normal_results)} ({100*n_fp/max(len(normal_results),1):.1f}%)", flush=True)

    # ========== Threshold sweep ==========
    print("\n[8] Threshold sweep...", flush=True)
    all_d = [r["ens_max"] for r in defect_results]
    all_n = [r["ens_max"] for r in normal_results]

    print(f"  {'Threshold':>10s} {'Detect':>8s} {'FP':>8s} {'F1':>8s}", flush=True)
    best_f1, best_th = 0, 0
    for th in np.arange(1.0, 6.0, 0.25):
        tp = sum(1 for v in all_d if v > th)
        fp = sum(1 for v in all_n if v > th)
        fn = len(all_d) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {th:10.2f} {100*tp/max(len(all_d),1):7.1f}% {100*fp/max(len(all_n),1):7.1f}% {f1:7.3f}", flush=True)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"\n  ★ Best F1={best_f1:.3f} at threshold={best_th:.2f}", flush=True)

    # ========== Save ==========
    meta = {
        "experiment": "exp6_clahe_ensmax",
        "clahe": {"clip_limit": CLAHE_CLIP_LIMIT, "tile_grid": list(CLAHE_TILE_GRID)},
        "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
        "tiles_per_image": len(positions), "grid": [n_tiles_x, n_tiles_y],
        "training_date": TRAIN_DATE,
        "training_folders": [f.name for f in normal_folders],
        "folder_image_counts": folder_counts,
        "total_train_images": total_images,
        "memory_bank_shape": list(memory_bank.shape),
        "coreset_ratio": CORESET_RATIO,
        "metrics": METRIC_NAMES,
        "ens_mean": ens_mean.tolist(), "ens_std": ens_std.tolist(),
        "best_f1": float(best_f1), "best_threshold": float(best_th),
        "defect_count": len(defect_results),
        "defect_anomaly": sum(1 for r in defect_results if r["anomaly"]),
        "defect_detection_rate": 100 * sum(1 for r in defect_results if r["anomaly"]) / max(len(defect_results), 1),
        "normal_count": len(normal_results),
        "normal_fp": n_fp,
        "normal_fp_rate": 100 * n_fp / max(len(normal_results), 1),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(output_dir / "defect_results.json", "w") as f:
        json.dump(defect_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "normal_results.json", "w") as f:
        json.dump(normal_results, f, indent=2, ensure_ascii=False)

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_n, bins=60, alpha=0.6, label=f"Normal (n={len(all_n)})", color='blue', edgecolor='black')
    ax.hist(all_d, bins=60, alpha=0.6, label=f"Defect (n={len(all_d)})", color='red', edgecolor='black')
    ax.axvline(best_th, color='green', ls='--', lw=2, label=f'Best th={best_th:.2f} (F1={best_f1:.3f})')
    ax.set_xlabel("Ens-MAX Score"); ax.set_ylabel("Count")
    ax.set_title(f"Exp6: CLAHE + 64x64 + Ens-MAX | Group {GROUP_ID}")
    ax.legend(); plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=150); plt.close()

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {elapsed:.1f} min", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Best F1: {best_f1:.3f} at threshold {best_th:.2f}", flush=True)
    print(f"Detection: {meta['defect_detection_rate']:.1f}%, FP: {meta['normal_fp_rate']:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
