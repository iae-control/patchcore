#!/usr/bin/env python3
"""PatchCore v5b-FULL — Half-Resolution (960x600) Training for Fast Inference.

Changes from original 1920x1200 training:
  - Input resized to 960x600 (4x fewer pixels)
  - Spatial output: 25x40 = 1000 features/image (was 50x80 = 4000)
  - Expected inference: ~6.8ms/img backbone + ~1ms scoring = ~8ms/img
  - Dual A40: ~4ms/img → 8000 images in ~32 seconds

Group-Camera mapping (H-beam cross-section, symmetric pairs):
  Group 1: cam [1, 10]  — flange top surface     (mirror: cam 10)
  Group 2: cam [2, 9]   — flange-web junction top (mirror: cam 9)
  Group 3: cam [3, 8]   — flange side             (mirror: cam 8)
  Group 4: cam [4, 7]   — flange-web junction bot (mirror: cam 7)
  Group 5: cam [5, 6]   — flange bottom surface   (mirror: cam 6)

Date-balanced strategy:
  - 20250630 (defect day): sparse sampling (subsample=15)
  - Other dates: dense sampling (subsample=3)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, time, json, re, gc
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
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5b_half")

# Original capture resolution
IMAGE_WIDTH_ORIG = 1920
IMAGE_HEIGHT_ORIG = 1200

# Half resolution for training/inference
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 600

SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

TRIM_HEAD = 100
TRIM_TAIL = 100

TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"
DEFECT_DATE = "20250630"

# ===== GROUP DEFINITIONS =====
GROUP_CONFIG = {
    1: {"cams": [1, 10], "mirror_cam": 10, "infer_cam": 1},
    2: {"cams": [2, 9],  "mirror_cam": 9,  "infer_cam": 2},
    3: {"cams": [3, 8],  "mirror_cam": 8,  "infer_cam": 3},
    4: {"cams": [4, 7],  "mirror_cam": 7,  "infer_cam": 4},
    5: {"cams": [5, 6],  "mirror_cam": 6,  "infer_cam": 5},
}

# Sampling
SUBSAMPLE_NORMAL = 3
SUBSAMPLE_DEFECT_DATE = 15

# Memory management
MEMORY_CAP_FEATURES = 8_000_000
CORESET_MAX_FEATURES = 4_000_000
CORESET_RATIO = 0.003
CORESET_PROJECTION_DIM = 128

SELF_VAL_MAD_K = 3.5

# Per-position normalization
NORM_SAMPLE_PER_DATE = 200

# Heatmap
GAUSSIAN_SIGMA = 4
SCORE_PERCENTILE_CAP = 99.5
Z_SCORE_THRESHOLD = 3.0

# Which groups to train
GROUPS_TO_TRAIN = [1, 2, 3, 4, 5]


# ===== UTILITIES =====
def natural_sort_key(path):
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


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


def discover_spec_folders(spec_pattern):
    date_folders = defaultdict(list)
    defect_folder = None
    for entry in sorted(NAS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if re.match(r'^\d{8}$', entry.name):
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and spec_pattern in sub.name:
                        if (sub / "camera_1").is_dir():
                            if DEFECT_FOLDER_PREFIX in sub.name:
                                defect_folder = sub
                            else:
                                date_folders[entry.name].append(sub)
            except PermissionError:
                continue
        elif spec_pattern in entry.name:
            if (entry / "camera_1").is_dir():
                if DEFECT_FOLDER_PREFIX in entry.name:
                    defect_folder = entry
                else:
                    date_folders["toplevel"].append(entry)
    return date_folders, defect_folder


# ===== FEATURE EXTRACTOR =====
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
            print("  Loading WideResNet50 (downloading)...")
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
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
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== FEATURE EXTRACTION =====
def extract_features_date_balanced(date_folders, cam_ids, mirror_cam, extractor):
    all_features = []
    total_features_count = 0
    total_images = 0
    feat_spatial_size = None
    rng = np.random.RandomState(0)
    date_stats = {}

    for date_str, folders in sorted(date_folders.items()):
        if date_str == DEFECT_DATE:
            subsample = SUBSAMPLE_DEFECT_DATE
            label = f"{date_str} (DEFECT - sparse)"
        else:
            subsample = SUBSAMPLE_NORMAL
            label = f"{date_str} (normal - dense)"

        date_images = 0
        date_features = 0
        print(f"\n    {label}: {len(folders)} folders, subsample={subsample}")
        sys.stdout.flush()

        for fi, folder in enumerate(folders):
            for cam_id in cam_ids:
                images = get_image_paths(folder, cam_id, subsample)
                for img_path in images:
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
                    if img.size != (IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG):
                        continue
                    if cam_id == mirror_cam:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    tensor = TRANSFORM(img).unsqueeze(0)
                    features, spatial_size = extractor.extract_spatial(tensor)
                    all_features.append(features)
                    total_features_count += features.shape[0]
                    date_features += features.shape[0]

                    if feat_spatial_size is None:
                        feat_spatial_size = spatial_size

                    total_images += 1
                    date_images += 1

                    if total_features_count > MEMORY_CAP_FEATURES:
                        concat = np.concatenate(all_features, axis=0)
                        keep = MEMORY_CAP_FEATURES // 2
                        idx = rng.choice(concat.shape[0], keep, replace=False)
                        all_features = [concat[np.sort(idx)]]
                        total_features_count = keep
                        del concat; gc.collect()

            if (fi + 1) % 10 == 0:
                print(f"      {fi+1}/{len(folders)} folders, {date_images} images")
                sys.stdout.flush()

        date_stats[date_str] = {"images": date_images, "features": date_features,
                                "folders": len(folders), "subsample": subsample}
        print(f"    -> {date_str}: {date_images} images")
        sys.stdout.flush()

    print(f"\n    Total: {total_images} images")
    result = np.concatenate(all_features, axis=0)
    del all_features; gc.collect()

    if result.shape[0] > CORESET_MAX_FEATURES:
        print(f"    Subsample: {result.shape[0]:,} -> {CORESET_MAX_FEATURES:,}")
        idx = np.sort(rng.choice(result.shape[0], CORESET_MAX_FEATURES, replace=False))
        result = result[idx].copy(); gc.collect()

    print(f"    Final: {result.shape} ({result.nbytes / 1024**3:.1f}GB)")
    sys.stdout.flush()
    return result, feat_spatial_size, total_images, date_stats


# ===== CORESET =====
def greedy_coreset(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    print(f"    Coreset: {n:,} -> {target:,}")
    sys.stdout.flush()

    if target >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj = rng.randn(d, proj_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    proj_t = torch.from_numpy(proj).cuda()
    projected_list = []
    for i in range(0, n, 500000):
        chunk = torch.from_numpy(features[i:i+500000].astype(np.float32)).cuda()
        projected_list.append((chunk @ proj_t).cpu())
        del chunk
    torch.cuda.empty_cache()

    projected = torch.cat(projected_list, dim=0).cuda()
    del projected_list

    selected = [rng.randint(n)]
    min_dists = torch.full((n,), float('inf'), device='cuda')

    for i in tqdm(range(target - 1), desc="Coreset", leave=True):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(torch.argmax(min_dists).item())

    del projected; torch.cuda.empty_cache()
    return features[np.array(selected)], np.array(selected)


# ===== SCORING =====
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


# ===== PER-POSITION NORM =====
def compute_position_stats(date_folders, cam_ids, mirror_cam, extractor, spatial_size,
                           memory_bank, output_dir):
    print(f"    Computing per-position stats...")
    score_accum = []
    total_count = 0

    for date_str, folders in sorted(date_folders.items()):
        subsample = SUBSAMPLE_DEFECT_DATE if date_str == DEFECT_DATE else SUBSAMPLE_NORMAL * 2
        date_count = 0
        for folder in folders:
            if date_count >= NORM_SAMPLE_PER_DATE:
                break
            for cam_id in cam_ids:
                if date_count >= NORM_SAMPLE_PER_DATE:
                    break
                images = get_image_paths(folder, cam_id, subsample)
                for img_path in images:
                    if date_count >= NORM_SAMPLE_PER_DATE:
                        break
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
                    if img.size != (IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG):
                        continue
                    if cam_id == mirror_cam:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    tensor = TRANSFORM(img).unsqueeze(0)
                    features, _ = extractor.extract_spatial(tensor)
                    scores = score_spatial(features, memory_bank)
                    score_accum.append(scores)
                    date_count += 1
                    total_count += 1
        print(f"      {date_str}: {date_count} images")

    print(f"      Total: {total_count} images")
    score_matrix = np.array(score_accum)
    pos_mean = np.mean(score_matrix, axis=0)
    pos_std = np.clip(np.std(score_matrix, axis=0), 0.01, None)
    np.save(output_dir / "pos_mean.npy", pos_mean)
    np.save(output_dir / "pos_std.npy", pos_std)
    print(f"      Mean: {pos_mean.min():.4f}~{pos_mean.max():.4f}, "
          f"Std: {pos_std.min():.4f}~{pos_std.max():.4f}")
    return pos_mean, pos_std


# ===== HEATMAP =====
def generate_heatmap_zscore(img_path, memory_bank, extractor, spatial_size, output_path,
                            pos_mean, pos_std, threshold_z, group_id, mirror=False):
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG):
        return None, None, None

    img_display = img.copy()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = TRANSFORM(img).unsqueeze(0)
    features, _ = extractor.extract_spatial(tensor)
    scores = score_spatial(features, memory_bank)
    z_scores = (scores - pos_mean) / pos_std

    Hp, Wp = spatial_size
    z_map = z_scores.reshape(Hp, Wp)
    z_tensor = torch.from_numpy(z_map).unsqueeze(0).unsqueeze(0).float()
    heatmap_full = F.interpolate(
        z_tensor, size=(IMAGE_HEIGHT_ORIG, IMAGE_WIDTH_ORIG),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()
    heatmap_full = gaussian_filter(heatmap_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)

    if mirror:
        heatmap_full = np.fliplr(heatmap_full)

    max_z = float(heatmap_full.max())
    raw_max = float(scores.max())

    img_arr = np.array(img_display)
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    axes[0].imshow(img_arr)
    axes[0].set_title(f"Original: {img_path.name}", fontsize=14)
    axes[0].axis("off")

    vmax_z = max(threshold_z * 2, np.percentile(heatmap_full, SCORE_PERCENTILE_CAP))
    im1 = axes[1].imshow(heatmap_full, cmap='hot', vmin=0, vmax=vmax_z)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Z-Score (max={max_z:.2f})", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(img_arr)
    overlay_min = threshold_z * 0.7
    masked = np.ma.masked_where(heatmap_full < overlay_min, heatmap_full)
    im2 = axes[2].imshow(masked, cmap='jet', alpha=0.65,
                          vmin=overlay_min, vmax=max(vmax_z, threshold_z * 2))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    if max_z > threshold_z:
        axes[2].set_title(f"ANOMALY (z={max_z:.2f})", fontsize=14, color='red', fontweight='bold')
    else:
        axes[2].set_title(f"NORMAL (z={max_z:.2f})", fontsize=14, color='green', fontweight='bold')
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v5b-HALF | {TARGET_SPEC}/group_{group_id} | {img_path.name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return max_z, raw_max, float(np.mean(scores))


# ===== TRAIN ONE GROUP =====
def train_group(group_id, date_folders, defect_folder, extractor, spatial_size):
    cfg = GROUP_CONFIG[group_id]
    cam_ids = cfg["cams"]
    mirror_cam = cfg["mirror_cam"]
    infer_cam = cfg["infer_cam"]

    t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"  GROUP {group_id}: cameras {cam_ids} (mirror={mirror_cam})")
    print(f"  Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT} (half)")
    print(f"{'#'*60}")

    output_dir = OUTPUT_DIR / TARGET_SPEC / f"group_{group_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # [A] Extract features
    print(f"\n  [A] Feature extraction...")
    features, spat_size, n_images, date_stats = extract_features_date_balanced(
        date_folders, cam_ids, mirror_cam, extractor
    )

    if features.shape[0] == 0:
        print(f"  ERROR: no features for group {group_id}!")
        return

    # [B] Coreset
    print(f"\n  [B] Coreset selection...")
    memory_bank, indices = greedy_coreset(features)
    print(f"    Memory bank: {memory_bank.shape}")
    del features; gc.collect()

    # [C] Self-validation
    print(f"\n  [C] Self-validation...")
    val_folders = {d: fs[:3] for d, fs in date_folders.items() if d != DEFECT_DATE}
    val_features, _, _, _ = extract_features_date_balanced(
        val_folders, cam_ids, mirror_cam, extractor
    )
    val_scores = score_spatial(val_features, memory_bank)
    median = np.median(val_scores)
    mad = np.median(np.abs(val_scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    print(f"    Median={median:.4f}, MAD_std={mad_std:.4f}, Threshold={threshold:.4f}")

    plt.figure(figsize=(12, 5))
    plt.hist(val_scores, bins=200, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Thr: {threshold:.4f}')
    plt.axvline(median, color='blue', linestyle='-', alpha=0.5, label=f'Med: {median:.4f}')
    plt.title(f"v5b-HALF | {TARGET_SPEC}/group_{group_id} | {n_images} imgs | bank={memory_bank.shape[0]}")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100); plt.close()
    del val_features, val_scores

    # Save model
    np.save(output_dir / "memory_bank.npy", memory_bank)
    np.save(output_dir / "spatial_size.npy", np.array(spat_size))

    meta = {
        "group": group_id,
        "cameras": cam_ids,
        "mirror_cam": mirror_cam,
        "n_training_images": n_images,
        "date_stats": date_stats,
        "memory_bank_shape": list(memory_bank.shape),
        "spatial_size": list(spat_size),
        "input_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "original_resolution": [IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG],
        "threshold_mad": float(threshold),
        "median": float(median),
        "mad_std": float(mad_std),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # [D] Per-position normalization
    print(f"\n  [D] Per-position normalization...")
    pos_mean, pos_std = compute_position_stats(
        date_folders, cam_ids, mirror_cam, extractor, spat_size, memory_bank, output_dir
    )

    # [E] Inference on defect folder
    print(f"\n  [E] Inference on {defect_folder.name}...")
    cam_images = get_image_paths(defect_folder, cam_id=infer_cam, subsample_step=1)
    print(f"    Camera {infer_cam} images: {len(cam_images)}")

    heatmap_dir = output_dir / "heatmaps_zscore" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in tqdm(range(len(cam_images)), desc=f"G{group_id} Heatmaps"):
        img_path = cam_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_z, raw_max, mean_score = generate_heatmap_zscore(
            img_path, memory_bank, extractor, spat_size, out_path,
            pos_mean, pos_std, Z_SCORE_THRESHOLD, group_id
        )
        if max_z is not None:
            results.append({
                "idx": idx, "file": img_path.name,
                "max_z_score": max_z, "raw_max_score": raw_max,
                "mean_score": mean_score, "anomaly": max_z > Z_SCORE_THRESHOLD,
            })

    with open(output_dir / "inference_results_full.json", "w") as f:
        json.dump(results, f, indent=2)

    n_anomaly = sum(1 for r in results if r["anomaly"])
    all_z = [r["max_z_score"] for r in results]
    elapsed = time.time() - t0

    print(f"\n  === GROUP {group_id} COMPLETE ===")
    print(f"  Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"  Training: {n_images} images, bank={memory_bank.shape}")
    print(f"  Inference: {n_anomaly}/{len(results)} anomaly")
    if all_z:
        print(f"  Z-score: {min(all_z):.2f} ~ {max(all_z):.2f}")
    print(f"  Time: {elapsed/60:.1f}min")
    print(f"  Heatmaps: {heatmap_dir}")
    sys.stdout.flush()

    del memory_bank, pos_mean, pos_std
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "group": group_id,
        "n_images": n_images,
        "bank_size": meta["memory_bank_shape"][0],
        "n_anomaly": n_anomaly,
        "n_defect_images": len(results),
        "z_range": [min(all_z), max(all_z)] if all_z else None,
        "time_min": elapsed / 60,
    }


# ===== MAIN =====
def main():
    t_start = time.time()
    print("=" * 60)
    print("PatchCore v5b-HALF — HALF-RESOLUTION TRAINING")
    print("=" * 60)
    print(f"Spec: {TARGET_SPEC}")
    print(f"Input: {IMAGE_WIDTH}x{IMAGE_HEIGHT} (from {IMAGE_WIDTH_ORIG}x{IMAGE_HEIGHT_ORIG})")
    print(f"Groups to train: {GROUPS_TO_TRAIN}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Discover folders
    print("[SETUP] Scanning NAS...")
    date_folders, defect_folder = discover_spec_folders(TARGET_SPEC)

    total_folders = sum(len(v) for v in date_folders.values())
    for date_str, folders in sorted(date_folders.items()):
        marker = " *** DEFECT" if date_str == DEFECT_DATE else ""
        print(f"  {date_str}: {len(folders)} folders{marker}")
    print(f"  Total: {total_folders} folders")

    if defect_folder is None:
        print("ERROR: defect folder not found!")
        return
    print(f"  Defect: {defect_folder.name}")

    # Load extractor
    print("\n[SETUP] Loading WideResNet50...")
    extractor = SpatialFeatureExtractor(device="cuda")

    with torch.no_grad():
        dummy = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device="cuda")
        feat = extractor(dummy)
        spatial_size = (feat.shape[2], feat.shape[3])
        print(f"  Input: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        print(f"  Spatial: {spatial_size[0]}x{spatial_size[1]} = {spatial_size[0]*spatial_size[1]} features/image")
        del dummy, feat; torch.cuda.empty_cache()

    # Train each group
    all_results = []
    for group_id in GROUPS_TO_TRAIN:
        result = train_group(group_id, date_folders, defect_folder, extractor, spatial_size)
        if result:
            all_results.append(result)

    # Final summary
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("ALL GROUPS TRAINING COMPLETE (HALF-RES)")
    print(f"{'='*60}")
    for r in all_results:
        z_str = f"{r['z_range'][0]:.2f}~{r['z_range'][1]:.2f}" if r['z_range'] else "N/A"
        print(f"  Group {r['group']}: {r['n_images']} imgs, bank={r['bank_size']}, "
              f"anomaly={r['n_anomaly']}/{r['n_defect_images']}, z={z_str}, {r['time_min']:.1f}min")
    print(f"\n  Total time: {total_time/60:.1f}min ({total_time/3600:.1f}hr)")
    print("DONE")

    with open(OUTPUT_DIR / TARGET_SPEC / "all_groups_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
