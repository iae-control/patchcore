#!/usr/bin/env python3
"""PatchCore v5b-FULL — Train with ALL available 596x199 data.

Strategy (based on NAS inventory):
- 3 dates available: 20250630 (defect day!), 20250831, 20251027
- 20250630: SPARSE sampling (subsample=15) — defects may be mixed in
- 20250831, 20251027: DENSE sampling (subsample=3)
- Date-balanced: equal weight per date to avoid bias from heat treatment conditions
- Total ~17,000 images (was 300)

Pipeline:
1. Discover 596x199 folders grouped by date
2. Extract features with date-aware sampling
3. Coreset selection (4M features → ~12K memory bank)
4. Per-position normalization stats
5. Z-score inference on defect folder
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
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5b_full")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

TRIM_HEAD = 100
TRIM_TAIL = 100

TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"
DEFECT_DATE = "20250630"  # date with known defects

CAMERA_CAMS = [1, 10]
MIRROR_CAM = 10

# Sampling strategy per date
SUBSAMPLE_NORMAL = 3      # dense for safe dates
SUBSAMPLE_DEFECT_DATE = 15 # sparse for defect date (may have mixed defects)

# Memory management
MEMORY_CAP_FEATURES = 8_000_000
CORESET_MAX_FEATURES = 4_000_000
CORESET_RATIO = 0.003
CORESET_PROJECTION_DIM = 128

SELF_VAL_MAD_K = 3.5

# Per-position normalization
NORM_SAMPLE_PER_DATE = 200  # images per date for norm stats

# Heatmap config
GAUSSIAN_SIGMA = 4
SCORE_PERCENTILE_CAP = 99.5
Z_SCORE_THRESHOLD = 3.0


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
    """Discover all folders for target spec, grouped by date."""
    date_folders = defaultdict(list)
    defect_folder = None

    for entry in sorted(NAS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if re.match(r'^\d{8}$', entry.name):
            date_str = entry.name
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and spec_pattern in sub.name:
                        if (sub / "camera_1").is_dir():
                            if DEFECT_FOLDER_PREFIX in sub.name:
                                defect_folder = sub
                            else:
                                date_folders[date_str].append(sub)
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
            print(f"  Loading WideResNet50 from local cache: {local_weights}")
            backbone = wide_resnet50_2(weights=None)
            state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        else:
            print("  Loading WideResNet50 (downloading weights)...")
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


# ===== DATE-BALANCED FEATURE EXTRACTION =====
def extract_features_date_balanced(date_folders, cam_ids, mirror_cam, extractor):
    """Extract features with date-aware sampling and balanced representation."""
    all_features = []
    total_features_count = 0
    total_images = 0
    feat_spatial_size = None
    rng = np.random.RandomState(0)

    date_stats = {}

    for date_str, folders in sorted(date_folders.items()):
        # Choose subsample rate based on date
        if date_str == DEFECT_DATE:
            subsample = SUBSAMPLE_DEFECT_DATE
            date_label = f"{date_str} (DEFECT DATE - sparse)"
        else:
            subsample = SUBSAMPLE_NORMAL
            date_label = f"{date_str} (normal - dense)"

        date_images = 0
        date_features = 0

        print(f"\n  Processing {date_label}: {len(folders)} folders, subsample={subsample}")
        sys.stdout.flush()

        for fi, folder in enumerate(folders):
            for cam_id in cam_ids:
                images = get_image_paths(folder, cam_id, subsample)
                for img_path in images:
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue

                    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
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
                        print(f"  Spatial: {spatial_size[0]}x{spatial_size[1]} "
                              f"= {spatial_size[0]*spatial_size[1]} features/image")

                    total_images += 1
                    date_images += 1

                    # Memory management
                    if total_features_count > MEMORY_CAP_FEATURES:
                        concat = np.concatenate(all_features, axis=0)
                        keep = MEMORY_CAP_FEATURES // 2
                        idx = rng.choice(concat.shape[0], keep, replace=False)
                        all_features = [concat[np.sort(idx)]]
                        total_features_count = keep
                        del concat
                        gc.collect()
                        print(f"    [Memory trim] kept {keep:,} features")

            if (fi + 1) % 10 == 0:
                print(f"    {fi+1}/{len(folders)} folders, {date_images} images")
                sys.stdout.flush()

        date_stats[date_str] = {"images": date_images, "features": date_features,
                                "folders": len(folders), "subsample": subsample}
        print(f"  -> {date_str}: {date_images} images, {date_features:,} features")
        sys.stdout.flush()

    # Print date balance summary
    print(f"\n  === Date Balance Summary ===")
    for d, s in sorted(date_stats.items()):
        print(f"    {d}: {s['images']:>6} images from {s['folders']:>3} folders "
              f"(subsample={s['subsample']})")
    print(f"  Total: {total_images} images")

    # Final concatenation
    result = np.concatenate(all_features, axis=0)
    del all_features
    gc.collect()
    print(f"  Concatenated: {result.shape} ({result.nbytes / 1024**3:.1f}GB)")

    # Subsample if needed
    if result.shape[0] > CORESET_MAX_FEATURES:
        print(f"  Random subsample: {result.shape[0]:,} -> {CORESET_MAX_FEATURES:,}")

        # DATE-BALANCED subsampling: equal proportion from each date
        # (approximate by uniform random since features are interleaved)
        idx = np.sort(rng.choice(result.shape[0], CORESET_MAX_FEATURES, replace=False))
        result = result[idx].copy()
        gc.collect()

    print(f"  Final features: {result.shape} ({result.nbytes / 1024**3:.1f}GB)")
    sys.stdout.flush()
    return result, feat_spatial_size, total_images, date_stats


# ===== CORESET =====
def greedy_coreset(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    print(f"  Coreset: {n:,} -> {target:,} (ratio={ratio})")
    sys.stdout.flush()

    if target >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj = rng.randn(d, proj_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    proj_t = torch.from_numpy(proj).cuda()
    proj_batch = 500000
    projected_list = []
    for i in range(0, n, proj_batch):
        chunk = torch.from_numpy(features[i:i + proj_batch].astype(np.float32)).cuda()
        projected_list.append((chunk @ proj_t).cpu())
        del chunk
    torch.cuda.empty_cache()

    projected = torch.cat(projected_list, dim=0).cuda()
    del projected_list
    print(f"  Projected: {projected.shape} ({projected.nbytes / 1024**3:.1f}GB)")
    sys.stdout.flush()

    selected = [rng.randint(n)]
    min_dists = torch.full((n,), float('inf'), device='cuda')

    for i in tqdm(range(target - 1), desc="Coreset selection", leave=True):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(torch.argmax(min_dists).item())

        if (i + 1) % 2000 == 0:
            print(f"    Coreset: {i+1}/{target-1}")
            sys.stdout.flush()

    del projected
    torch.cuda.empty_cache()

    indices = np.array(selected)
    return features[indices], indices


# ===== SCORING =====
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


# ===== PER-POSITION NORMALIZATION =====
def compute_position_stats(date_folders, cam_ids, mirror_cam, extractor, spatial_size,
                           memory_bank, output_dir):
    """Compute per-position mean/std from balanced date sampling."""
    print(f"\n[NORM] Computing per-position stats ({NORM_SAMPLE_PER_DATE} imgs/date)...")
    Hp, Wp = spatial_size

    score_accum = []
    total_count = 0

    for date_str, folders in sorted(date_folders.items()):
        if date_str == DEFECT_DATE:
            subsample = SUBSAMPLE_DEFECT_DATE
        else:
            subsample = SUBSAMPLE_NORMAL * 2  # slightly sparser for norm stats

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
                    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                        continue
                    if cam_id == mirror_cam:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    tensor = TRANSFORM(img).unsqueeze(0)
                    features, _ = extractor.extract_spatial(tensor)
                    scores = score_spatial(features, memory_bank)
                    score_accum.append(scores)
                    date_count += 1
                    total_count += 1

        print(f"  {date_str}: {date_count} images for norm stats")

    print(f"  Total norm images: {total_count}")
    score_matrix = np.array(score_accum)

    pos_mean = np.mean(score_matrix, axis=0)
    pos_std = np.std(score_matrix, axis=0)
    pos_std = np.clip(pos_std, 0.01, None)

    print(f"  Mean range: {pos_mean.min():.4f} ~ {pos_mean.max():.4f}")
    print(f"  Std range: {pos_std.min():.4f} ~ {pos_std.max():.4f}")

    np.save(output_dir / "pos_mean.npy", pos_mean)
    np.save(output_dir / "pos_std.npy", pos_std)
    print(f"  Saved pos_mean.npy, pos_std.npy")
    sys.stdout.flush()
    return pos_mean, pos_std


# ===== HEATMAP (Z-SCORE) =====
def generate_heatmap_zscore(img_path, memory_bank, extractor, spatial_size, output_path,
                            pos_mean, pos_std, threshold_z, mirror=False):
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
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
        z_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    heatmap_full = gaussian_filter(heatmap_full.astype(np.float64), sigma=GAUSSIAN_SIGMA)

    if mirror:
        heatmap_full = np.fliplr(heatmap_full)

    max_z = float(heatmap_full.max())
    raw_max = float(scores.max())

    # Visualization
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
    overlay_max = max(vmax_z, threshold_z * 2)
    masked = np.ma.masked_where(heatmap_full < overlay_min, heatmap_full)
    im2 = axes[2].imshow(masked, cmap='jet', alpha=0.65,
                          vmin=overlay_min, vmax=overlay_max)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if max_z > threshold_z:
        status = f"ANOMALY (z={max_z:.2f})"
        color = 'red'
    else:
        status = f"NORMAL (z={max_z:.2f})"
        color = 'green'
    axes[2].set_title(f"Overlay: {status}", fontsize=14, color=color, fontweight='bold')
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v5b-FULL | {TARGET_SPEC}/group_1 | {img_path.name} | "
                 f"Z-Score (thr={threshold_z})", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return max_z, raw_max, float(np.mean(scores))


# ===== MAIN =====
def main():
    t_start = time.time()
    print("=" * 60)
    print("PatchCore v5b-FULL — DATE-BALANCED MAXIMUM TRAINING")
    print("=" * 60)
    print(f"Spec: {TARGET_SPEC}")
    print(f"Defect date: {DEFECT_DATE} (sparse sampling)")
    print(f"Normal dates: dense sampling (subsample={SUBSAMPLE_NORMAL})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # [1/7] Discover folders
    print("[1/7] Scanning NAS for 596x199 folders...")
    date_folders, defect_folder = discover_spec_folders(TARGET_SPEC)

    total_folders = sum(len(v) for v in date_folders.values())
    print(f"  Dates with data: {len(date_folders)}")
    for date_str, folders in sorted(date_folders.items()):
        marker = " *** DEFECT DATE" if date_str == DEFECT_DATE else ""
        print(f"    {date_str}: {len(folders)} folders{marker}")
    print(f"  Total training folders: {total_folders}")

    if defect_folder is None:
        print("ERROR: defect folder not found!")
        return
    print(f"  Defect folder: {defect_folder.name}")

    output_dir = OUTPUT_DIR / TARGET_SPEC / "group_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # [2/7] Load extractor
    print("\n[2/7] Loading WideResNet50...")
    extractor = SpatialFeatureExtractor(device="cuda")

    with torch.no_grad():
        dummy = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device="cuda")
        feat = extractor(dummy)
        Hp, Wp = feat.shape[2], feat.shape[3]
        print(f"  Feature map: {feat.shape} -> {Hp}x{Wp} = {Hp*Wp} features/image")
        del dummy, feat
        torch.cuda.empty_cache()

    # [3/7] Extract features (date-balanced)
    print(f"\n[3/7] Extracting features (date-balanced)...")
    t0 = time.time()
    features, spatial_size, n_images, date_stats = extract_features_date_balanced(
        date_folders, CAMERA_CAMS, MIRROR_CAM, extractor
    )
    extract_time = time.time() - t0
    print(f"  Done in {extract_time:.1f}s ({extract_time/60:.1f}min)")
    sys.stdout.flush()

    if features.shape[0] == 0:
        print("ERROR: no features!")
        return

    # [4/7] Coreset selection
    print(f"\n[4/7] Coreset selection...")
    t0 = time.time()
    memory_bank, indices = greedy_coreset(features)
    coreset_time = time.time() - t0
    print(f"  Memory bank: {memory_bank.shape} in {coreset_time:.1f}s ({coreset_time/60:.1f}min)")
    sys.stdout.flush()

    del features
    gc.collect()

    # [5/7] Self-validation
    print("\n[5/7] Self-validation...")
    # Use small sample from each safe date
    val_date_folders = {d: fs[:3] for d, fs in date_folders.items() if d != DEFECT_DATE}
    val_features, _, _, _ = extract_features_date_balanced(
        val_date_folders, CAMERA_CAMS, MIRROR_CAM, extractor
    )
    val_scores = score_spatial(val_features, memory_bank)
    median = np.median(val_scores)
    mad = np.median(np.abs(val_scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    print(f"  Median={median:.4f}, MAD_std={mad_std:.4f}, Threshold={threshold:.4f}")

    plt.figure(figsize=(12, 5))
    plt.hist(val_scores, bins=200, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.4f}')
    plt.axvline(median, color='blue', linestyle='-', alpha=0.5,
                label=f'Median: {median:.4f}')
    plt.title(f"v5b-FULL | {TARGET_SPEC} | {n_images} training images | "
              f"{memory_bank.shape[0]:,} memory bank")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100)
    plt.close()
    del val_features, val_scores

    # Save model + metadata
    np.save(output_dir / "memory_bank.npy", memory_bank)
    np.save(output_dir / "spatial_size.npy", np.array(spatial_size))

    meta = {
        "n_training_images": n_images,
        "date_stats": {d: s for d, s in date_stats.items()},
        "memory_bank_shape": list(memory_bank.shape),
        "spatial_size": list(spatial_size),
        "threshold_mad": float(threshold),
        "median": float(median),
        "mad_std": float(mad_std),
        "extract_time_s": extract_time,
        "coreset_time_s": coreset_time,
        "config": {
            "subsample_normal": SUBSAMPLE_NORMAL,
            "subsample_defect_date": SUBSAMPLE_DEFECT_DATE,
            "coreset_max_features": CORESET_MAX_FEATURES,
            "coreset_ratio": CORESET_RATIO,
        }
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved memory_bank.npy ({memory_bank.shape})")

    # [6/7] Per-position normalization
    pos_mean, pos_std = compute_position_stats(
        date_folders, CAMERA_CAMS, MIRROR_CAM, extractor, spatial_size,
        memory_bank, output_dir
    )

    # [7/7] Inference on defect folder
    print(f"\n[7/7] Inference: {defect_folder.name}")
    print("=" * 60)

    cam1_images = get_image_paths(defect_folder, cam_id=1, subsample_step=1)
    print(f"  Camera 1 images: {len(cam1_images)}")

    heatmap_dir = output_dir / "heatmaps_zscore" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating z-score heatmaps for ALL {len(cam1_images)} images...")
    sys.stdout.flush()

    results = []
    for idx in tqdm(range(len(cam1_images)), desc="Z-Score Heatmaps"):
        img_path = cam1_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_z, raw_max, mean_score = generate_heatmap_zscore(
            img_path, memory_bank, extractor, spatial_size, out_path,
            pos_mean, pos_std, Z_SCORE_THRESHOLD
        )
        if max_z is not None:
            results.append({
                "idx": idx,
                "file": img_path.name,
                "max_z_score": max_z,
                "raw_max_score": raw_max,
                "mean_score": mean_score,
                "anomaly": max_z > Z_SCORE_THRESHOLD,
            })

    with open(output_dir / "inference_results_full.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_anomaly = sum(1 for r in results if r["anomaly"])
    all_z = [r["max_z_score"] for r in results]
    all_raw = [r["raw_max_score"] for r in results]

    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print("TRAINING + INFERENCE COMPLETE (v5b-FULL)")
    print(f"{'='*60}")
    print(f"  Training images: {n_images} (was 300)")
    for d, s in sorted(date_stats.items()):
        print(f"    {d}: {s['images']} images (subsample={s['subsample']})")
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Defect images: {len(results)}")
    print(f"  Anomaly (z>{Z_SCORE_THRESHOLD}): {n_anomaly}/{len(results)}")
    if all_z:
        print(f"  Z-score range: {min(all_z):.2f} ~ {max(all_z):.2f}")
        print(f"  Raw score range: {min(all_raw):.3f} ~ {max(all_raw):.3f}")
    print(f"  Heatmaps: {heatmap_dir}")
    print(f"  Total time: {total_time/60:.1f}min ({total_time/3600:.1f}hr)")
    print(f"{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
