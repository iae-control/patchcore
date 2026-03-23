#!/usr/bin/env python3
"""PatchCore v5b - Proper spatial features (no global avg pool).

Original PatchCore paper approach:
- Full resolution image → backbone → spatial feature map
- Each spatial position = one feature vector in memory bank
- Pixel-level anomaly heatmap via spatial scoring
- NO tiling, NO global average pooling
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, time, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5b")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

# Spatial pooling to manage memory (pool layer2 features)
# layer2 native: 240x150, pool 3x3 stride 3 -> 80x50 = 4000 features/image
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

TRIM_HEAD = 100
TRIM_TAIL = 100
IMAGE_SUBSAMPLE = 15  # more aggressive subsampling (full-res spatial compensates)
MAX_TRAIN_FOLDERS = 30  # limit folders for memory management

CORESET_RATIO = 0.005
CORESET_PROJECTION_DIM = 128
CORESET_MAX_FEATURES = 2_000_000  # random subsample before coreset

SELF_VAL_MAD_K = 3.5

TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"
CAMERA_CAMS = [1, 10]
MIRROR_CAM = 10


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
    folders = []
    for entry in sorted(NAS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if re.match(r'^\d{8}$', entry.name):
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and spec_pattern in sub.name:
                        if (sub / "camera_1").is_dir():
                            folders.append(sub)
            except PermissionError:
                continue
        elif spec_pattern in entry.name:
            if (entry / "camera_1").is_dir():
                folders.append(entry)
    return folders


# ===== FEATURE EXTRACTOR (spatial - no global avg pool) =====
class SpatialFeatureExtractor(nn.Module):
    """Extract spatial feature maps from WideResNet50.

    Returns (B, 1536, H/8, W/8) feature maps - NOT pooled to single vector.
    This is the proper PatchCore approach.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Spatial average pooling (NOT global)
        self.spatial_pool = nn.AvgPool2d(
            kernel_size=SPATIAL_POOL_K, stride=SPATIAL_POOL_S
        )

        self.to(device)
        self.eval()
        print(f"  SpatialFeatureExtractor on {device}")
        print(f"  Spatial pool: {SPATIAL_POOL_K}x{SPATIAL_POOL_K}, stride {SPATIAL_POOL_S}")

    @torch.no_grad()
    def forward(self, x):
        """Extract spatial features. Returns (N_spatial, 1536) per image."""
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = self.layer1(x)
            f2 = self.layer2(h)        # (B, 512, H/8, W/8)
            f3 = self.layer3(f2)       # (B, 1024, H/16, W/16)

            # Upsample layer3 to match layer2 spatial size
            f3_up = F.interpolate(f3, size=f2.shape[2:],
                                  mode="bilinear", align_corners=False)

            # Concat: (B, 1536, H/8, W/8)
            features = torch.cat([f2, f3_up], dim=1)

            # Spatial pooling to reduce memory (NOT global)
            features = self.spatial_pool(features)

        features = features.float()
        return features  # (B, 1536, Hp, Wp)

    def extract_spatial(self, x):
        """Extract and reshape to (N_spatial, 1536). Single image."""
        feat_map = self.forward(x)  # (1, 1536, Hp, Wp)
        B, C, H, W = feat_map.shape
        # Reshape: (B, C, H, W) -> (B*H*W, C)
        features = feat_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return features.cpu().numpy(), (H, W)


# ===== IMAGE TRANSFORM =====
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== FEATURE EXTRACTION =====
def extract_all_spatial_features(folders, cam_ids, mirror_cam, subsample_step, extractor):
    """Extract spatial features from all training images."""
    all_features = []
    total_images = 0
    feat_spatial_size = None

    for folder in tqdm(folders, desc="Folders"):
        for cam_id in cam_ids:
            images = get_image_paths(folder, cam_id, subsample_step)
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue

                if cam_id == mirror_cam:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                tensor = TRANSFORM(img).unsqueeze(0)  # (1, 3, H, W)
                features, spatial_size = extractor.extract_spatial(tensor)
                all_features.append(features)

                if feat_spatial_size is None:
                    feat_spatial_size = spatial_size
                    print(f"  Spatial feature size: {spatial_size[0]}x{spatial_size[1]} "
                          f"= {spatial_size[0] * spatial_size[1]} features/image")

                total_images += 1
                if total_images % 100 == 0:
                    n_feat = sum(f.shape[0] for f in all_features)
                    mem_gb = n_feat * 1536 * 4 / 1024**3
                    print(f"  {total_images} images, {n_feat:,} features ({mem_gb:.1f}GB)")

    print(f"  Total: {total_images} images")
    sys.stdout.flush()

    # Memory-efficient: subsample directly from list (avoid 65GB concatenate)
    total_feats = sum(f.shape[0] for f in all_features)
    print(f"  Total features: {total_feats:,} across {len(all_features)} arrays")

    if CORESET_MAX_FEATURES and total_feats > CORESET_MAX_FEATURES:
        print(f"  Direct subsample: {total_feats:,} -> {CORESET_MAX_FEATURES:,}")
        sys.stdout.flush()
        rng_sub = np.random.RandomState(0)
        idx_global = np.sort(rng_sub.choice(total_feats, CORESET_MAX_FEATURES, replace=False))

        # Map global indices to (array_idx, local_idx)
        result = np.empty((CORESET_MAX_FEATURES, all_features[0].shape[1]), dtype=np.float32)
        cumsum = 0
        write_pos = 0
        for arr in all_features:
            arr_start = cumsum
            arr_end = cumsum + arr.shape[0]
            # Find indices that fall in this array
            mask = (idx_global >= arr_start) & (idx_global < arr_end)
            local_idx = idx_global[mask] - arr_start
            if len(local_idx) > 0:
                result[write_pos:write_pos + len(local_idx)] = arr[local_idx]
                write_pos += len(local_idx)
            cumsum = arr_end
        del all_features
        import gc; gc.collect()
        print(f"  Subsampled array: {result.shape} ({result.nbytes / 1024**3:.1f}GB)")
        sys.stdout.flush()
        return result, feat_spatial_size, True  # True = already subsampled
    else:
        result = np.concatenate(all_features, axis=0)
        del all_features
        import gc; gc.collect()
        return result, feat_spatial_size, False


# ===== CORESET =====
def greedy_coreset(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM,
                   already_subsampled=False):
    n, d = features.shape
    target = max(1, int(n * ratio))
    print(f"  Coreset: {n:,} -> {target:,} (ratio={ratio})")
    sys.stdout.flush()

    if target >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj = rng.randn(d, proj_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    # Project features in batches on GPU (full features stay on CPU)
    proj_t = torch.from_numpy(proj).cuda()
    proj_batch = 500000
    projected_list = []
    for i in range(0, n, proj_batch):
        chunk = torch.from_numpy(features[i:i + proj_batch].astype(np.float32)).cuda()
        projected_list.append((chunk @ proj_t).cpu())
        del chunk
    torch.cuda.empty_cache()

    # Keep projected features on GPU (n x 128 = manageable)
    projected = torch.cat(projected_list, dim=0).cuda()
    del projected_list
    print(f"  Projected to GPU: {projected.shape} ({projected.nbytes / 1024**3:.1f}GB)")
    sys.stdout.flush()

    selected = [rng.randint(n)]
    min_dists = torch.full((n,), float('inf'), device='cuda')

    for i in tqdm(range(target - 1), desc="Coreset selection", leave=False):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(torch.argmax(min_dists).item())

    del projected
    torch.cuda.empty_cache()

    indices = np.array(selected)
    return features[indices], indices


# ===== SCORING =====
def score_spatial(features, memory_bank, batch_size=4096):
    """Score spatial features against memory bank. Returns per-position scores."""
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(memory_bank).cuda()

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i + batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        min_d, _ = dists.min(dim=1)
        scores[i:i + batch_size] = min_d.cpu().numpy()

    return scores


# ===== HEATMAP GENERATION =====
def generate_heatmap(img_path, memory_bank, extractor, spatial_size, output_path,
                     mirror=False):
    """Generate pixel-level anomaly heatmap for a single image."""
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        return None, None

    img_display = img.copy()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = TRANSFORM(img).unsqueeze(0)
    features, _ = extractor.extract_spatial(tensor)

    scores = score_spatial(features, memory_bank)

    # Reshape scores to spatial grid
    Hp, Wp = spatial_size
    score_map = scores.reshape(Hp, Wp)

    # Upsample to full image resolution using bilinear interpolation
    score_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    heatmap_full = F.interpolate(
        score_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    if mirror:
        heatmap_full = np.fliplr(heatmap_full)

    # Visualize
    img_arr = np.array(img_display)
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    axes[0].imshow(img_arr)
    axes[0].set_title(f"Original: {img_path.name}", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap_full, cmap='hot',
                          vmin=np.percentile(heatmap_full, 1),
                          vmax=np.percentile(heatmap_full, 99))
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Heatmap (max={heatmap_full.max():.4f})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(img_arr, alpha=0.5)
    im2 = axes[2].imshow(heatmap_full, cmap='jet', alpha=0.5,
                          vmin=np.percentile(heatmap_full, 50),
                          vmax=np.percentile(heatmap_full, 99))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_title("Overlay (pixel-level)", fontsize=10)
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v5b SPATIAL | {TARGET_SPEC}/group_1 | {img_path.name}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return float(heatmap_full.max()), float(np.mean(scores))


# ===== MAIN =====
def main():
    t_start = time.time()
    print("=" * 60)
    print("PatchCore v5b - SPATIAL (Proper Paper Implementation)")
    print("=" * 60)
    print(f"Spec: {TARGET_SPEC}, Group: 1 (cams {CAMERA_CAMS})")
    print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT} (FULL)")
    print(f"NO tiling, NO global avg pool")
    print(f"Spatial pool: {SPATIAL_POOL_K}x{SPATIAL_POOL_K} stride {SPATIAL_POOL_S}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 1. Discover folders
    print("[1/6] Scanning NAS...")
    all_folders = discover_spec_folders(TARGET_SPEC)
    print(f"  Found {len(all_folders)} total folders")

    defect_folder = None
    train_folders = []
    for f in all_folders:
        if DEFECT_FOLDER_PREFIX in f.name:
            defect_folder = f
        else:
            train_folders.append(f)

    if defect_folder is None:
        print("ERROR: defect folder not found!")
        return

    # Limit training folders for memory
    if len(train_folders) > MAX_TRAIN_FOLDERS:
        step = len(train_folders) // MAX_TRAIN_FOLDERS
        train_folders = train_folders[::step][:MAX_TRAIN_FOLDERS]

    print(f"  Defect: {defect_folder.name}")
    print(f"  Training: {len(train_folders)} folders (subsampled)")

    # 2. Setup
    output_dir = OUTPUT_DIR / TARGET_SPEC / "group_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load extractor
    print("\n[2/6] Loading WideResNet50 (spatial mode)...")
    extractor = SpatialFeatureExtractor(device="cuda")

    # Quick test to verify spatial size
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device="cuda")
        feat = extractor(dummy)
        print(f"  Feature map shape: {feat.shape}")
        Hp, Wp = feat.shape[2], feat.shape[3]
        print(f"  Spatial: {Hp}x{Wp} = {Hp * Wp} features per image")
        del dummy, feat
        torch.cuda.empty_cache()

    # 4. Extract features
    print(f"\n[3/6] Extracting spatial features (subsample={IMAGE_SUBSAMPLE})...")
    t0 = time.time()
    features, spatial_size, already_sub = extract_all_spatial_features(
        train_folders, CAMERA_CAMS, MIRROR_CAM, IMAGE_SUBSAMPLE, extractor
    )
    print(f"  Features: {features.shape} in {time.time() - t0:.1f}s")
    print(f"  Memory: {features.nbytes / 1024**3:.1f}GB")
    sys.stdout.flush()

    if features.shape[0] == 0:
        print("ERROR: no features!")
        return

    # 5. Coreset
    print(f"\n[4/6] Coreset selection...")
    sys.stdout.flush()
    t0 = time.time()
    memory_bank, indices = greedy_coreset(features, already_subsampled=already_sub)
    print(f"  Memory bank: {memory_bank.shape} in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # Free full features
    del features
    import gc; gc.collect()

    # 6. Self-validation stats
    print("\n[5/6] Scoring training data for stats...")
    # Re-extract a small sample for validation stats
    sample_folders = train_folders[:5]
    val_features, _, _ = extract_all_spatial_features(
        sample_folders, CAMERA_CAMS, MIRROR_CAM, IMAGE_SUBSAMPLE * 3, extractor
    )
    val_scores = score_spatial(val_features, memory_bank)
    median = np.median(val_scores)
    mad = np.median(np.abs(val_scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    print(f"  Validation scores: median={median:.4f}, MAD_std={mad_std:.4f}")
    print(f"  Suggested threshold: {threshold:.4f}")

    # Save histogram
    plt.figure(figsize=(12, 5))
    plt.hist(val_scores, bins=200, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.4f}')
    plt.axvline(median, color='blue', linestyle='-', alpha=0.5,
                label=f'Median: {median:.4f}')
    plt.title(f"v5b SPATIAL Scores | {TARGET_SPEC}/group_1 | {val_features.shape[0]:,} features")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100)
    plt.close()
    del val_features, val_scores

    # Save model
    model_path = output_dir / "memory_bank.npy"
    np.save(model_path, memory_bank)
    np.save(output_dir / "spatial_size.npy", np.array(spatial_size))
    print(f"  Saved: {model_path} ({memory_bank.shape})")

    # 7. Inference on defect folder
    print(f"\n[6/6] Inference: {defect_folder.name}")
    print("=" * 60)

    cam1_images = get_image_paths(defect_folder, cam_id=1, subsample_step=1)
    print(f"  Camera 1 images: {len(cam1_images)}")

    heatmap_dir = output_dir / "heatmaps" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Sample images: first 5, last 5, every 100th, quarter points
    n = len(cam1_images)
    sample_idx = set()
    sample_idx.update(range(min(5, n)))
    sample_idx.update(range(max(0, n - 5), n))
    sample_idx.update(range(0, n, 100))
    sample_idx.update(range(n // 4, min(n // 4 + 3, n)))
    sample_idx.update(range(n // 2, min(n // 2 + 3, n)))
    sample_idx.update(range(3 * n // 4, min(3 * n // 4 + 3, n)))
    sample_indices = sorted(sample_idx)

    print(f"  Generating heatmaps for {len(sample_indices)} images...")

    results = []
    for idx in tqdm(sample_indices, desc="Heatmaps"):
        img_path = cam1_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_score, mean_score = generate_heatmap(
            img_path, memory_bank, extractor, spatial_size, out_path
        )
        if max_score is not None:
            results.append({
                "idx": idx,
                "file": img_path.name,
                "max_score": max_score,
                "mean_score": mean_score,
            })

    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    all_max = [r["max_score"] for r in results]
    p90 = np.percentile(all_max, 90) if all_max else 0

    print(f"\n{'='*60}")
    print("INFERENCE RESULTS (v5b SPATIAL)")
    print(f"{'='*60}")
    print(f"{'Idx':>5} {'File':>20} {'MaxScore':>10} {'MeanScore':>10}")
    for r in results:
        flag = " *** HIGH" if r["max_score"] > p90 else ""
        print(f"{r['idx']:5d} {r['file']:>20} {r['max_score']:10.4f} "
              f"{r['mean_score']:10.4f}{flag}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Heatmaps: {heatmap_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
