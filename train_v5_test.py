#!/usr/bin/env python3
"""PatchCore v5 - Accuracy-first training & inference test.

Changes from v4:
- Full resolution (1920x1200) - no downscaling
- 256x256 tiles, stride 128 (50% overlap) -> ~126 tiles/image (vs 6)
- 1 round self-validation only
- Defect folder excluded from training, used for inference
- Image-by-image processing (efficient I/O)
- Heatmap generation for defect visualization
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
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v5")

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
TILE_SIZE = 256
TILE_STRIDE = 128

TRIM_HEAD = 100
TRIM_TAIL = 100
IMAGE_SUBSAMPLE = 5

CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128
BATCH_SIZE = 128
NUM_WORKERS = 8

SELF_VAL_MAD_K = 3.5
SELF_VAL_MAX_REJECT_PCT = 5.0

TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "20250630160852"
CAMERA_CAMS = [1, 10]
MIRROR_CAM = 10


# ===== UTILITIES =====
def natural_sort_key(path):
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def tile_positions(img_w, img_h, tile_size, stride):
    xs = list(range(0, img_w - tile_size + 1, stride))
    if xs and xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)
    ys = list(range(0, img_h - tile_size + 1, stride))
    if ys and ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)
    return [(x, y) for y in ys for x in xs]


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


# ===== FEATURE EXTRACTOR =====
class FeatureExtractor(nn.Module):
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
        self.to(device)
        self.eval()
        try:
            self._compiled = torch.compile(self._forward_impl, mode="max-autotune")
            with torch.no_grad():
                dummy = torch.randn(2, 3, TILE_SIZE, TILE_SIZE, device=device)
                self._compiled(dummy)
            print("torch.compile: max-autotune OK")
            self._use_compiled = True
        except Exception as e:
            print(f"torch.compile failed: {e}, using eager")
            self._use_compiled = False

    def _forward_impl(self, x):
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return features

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if self._use_compiled:
                out = self._compiled(x)
            else:
                out = self._forward_impl(x)
        return out.float()


# ===== FEATURE EXTRACTION (image-by-image) =====
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features_from_folders(folders, cam_ids, mirror_cam, subsample_step, extractor):
    """Extract features image-by-image (one disk read per image)."""
    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles = len(positions)
    print(f"  Tiles per image: {n_tiles}")

    all_features = []
    total_images = 0
    total_skipped = 0

    for folder in tqdm(folders, desc="Folders"):
        for cam_id in cam_ids:
            images = get_image_paths(folder, cam_id, subsample_step)
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    total_skipped += 1
                    continue

                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    total_skipped += 1
                    continue

                mirror = (cam_id == mirror_cam)
                tiles = []
                for tx, ty in positions:
                    tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                    if mirror:
                        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
                    tiles.append(TRANSFORM(tile))

                batch = torch.stack(tiles)
                for i in range(0, len(batch), BATCH_SIZE):
                    sub = batch[i:i + BATCH_SIZE]
                    feats = extractor(sub)
                    all_features.append(feats.cpu().numpy())

                total_images += 1
                if total_images % 200 == 0:
                    n_feat = sum(f.shape[0] for f in all_features)
                    print(f"  {total_images} images, {n_feat:,} features")

    print(f"  Total: {total_images} images, {total_skipped} skipped")
    if not all_features:
        return np.array([])
    return np.concatenate(all_features, axis=0)


# ===== CORESET =====
def greedy_coreset(features, ratio=CORESET_RATIO, proj_dim=CORESET_PROJECTION_DIM):
    n, d = features.shape
    target = max(1, int(n * ratio))
    if target >= n:
        return features, np.arange(n)

    rng = np.random.RandomState(42)
    proj = rng.randn(d, proj_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    proj_t = torch.from_numpy(proj).cuda()
    feat_t = torch.from_numpy(features.astype(np.float32)).cuda()
    projected = feat_t @ proj_t

    selected = [rng.randint(n)]
    min_dists = torch.full((n,), float('inf'), device='cuda')

    for _ in tqdm(range(target - 1), desc="Coreset selection", leave=False):
        last = projected[selected[-1]]
        dists = torch.sum((projected - last) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(torch.argmax(min_dists).item())

    indices = np.array(selected)
    return features[indices], indices


# ===== SCORING =====
def score_features(features, memory_bank, batch_size=2048):
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
def generate_heatmap(img_path, memory_bank, extractor, output_path):
    """Generate anomaly heatmap for a single full-resolution image."""
    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)

    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        print(f"  WARN: unexpected size {img.size}, skipping")
        return None, None

    tiles = []
    for tx, ty in positions:
        tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
        tiles.append(TRANSFORM(tile))

    batch = torch.stack(tiles)
    all_features = []
    for i in range(0, len(batch), BATCH_SIZE):
        sub = batch[i:i + BATCH_SIZE]
        feats = extractor(sub)
        all_features.append(feats.cpu().numpy())
    features = np.concatenate(all_features, axis=0)

    scores = score_features(features, memory_bank)

    # Build heatmap with overlap averaging
    heatmap = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    count = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

    for idx, (tx, ty) in enumerate(positions):
        heatmap[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE] += scores[idx]
        count[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE] += 1

    count = np.maximum(count, 1)
    heatmap /= count

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    axes[0].imshow(img)
    axes[0].set_title(f"Original: {img_path.name}", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=np.percentile(heatmap, 99))
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Heatmap (max={heatmap.max():.4f})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(np.array(img), alpha=0.6)
    im2 = axes[2].imshow(heatmap, cmap='jet', alpha=0.4,
                          vmin=np.percentile(heatmap, 50),
                          vmax=np.percentile(heatmap, 99))
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    plt.suptitle(f"PatchCore v5 | {TARGET_SPEC}/group_1 | {img_path.name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return float(heatmap.max()), float(np.mean(scores))


# ===== MAIN =====
def main():
    t_start = time.time()
    print("=" * 60)
    print("PatchCore v5 - Accuracy First")
    print("=" * 60)
    print(f"Spec: {TARGET_SPEC}, Group: 1 (cams {CAMERA_CAMS})")
    print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT} (FULL - no downscale)")
    print(f"Tiles: {TILE_SIZE}x{TILE_SIZE}, Stride: {TILE_STRIDE} (50% overlap)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 1. Discover folders
    print("[1/6] Scanning NAS...")
    all_folders = discover_spec_folders(TARGET_SPEC)
    print(f"  Found {len(all_folders)} total folders for {TARGET_SPEC}")

    # Separate defect folder
    defect_folder = None
    train_folders = []
    for f in all_folders:
        if DEFECT_FOLDER_PREFIX in f.name:
            defect_folder = f
            print(f"  DEFECT folder (excluded from training): {f.name}")
        else:
            train_folders.append(f)

    print(f"  Training folders: {len(train_folders)}")
    if defect_folder is None:
        print("ERROR: defect folder not found!")
        return
    if len(train_folders) < 5:
        print(f"WARNING: only {len(train_folders)} training folders")

    # 2. Setup output
    output_dir = OUTPUT_DIR / TARGET_SPEC / "group_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    print(f"  Tile positions: {len(positions)} per image")

    # 3. Load backbone
    print("\n[2/6] Loading WideResNet50...")
    extractor = FeatureExtractor(device="cuda")

    # 4. Extract features
    print(f"\n[3/6] Extracting features (subsample={IMAGE_SUBSAMPLE})...")
    t0 = time.time()
    features = extract_features_from_folders(
        train_folders, CAMERA_CAMS, MIRROR_CAM, IMAGE_SUBSAMPLE, extractor
    )
    print(f"  Features shape: {features.shape} in {time.time() - t0:.1f}s")

    if features.shape[0] == 0:
        print("ERROR: no features extracted!")
        return

    # 5. Coreset
    print(f"\n[4/6] Coreset selection (ratio={CORESET_RATIO})...")
    t0 = time.time()
    memory_bank, indices = greedy_coreset(features)
    print(f"  Memory bank: {memory_bank.shape} in {time.time() - t0:.1f}s")

    # 6. Self-validation (1 round)
    print("\n[5/6] Self-validation (1 round)...")
    scores = score_features(features, memory_bank)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    mad_std = 1.4826 * mad
    threshold = median + SELF_VAL_MAD_K * mad_std
    pct_cap = np.percentile(scores, 100 - SELF_VAL_MAX_REJECT_PCT)
    effective = max(threshold, pct_cap)
    n_reject = int((scores >= effective).sum())

    print(f"  Scores: median={median:.4f}, MAD_std={mad_std:.4f}")
    print(f"  Threshold: {effective:.4f}")
    print(f"  Rejected: {n_reject} / {len(scores)} ({n_reject / len(scores) * 100:.2f}%)")

    # Save histogram
    plt.figure(figsize=(12, 5))
    plt.hist(scores, bins=150, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(effective, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {effective:.4f}')
    plt.axvline(median, color='blue', linestyle='-', alpha=0.5,
                label=f'Median: {median:.4f}')
    plt.title(f"v5 Training Scores | {TARGET_SPEC}/group_1 | "
              f"{features.shape[0]:,} tiles | Rejected: {n_reject}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "self_val_hist.png", dpi=100)
    plt.close()

    # Save model
    model_path = output_dir / "memory_bank.npy"
    np.save(model_path, memory_bank)
    print(f"  Saved: {model_path} ({memory_bank.shape})")

    # 7. Inference on defect folder
    print(f"\n[6/6] Inference on defect folder: {defect_folder.name}")
    print("=" * 60)

    cam1_images = get_image_paths(defect_folder, cam_id=1, subsample_step=1)
    print(f"  Camera 1 images (after trim): {len(cam1_images)}")

    heatmap_dir = output_dir / "heatmaps" / defect_folder.name
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Sample: first 5, last 5, every 100th, plus middle section
    n = len(cam1_images)
    sample_idx = set()
    sample_idx.update(range(min(5, n)))                    # first 5
    sample_idx.update(range(max(0, n - 5), n))             # last 5
    sample_idx.update(range(0, n, 100))                    # every 100th
    sample_idx.update(range(n // 4, n // 4 + 5))          # quarter point
    sample_idx.update(range(n // 2, min(n // 2 + 5, n)))  # middle
    sample_idx.update(range(3 * n // 4, min(3 * n // 4 + 5, n)))  # 3/4 point
    sample_indices = sorted(sample_idx)

    print(f"  Generating heatmaps for {len(sample_indices)} images...")

    results = []
    for idx in tqdm(sample_indices, desc="Heatmaps"):
        img_path = cam1_images[idx]
        out_path = heatmap_dir / f"heatmap_{idx:04d}_{img_path.stem}.png"
        max_score, mean_score = generate_heatmap(
            img_path, memory_bank, extractor, out_path
        )
        if max_score is not None:
            results.append({
                "idx": idx,
                "file": img_path.name,
                "max_score": max_score,
                "mean_score": mean_score,
            })

    # Save results
    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"{'Idx':>5} {'File':>20} {'MaxScore':>10} {'MeanScore':>10}")
    for r in results:
        flag = " *** HIGH" if r["max_score"] > np.percentile(
            [x["max_score"] for x in results], 90) else ""
        print(f"{r['idx']:5d} {r['file']:>20} {r['max_score']:10.4f} "
              f"{r['mean_score']:10.4f}{flag}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Heatmaps: {heatmap_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
