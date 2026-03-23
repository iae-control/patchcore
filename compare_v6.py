"""v6 tile-based: Compare defect vs normal z-score distributions."""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import re
from collections import defaultdict

# ===== Config (must match train_v6_tile.py) =====
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v6/596x199/group_1")
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
DEFECT_DATE = "20250630"
DEFECT_FOLDER_PREFIX = "160852"
NORMAL_DATE = "20250831"
CAM_IDS = [1, 10]
MIRROR_CAM = 10
TILE_SIZE = 128
TILE_STRIDE = 128
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
BRIGHTNESS_THRESHOLD = 30

def tile_positions(w, h, size, stride):
    positions = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            positions.append((x, y))
    return positions

def load_backbone():
    from torchvision.models import wide_resnet50_2
    local_path = Path("/home/dk-sdd/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth")
    model = wide_resnet50_2(weights=None)
    model.load_state_dict(torch.load(local_path, map_location="cpu", weights_only=True))
    model.eval()
    model.cuda()
    features = {}
    def hook(name):
        def fn(module, input, output):
            features[name] = output
        return fn
    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))
    return model, features

def extract_tile_features_single(img, backbone, features_dict, positions):
    """Extract tile features from a single image."""
    tiles = []
    valid_idx = []
    for i, (x, y) in enumerate(positions):
        tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        arr = np.array(tile)
        if arr.mean() < BRIGHTNESS_THRESHOLD:
            continue
        # Normalize
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        tiles.append(t)
        valid_idx.append(i)

    if not tiles:
        return np.array([]), valid_idx

    batch = torch.stack(tiles).cuda()
    with torch.no_grad():
        # Process in sub-batches
        all_feats = []
        bs = 64
        for j in range(0, len(batch), bs):
            sub = batch[j:j+bs]
            _ = backbone(sub)
            l2 = features_dict["layer2"]
            l3 = features_dict["layer3"]
            l3_up = F.interpolate(l3, size=l2.shape[2:], mode="bilinear", align_corners=False)
            cat = torch.cat([l2, l3_up], dim=1)
            pooled = F.adaptive_avg_pool2d(cat, 1).squeeze(-1).squeeze(-1)
            all_feats.append(pooled.cpu().numpy())
        feats = np.concatenate(all_feats, axis=0)
    return feats, valid_idx

def score_tiles(tile_features, memory_bank):
    """Score tiles against memory bank using L2 distance."""
    if len(tile_features) == 0:
        return np.array([])
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    # Batch scoring to avoid OOM
    scores = []
    bs = 256
    for i in range(0, len(tf), bs):
        dists = torch.cdist(tf[i:i+bs], mb)
        min_dists, _ = dists.min(dim=1)
        scores.append(min_dists.cpu().numpy())
    return np.concatenate(scores)

def spatial_heatmap_for_image(img, backbone, features_dict, memory_bank):
    """Generate high-res spatial heatmap using full-image spatial features (no tiling).
    Returns a 2D score map at backbone resolution."""
    img_arr = np.array(img)
    t = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    t = t.unsqueeze(0).cuda()

    with torch.no_grad():
        _ = backbone(t)
        l2 = features_dict["layer2"]   # (1, 512, H2, W2)
        l3 = features_dict["layer3"]   # (1, 1024, H3, W3)
        l3_up = F.interpolate(l3, size=l2.shape[2:], mode="bilinear", align_corners=False)
        cat = torch.cat([l2, l3_up], dim=1)  # (1, 1536, H2, W2)

    feat_h, feat_w = cat.shape[2], cat.shape[3]
    # Reshape to (H*W, 1536)
    spatial_feats = cat.squeeze(0).permute(1, 2, 0).reshape(-1, cat.shape[1])  # (H*W, 1536)

    # Score against memory bank
    mb = torch.from_numpy(memory_bank).cuda()
    scores_list = []
    bs = 512
    for i in range(0, len(spatial_feats), bs):
        dists = torch.cdist(spatial_feats[i:i+bs], mb)
        min_dists, _ = dists.min(dim=1)
        scores_list.append(min_dists.cpu().numpy())
    scores_flat = np.concatenate(scores_list)
    score_map = scores_flat.reshape(feat_h, feat_w)
    return score_map

def make_tile_heatmap(img, scores, valid_idx, positions, n_tiles_x, n_tiles_y, title, output_path, threshold,
                      backbone=None, features_dict=None, memory_bank=None):
    """Draw tile-based heatmap + spatial high-res heatmap overlaid on original image."""
    # Build score grid (NaN for invalid/masked tiles)
    score_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
    for k, idx in enumerate(valid_idx):
        row = idx // n_tiles_x
        col = idx % n_tiles_x
        score_grid[row, col] = scores[k]

    max_score = float(np.nanmax(scores)) if len(scores) > 0 else 0
    status = "ANOMALY" if max_score > threshold else "NORMAL"
    color = "red" if max_score > threshold else "green"
    vmax = max(threshold * 3, np.nanmax(scores)) if len(scores) > 0 else 1

    # Generate spatial heatmap
    has_spatial = backbone is not None and features_dict is not None and memory_bank is not None
    if has_spatial:
        spatial_map = spatial_heatmap_for_image(img, backbone, features_dict, memory_bank)

    n_cols = 4 if has_spatial else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # Tile heatmap overlay
    axes[1].imshow(img)
    heatmap = axes[1].imshow(
        score_grid,
        cmap='jet', alpha=0.5,
        extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
        interpolation='nearest',
        vmin=0, vmax=vmax
    )
    plt.colorbar(heatmap, ax=axes[1], fraction=0.03, pad=0.02)
    axes[1].set_title(f"Tile Heatmap ({n_tiles_x}x{n_tiles_y})", fontsize=12)
    axes[1].axis("off")

    # Tile score grid only
    im = axes[2].imshow(score_grid, cmap='jet', interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=axes[2], fraction=0.05, pad=0.02)
    axes[2].set_title(f"Tile Scores (max={max_score:.3f})", fontsize=12)

    # Spatial high-res heatmap
    if has_spatial:
        axes[3].imshow(img)
        sp_hm = axes[3].imshow(
            spatial_map,
            cmap='jet', alpha=0.5,
            extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
            interpolation='bilinear',
            vmin=0, vmax=vmax
        )
        plt.colorbar(sp_hm, ax=axes[3], fraction=0.03, pad=0.02)
        axes[3].set_title(f"Spatial Heatmap (high-res)", fontsize=12)
        axes[3].axis("off")

    fig.suptitle(f"{title} | max={max_score:.3f} | thr={threshold:.3f} | {status}",
                 fontsize=13, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def process_date_folder(date_str, date_folders, backbone, features_dict, positions, memory_bank,
                        max_folders=None, heatmap_dir=None, threshold=None,
                        n_tiles_x=15, n_tiles_y=9, max_heatmaps=10):
    """Process all images for a date, return per-image max scores."""
    image_scores = []
    folders = date_folders
    if max_folders:
        folders = folders[:max_folders]

    heatmap_count = 0

    for folder in folders:
        for cam_id in CAM_IDS:
            cam_dir = folder / f"camera_{cam_id}"
            if not cam_dir.exists():
                continue
            images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            # Trim first/last 100 images (unstable capture start/end)
            if len(images) > 200:
                images = images[100:-100]
            else:
                continue  # Too few images, skip
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
                    if cam_id == MIRROR_CAM:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    feats, valid_idx = extract_tile_features_single(img, backbone, features_dict, positions)
                    if len(feats) == 0:
                        continue
                    scores = score_tiles(feats, memory_bank)
                    max_score = float(scores.max())
                    image_scores.append({
                        "file": str(img_path.name),
                        "folder": folder.name,
                        "cam": cam_id,
                        "max_score": max_score,
                        "mean_score": float(scores.mean()),
                        "n_tiles": len(scores),
                    })

                    # Generate heatmaps for sample images
                    if heatmap_dir and heatmap_count < max_heatmaps:
                        title = f"{date_str} | {folder.name} | cam{cam_id} | {img_path.name}"
                        out_name = f"heatmap_{date_str}_cam{cam_id}_{folder.name}_{img_path.stem}.png"
                        make_tile_heatmap(img, scores, valid_idx, positions,
                                         n_tiles_x, n_tiles_y, title,
                                         heatmap_dir / out_name, threshold,
                                         backbone=backbone, features_dict=features_dict,
                                         memory_bank=memory_bank)
                        heatmap_count += 1

                except Exception as e:
                    print(f"  Error: {img_path}: {e}")
                    continue

    return image_scores

def main():
    t0 = time.time()
    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    print(f"Tiles per image: {len(positions)}")

    # Load model
    print("Loading backbone...")
    backbone, features_dict = load_backbone()

    # Load memory bank
    memory_bank = np.load(OUTPUT_DIR / "memory_bank.npy")
    print(f"Memory bank: {memory_bank.shape}")

    # Load threshold
    with open(OUTPUT_DIR / "training_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold_mad"]
    print(f"Threshold: {threshold}")

    # Discover folders (same structure as train_v6_tile.py)
    all_dates = defaultdict(list)
    defect_folder = None
    for entry in sorted(NAS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if re.match(r'^\d{8}$', entry.name):
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir() and TARGET_SPEC in sub.name:
                        if (sub / "camera_1").is_dir():
                            if DEFECT_FOLDER_PREFIX in sub.name:
                                defect_folder = sub
                            all_dates[entry.name].append(sub)
            except PermissionError:
                continue
    print(f"Dates found: {sorted(all_dates.keys())}")
    for d, fs in sorted(all_dates.items()):
        print(f"  {d}: {len(fs)} folders")

    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))

    # Heatmap output dirs
    heatmap_defect_dir = OUTPUT_DIR / "heatmaps_defect"
    heatmap_normal_dir = OUTPUT_DIR / "heatmaps_normal"
    heatmap_defect_dir.mkdir(exist_ok=True)
    heatmap_normal_dir.mkdir(exist_ok=True)

    # Defect folder only (20250630160852_596x199)
    if defect_folder is None:
        print("WARNING: defect folder not found!")
        defect_scores = []
    else:
        print(f"\nDefect folder: {defect_folder.name}")
        defect_scores = process_date_folder(DEFECT_DATE, [defect_folder], backbone, features_dict, positions, memory_bank,
                                            heatmap_dir=heatmap_defect_dir, threshold=threshold,
                                            n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, max_heatmaps=10)
        print(f"  Images scored: {len(defect_scores)}")

    # Normal date
    normal_folders = all_dates.get(NORMAL_DATE, [])
    print(f"\nNormal date ({NORMAL_DATE}): {len(normal_folders)} folders")
    normal_scores = process_date_folder(NORMAL_DATE, normal_folders, backbone, features_dict, positions, memory_bank,
                                        max_folders=5, heatmap_dir=heatmap_normal_dir, threshold=threshold,
                                        n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, max_heatmaps=10)
    print(f"  Images scored: {len(normal_scores)}")

    # Stats
    def_max = [s["max_score"] for s in defect_scores]
    nor_max = [s["max_score"] for s in normal_scores]

    print(f"\n=== Distribution ===")
    print(f"Defect  max_score: {np.min(def_max):.4f} ~ {np.max(def_max):.4f} (median={np.median(def_max):.4f})")
    print(f"Normal  max_score: {np.min(nor_max):.4f} ~ {np.max(nor_max):.4f} (median={np.median(nor_max):.4f})")
    print(f"Threshold: {threshold:.4f}")

    n_def_anom = sum(1 for s in def_max if s > threshold)
    n_nor_anom = sum(1 for s in nor_max if s > threshold)
    print(f"Defect anomaly rate: {n_def_anom}/{len(def_max)} ({100*n_def_anom/max(len(def_max),1):.1f}%)")
    print(f"Normal false positive: {n_nor_anom}/{len(nor_max)} ({100*n_nor_anom/max(len(nor_max),1):.1f}%)")

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(def_max, bins=50, alpha=0.6, label=f'Defect ({DEFECT_DATE}) n={len(def_max)}', color='red', edgecolor='darkred')
    ax.hist(nor_max, bins=50, alpha=0.6, label=f'Normal ({NORMAL_DATE}) n={len(nor_max)}', color='blue', edgecolor='darkblue')
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Max Tile Score (L2 distance)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'PatchCore v6 Tile 128x128 | Group 1 | Defect vs Normal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dist_compare_v6.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved: {OUTPUT_DIR / 'dist_compare_v6.png'}")

    # Save results
    results = {"defect": defect_scores, "normal": normal_scores, "threshold": threshold}
    with open(OUTPUT_DIR / "compare_results.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
