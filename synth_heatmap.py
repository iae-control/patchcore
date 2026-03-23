#!/usr/bin/env python3
"""Synthetic defect heatmap generator.
For each camera, pick a few sample images, draw synthetic defects at each level,
and produce tile-level heatmaps (10×4 grid) showing anomaly scores.
Also overlay the synthetic defect line on the original image for comparison.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re, time, random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import wide_resnet50_2
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v9_percam/596x199")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v9_percam/596x199/synth_heatmaps")
TARGET_SPEC = "596x199"
TRAIN_DATES = ["20250831", "20251027"]

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1200
TILE_W, TILE_H = 192, 300
N_COLS, N_ROWS = 10, 4
TRIM_HEAD, TRIM_TAIL = 100, 100
SUBSAMPLE_STEP = 3
VAL_OFFSET = 1
BATCH_SIZE = 64

LINE_WIDTH = 5
LINE_LENGTH_MIN = 80
LINE_LENGTH_MAX = 250

N_SAMPLE_IMAGES = 5  # heatmaps per camera

LEVELS = {
    0: ("normal", None),
    1: ("black", 0.0),
    3: ("medium_50pct", 0.50),
    5: ("subtle_90pct", 0.90),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]

def get_tile_positions():
    positions = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            positions.append((col * TILE_W, row * TILE_H, col, row))
    return positions

def get_holdout_images(folder, cam_id):
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')],
        key=natural_sort_key
    )
    if len(images) <= TRIM_HEAD + TRIM_TAIL:
        return []
    images = images[TRIM_HEAD:len(images) - TRIM_TAIL]
    return images[VAL_OFFSET::SUBSAMPLE_STEP]


def draw_synthetic_line(img_array, brightness_ratio):
    """Draw line and return (modified_array, line_info_dict)."""
    h, w = img_array.shape[:2]
    cx = random.randint(w // 6, w * 5 // 6)
    cy = random.randint(h // 6, h * 5 // 6)
    length = random.randint(LINE_LENGTH_MIN, LINE_LENGTH_MAX)
    angle_offset = random.randint(-15, 15)

    region_y1 = max(0, cy - length // 2)
    region_y2 = min(h, cy + length // 2)
    region_x1 = max(0, cx - 20)
    region_x2 = min(w, cx + 20)
    region = img_array[region_y1:region_y2, region_x1:region_x2]
    if region.size == 0:
        return img_array, None

    local_avg = float(np.mean(region))
    line_brightness = int(local_avg * brightness_ratio)

    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    y1 = cy - length // 2
    y2 = cy + length // 2
    x1 = cx - angle_offset // 2
    x2 = cx + angle_offset // 2
    color = (line_brightness, line_brightness, line_brightness)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=LINE_WIDTH)

    line_info = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "brightness": line_brightness}
    return np.array(img_pil), line_info


class TileFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        weights_path = Path("/home/dk-sdd/patchcore/models/wide_resnet50_2.pth")
        if weights_path.exists():
            model = wide_resnet50_2(weights=None)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            model = wide_resnet50_2(weights="IMAGENET1K_V1")
        model.eval()
        self.layer2 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                     model.layer1, model.layer2)
        self.layer3 = nn.Sequential(model.layer3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to(device)

    @torch.no_grad()
    def __call__(self, batch):
        batch = batch.to(self.device)
        f2 = self.layer2(batch)
        f3 = self.layer3(f2)
        f2_pooled = self.pool(f2).flatten(1)
        f3_pooled = self.pool(f3).flatten(1)
        return torch.cat([f2_pooled, f3_pooled], dim=1)


def score_tiles_gpu(features, memory_bank, device="cuda"):
    feat_t = torch.from_numpy(features).float().to(device)
    bank_t = torch.from_numpy(memory_bank).float().to(device)
    scores = []
    chunk = 256
    for i in range(0, len(feat_t), chunk):
        d = torch.cdist(feat_t[i:i+chunk], bank_t)
        scores.append(d.min(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def get_tile_scores(img_pil, extractor, memory_bank, all_positions):
    """Return per-tile scores as (N_ROWS, N_COLS) array."""
    tiles = []
    for tx, ty, col, row in all_positions:
        tile = img_pil.crop((tx, ty, tx + TILE_W, ty + TILE_H))
        tiles.append(TRANSFORM(tile))
    all_feats = []
    for bs in range(0, len(tiles), BATCH_SIZE):
        batch = torch.stack(tiles[bs:bs + BATCH_SIZE])
        feats = extractor(batch).cpu().numpy()
        all_feats.append(feats)
    feats = np.concatenate(all_feats, axis=0)
    raw_scores = score_tiles_gpu(feats, memory_bank)

    # Reshape to grid
    score_grid = np.zeros((N_ROWS, N_COLS))
    for idx, (tx, ty, col, row) in enumerate(all_positions):
        score_grid[row, col] = raw_scores[idx]
    return score_grid


def plot_heatmap_comparison(cam_id, img_idx, img_array, level_results, output_dir):
    """Plot one figure per sample image: original + heatmaps for each level side by side."""
    n_levels = len(level_results)
    fig, axes = plt.subplots(2, n_levels, figsize=(5 * n_levels, 10))

    # Collect all scores for consistent colorbar
    all_scores = []
    for lr in level_results:
        all_scores.extend(lr["score_grid"].flatten().tolist())
    vmin = min(all_scores)
    vmax = max(all_scores)

    for col_idx, lr in enumerate(level_results):
        level_name = lr["level_name"]
        score_grid = lr["score_grid"]
        display_img = lr["display_img"]
        line_info = lr.get("line_info")

        # Top row: image with defect line highlighted
        ax_img = axes[0, col_idx] if n_levels > 1 else axes[0]
        ax_img.imshow(display_img)
        ax_img.set_title(f"{level_name}", fontsize=11, fontweight="bold")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Draw tile grid
        for r in range(N_ROWS + 1):
            ax_img.axhline(y=r * TILE_H, color="white", linewidth=0.5, alpha=0.5)
        for c in range(N_COLS + 1):
            ax_img.axvline(x=c * TILE_W, color="white", linewidth=0.5, alpha=0.5)

        # Highlight defect line
        if line_info:
            ax_img.plot([line_info["x1"], line_info["x2"]],
                       [line_info["y1"], line_info["y2"]],
                       color="red", linewidth=2, linestyle="--")

        # Bottom row: heatmap
        ax_heat = axes[1, col_idx] if n_levels > 1 else axes[1]
        im = ax_heat.imshow(score_grid, cmap="hot", interpolation="nearest",
                           vmin=vmin, vmax=vmax, aspect="auto")

        # Annotate scores
        for r in range(N_ROWS):
            for c in range(N_COLS):
                val = score_grid[r, c]
                text_color = "white" if val > (vmin + vmax) / 2 else "black"
                ax_heat.text(c, r, f"{val:.3f}", ha="center", va="center",
                           fontsize=7, color=text_color)

        ax_heat.set_title(f"mean={np.mean(score_grid):.4f}, max={np.max(score_grid):.4f}",
                         fontsize=9)
        ax_heat.set_xlabel("Column")
        ax_heat.set_ylabel("Row")
        ax_heat.set_xticks(range(N_COLS))
        ax_heat.set_yticks(range(N_ROWS))

    fig.suptitle(f"Camera {cam_id} — Sample {img_idx+1} — Tile Heatmaps (5px line)",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, label="NN distance")

    fname = f"cam{cam_id}_sample{img_idx+1}_heatmap.png"
    fig.savefig(output_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Synthetic Defect Heatmap Generator — line_width={LINE_WIDTH}px")
    print(f"Levels: {[(k, v[0]) for k, v in LEVELS.items()]}")
    print(f"Sample images per camera: {N_SAMPLE_IMAGES}")
    print("=" * 60)

    all_positions = get_tile_positions()
    random.seed(42)
    np.random.seed(42)

    val_folders = []
    for date_str in TRAIN_DATES:
        date_dir = NAS_ROOT / date_str
        if not date_dir.is_dir():
            continue
        for sub in sorted(date_dir.iterdir()):
            if sub.is_dir() and TARGET_SPEC in sub.name and (sub / "camera_1").is_dir():
                val_folders.append(sub)

    print(f"Validation folders: {len(val_folders)}")
    print("Loading backbone...")
    extractor = TileFeatureExtractor()

    test_cameras = [1, 2, 4, 5, 6, 7, 9, 10]

    for cam_id in test_cameras:
        cam_dir = MODEL_DIR / f"camera_{cam_id}"
        if not cam_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  CAMERA {cam_id}")
        print(f"{'='*60}")

        memory_bank = np.load(cam_dir / "memory_bank.npy")
        print(f"  Memory bank: {memory_bank.shape}")

        all_images = []
        for folder in val_folders:
            all_images.extend(get_holdout_images(folder, cam_id))

        # Pick evenly spaced samples
        if len(all_images) > N_SAMPLE_IMAGES:
            step = len(all_images) // N_SAMPLE_IMAGES
            sample_images = all_images[::step][:N_SAMPLE_IMAGES]
        else:
            sample_images = all_images[:N_SAMPLE_IMAGES]
        print(f"  Sample images: {len(sample_images)}")

        for img_idx, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    continue
                img_array = np.array(img)
            except:
                continue

            print(f"\n  Sample {img_idx+1}: {img_path.name}")
            level_results = []

            for level_id, (level_name, brightness_ratio) in LEVELS.items():
                random.seed(42 + level_id + img_idx * 100)

                if brightness_ratio is not None:
                    modified, line_info = draw_synthetic_line(img_array.copy(), brightness_ratio)
                    img_pil = Image.fromarray(modified)
                    display_img = modified
                else:
                    img_pil = Image.fromarray(img_array)
                    display_img = img_array
                    line_info = None

                score_grid = get_tile_scores(img_pil, extractor, memory_bank, all_positions)
                print(f"    {level_name}: mean={np.mean(score_grid):.4f}, max={np.max(score_grid):.4f}")

                level_results.append({
                    "level_name": level_name,
                    "score_grid": score_grid,
                    "display_img": display_img,
                    "line_info": line_info,
                })

            plot_heatmap_comparison(cam_id, img_idx, img_array, level_results, OUTPUT_DIR)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone! Heatmaps saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
