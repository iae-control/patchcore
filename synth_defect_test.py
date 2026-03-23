#!/usr/bin/env python3
"""Synthetic defect test v2 — 5px lines, histogram output.
Draws line defects on normal images, saves score histograms.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, re, time, random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v9_percam/596x199")
OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v9_percam/596x199/synth_test")
TARGET_SPEC = "596x199"
TRAIN_DATES = ["20250831", "20251027"]

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1200
TILE_W, TILE_H = 192, 300
N_COLS, N_ROWS = 10, 4
TRIM_HEAD, TRIM_TAIL = 100, 100
BATCH_SIZE = 64
SUBSAMPLE_STEP = 3
VAL_OFFSET = 1

LINE_WIDTH = 5
LINE_LENGTH_MIN = 80
LINE_LENGTH_MAX = 250
N_LINES = 1

LEVELS = {
    0: ("normal", None),
    1: ("black", 0.0),
    2: ("dark_25pct", 0.25),
    3: ("medium_50pct", 0.50),
    4: ("light_75pct", 0.75),
    5: ("subtle_90pct", 0.90),
    6: ("near_invis_95pct", 0.95),
}

N_TEST_IMAGES = 500

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
        return img_array

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
    return np.array(img_pil)


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


def score_image(img_pil, extractor, memory_bank, all_positions):
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
    return score_tiles_gpu(feats, memory_bank)


def plot_histograms(cam_id, cam_results, output_dir):
    """Plot histograms for mean_score and max_score, all levels overlaid."""
    for score_key in ["mean_score", "max_score"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {
            "normal": "gray",
            "black": "red",
            "dark_25pct": "orangered",
            "medium_50pct": "orange",
            "light_75pct": "gold",
            "subtle_90pct": "yellowgreen",
            "near_invis_95pct": "lightgreen",
        }

        for level_name, res in cam_results.items():
            vals = res[score_key]["values"]
            alpha = 0.7 if level_name == "normal" else 0.4
            lw = 2 if level_name in ["normal", "black"] else 1
            ax.hist(vals, bins=60, alpha=alpha, label=level_name,
                    color=colors.get(level_name, "blue"), density=True, histtype="stepfilled")

        ax.set_xlabel(f"{score_key} (raw NN distance)")
        ax.set_ylabel("Density")
        ax.set_title(f"Camera {cam_id} — {score_key} distribution (line_width={LINE_WIDTH}px)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"cam{cam_id}_{score_key}_hist.png", dpi=120)
        plt.close(fig)
        print(f"    Saved: cam{cam_id}_{score_key}_hist.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Synthetic Defect Test — line_width={LINE_WIDTH}px")
    print(f"Levels: {[(k, v[0]) for k, v in LEVELS.items()]}")
    print(f"Test images per camera: {N_TEST_IMAGES}")
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
    all_results = {}

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
        if len(all_images) > N_TEST_IMAGES:
            step = len(all_images) // N_TEST_IMAGES
            all_images = all_images[::step][:N_TEST_IMAGES]
        print(f"  Test images: {len(all_images)}")

        # Pre-load images to ensure same base images for all levels
        base_images = []
        for img_path in all_images:
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size == (IMAGE_WIDTH, IMAGE_HEIGHT):
                    base_images.append(np.array(img))
            except:
                pass
        print(f"  Loaded: {len(base_images)} images")

        cam_results = {}

        for level_id, (level_name, brightness_ratio) in LEVELS.items():
            random.seed(42 + level_id)  # reproducible but different per level

            print(f"  Level {level_id}: {level_name}", end="")
            if brightness_ratio is not None:
                print(f" ({brightness_ratio*100:.0f}% of local avg)")
            else:
                print(" (baseline)")

            mean_scores = []
            max_scores = []
            t0 = time.time()

            for img_array in base_images:
                if brightness_ratio is not None:
                    modified = draw_synthetic_line(img_array.copy(), brightness_ratio)
                    img_pil = Image.fromarray(modified)
                else:
                    img_pil = Image.fromarray(img_array)

                raw_scores = score_image(img_pil, extractor, memory_bank, all_positions)
                mean_scores.append(float(np.mean(raw_scores)))
                max_scores.append(float(np.max(raw_scores)))

            elapsed = time.time() - t0
            n = len(mean_scores)
            mean_arr = np.array(mean_scores)
            max_arr = np.array(max_scores)

            print(f"    N={n}, {elapsed:.0f}s")
            print(f"    mean: med={np.median(mean_arr):.4f}, p95={np.percentile(mean_arr,95):.4f}")
            print(f"    max:  med={np.median(max_arr):.4f}, p95={np.percentile(max_arr,95):.4f}")

            cam_results[level_name] = {
                "level": level_id,
                "brightness_ratio": brightness_ratio,
                "n_images": n,
                "mean_score": {
                    "min": round(float(mean_arr.min()), 4),
                    "median": round(float(np.median(mean_arr)), 4),
                    "p95": round(float(np.percentile(mean_arr, 95)), 4),
                    "max": round(float(mean_arr.max()), 4),
                    "values": [round(float(x), 4) for x in mean_arr],
                },
                "max_score": {
                    "min": round(float(max_arr.min()), 4),
                    "median": round(float(np.median(max_arr)), 4),
                    "p95": round(float(np.percentile(max_arr, 95)), 4),
                    "max": round(float(max_arr.max()), 4),
                    "values": [round(float(x), 4) for x in max_arr],
                },
            }

        # Compute AUROC
        if "normal" in cam_results:
            from sklearn.metrics import roc_auc_score
            normal_means = cam_results["normal"]["mean_score"]["values"]
            normal_maxs = cam_results["normal"]["max_score"]["values"]
            print(f"\n  --- AUROC ---")
            print(f"  {'Level':>20s}  {'mean AUROC':>10}  {'max AUROC':>10}  {'Det@FP5%(mean)':>14}  {'Det@FP5%(max)':>13}")

            for level_name, res in cam_results.items():
                if level_name == "normal":
                    continue

                for score_key, normal_vals in [("mean_score", normal_means), ("max_score", normal_maxs)]:
                    defect_vals = res[score_key]["values"]
                    labels = np.array([0]*len(normal_vals) + [1]*len(defect_vals))
                    scores = np.array(normal_vals + defect_vals)
                    try:
                        auroc = roc_auc_score(labels, scores)
                    except:
                        auroc = 0.0
                    res[f"auroc_{score_key}"] = round(auroc, 4)

                    normal_arr = np.array(normal_vals)
                    defect_arr = np.array(defect_vals)
                    th5 = np.percentile(normal_arr, 95)
                    det5 = float(np.mean(defect_arr > th5) * 100)
                    res[f"det_fp5_{score_key}"] = round(det5, 1)

                print(f"  {level_name:>20s}  {res['auroc_mean_score']:>10.4f}  {res['auroc_max_score']:>10.4f}  "
                      f"{res['det_fp5_mean_score']:>13.1f}%  {res['det_fp5_max_score']:>12.1f}%")

        # Plot histograms
        print(f"\n  Plotting histograms...")
        plot_histograms(cam_id, cam_results, OUTPUT_DIR)

        all_results[cam_id] = cam_results

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — mean_score AUROC by camera × defect level")
    print(f"{'='*60}")
    header = f"{'Cam':>4}"
    for ln in ["black", "dark_25pct", "medium_50pct", "light_75pct", "subtle_90pct", "near_invis_95pct"]:
        header += f" {ln[:10]:>10}"
    print(header)
    for cam_id in test_cameras:
        if cam_id not in all_results:
            continue
        line = f"{cam_id:>4}"
        for ln in ["black", "dark_25pct", "medium_50pct", "light_75pct", "subtle_90pct", "near_invis_95pct"]:
            if ln in all_results[cam_id] and f"auroc_mean_score" in all_results[cam_id][ln]:
                line += f" {all_results[cam_id][ln]['auroc_mean_score']:>10.4f}"
            else:
                line += f" {'N/A':>10}"
        print(line)

    out_path = OUTPUT_DIR / "synth_results.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Histograms: {OUTPUT_DIR}/cam*_hist.png")


if __name__ == "__main__":
    main()
