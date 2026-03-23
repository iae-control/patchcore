"""0630 비결함 폴더 29개에서 폴더별 샘플 이미지 + 스코어 시트 생성."""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
from collections import defaultdict

OUTPUT_DIR = Path("/home/dk-sdd/patchcore/output_v6/596x199/group_1")
SAMPLE_DIR = OUTPUT_DIR / "samples_0630"
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
DEFECT_FOLDER_PREFIX = "160852"
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
    tiles = []
    valid_idx = []
    for i, (x, y) in enumerate(positions):
        tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        arr = np.array(tile)
        if arr.mean() < BRIGHTNESS_THRESHOLD:
            continue
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
    if len(tile_features) == 0:
        return np.array([])
    tf = torch.from_numpy(tile_features).cuda()
    mb = torch.from_numpy(memory_bank).cuda()
    scores = []
    bs = 256
    for i in range(0, len(tf), bs):
        dists = torch.cdist(tf[i:i+bs], mb)
        min_dists, _ = dists.min(dim=1)
        scores.append(min_dists.cpu().numpy())
    return np.concatenate(scores)

def make_tile_overlay(img, scores, valid_idx, n_tiles_x, n_tiles_y, threshold):
    """Return image with tile heatmap overlay."""
    score_grid = np.full((n_tiles_y, n_tiles_x), np.nan)
    for k, idx in enumerate(valid_idx):
        row = idx // n_tiles_x
        col = idx % n_tiles_x
        score_grid[row, col] = scores[k]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.imshow(img)
    vmax = max(threshold * 3, np.nanmax(scores)) if len(scores) > 0 else 1
    hm = ax.imshow(score_grid, cmap='jet', alpha=0.5,
                    extent=[0, IMAGE_WIDTH, IMAGE_HEIGHT, 0],
                    interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(hm, ax=ax, fraction=0.03, pad=0.02)
    ax.axis("off")
    max_score = float(np.nanmax(scores)) if len(scores) > 0 else 0
    status = "ANOMALY" if max_score > threshold else "NORMAL"
    ax.set_title(f"max={max_score:.3f} thr={threshold:.3f} {status}",
                 fontsize=10, color='red' if max_score > threshold else 'green')
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close()
    return Image.fromarray(buf)

def main():
    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    n_tiles_x = len(range(0, IMAGE_WIDTH - TILE_SIZE + 1, TILE_STRIDE))
    n_tiles_y = len(range(0, IMAGE_HEIGHT - TILE_SIZE + 1, TILE_STRIDE))

    print("Loading backbone...")
    backbone, features_dict = load_backbone()
    memory_bank = np.load(OUTPUT_DIR / "memory_bank.npy")
    with open(OUTPUT_DIR / "training_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold_mad"]
    print(f"Threshold: {threshold}")

    # Discover 0630 folders
    all_0630 = []
    defect_folder = None
    date_dir = NAS_ROOT / "20250630"
    for sub in sorted(date_dir.iterdir()):
        if sub.is_dir() and TARGET_SPEC in sub.name:
            if (sub / "camera_1").is_dir():
                if DEFECT_FOLDER_PREFIX in sub.name:
                    defect_folder = sub
                else:
                    all_0630.append(sub)

    # Include defect folder too
    if defect_folder:
        all_0630.append(defect_folder)

    print(f"Total 0630 folders: {len(all_0630)} (including defect)")

    SAMPLE_DIR.mkdir(exist_ok=True)
    SAMPLES_PER_FOLDER = 5  # 폴더당 5장 샘플

    for folder in all_0630:
        is_defect = DEFECT_FOLDER_PREFIX in folder.name
        tag = "DEFECT_" if is_defect else ""
        print(f"\n{'='*40}")
        print(f"Folder: {tag}{folder.name}")

        # Collect images from both cams
        all_images = []
        for cam_id in CAM_IDS:
            cam_dir = folder / f"camera_{cam_id}"
            if not cam_dir.exists():
                continue
            images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if len(images) > 200:
                images = images[100:-100]
            else:
                continue
            for p in images:
                all_images.append((p, cam_id))

        if not all_images:
            print("  No images after trim, skipping")
            continue

        # Sample evenly
        step = max(1, len(all_images) // SAMPLES_PER_FOLDER)
        sampled = all_images[::step][:SAMPLES_PER_FOLDER]

        # Generate contact sheet: 5 columns, 2 rows (original + heatmap)
        panels = []
        for img_path, cam_id in sampled:
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

                overlay = make_tile_overlay(img, scores, valid_idx, n_tiles_x, n_tiles_y, threshold)
                panels.append((img, overlay, img_path.name, cam_id, max_score))
            except Exception as e:
                print(f"  Error: {e}")
                continue

        if not panels:
            continue

        # Create contact sheet
        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(5*n, 7))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i, (orig, overlay, fname, cam_id, max_score) in enumerate(panels):
            axes[0, i].imshow(orig)
            axes[0, i].set_title(f"cam{cam_id} {fname}\nmax={max_score:.3f}", fontsize=7)
            axes[0, i].axis("off")
            axes[1, i].imshow(overlay)
            axes[1, i].axis("off")

        status = "DEFECT" if is_defect else "NORMAL?"
        fig.suptitle(f"{folder.name} [{status}]", fontsize=12, fontweight='bold',
                     color='red' if is_defect else 'black')
        plt.tight_layout()
        out_name = f"{tag}{folder.name}.png"
        plt.savefig(SAMPLE_DIR / out_name, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_name} ({n} samples)")

    print(f"\nAll sheets saved to: {SAMPLE_DIR}")

if __name__ == "__main__":
    main()
