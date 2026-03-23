"""v6 검증: 0630 비결함 폴더 29개 추론하여 날짜 차이 vs 진짜 결함 감지인지 확인."""
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

def process_folders(folders, date_str, backbone, features_dict, positions, memory_bank, max_per_folder=50):
    """Process folders, return per-image max scores with folder info."""
    image_scores = []
    for folder in folders:
        folder_count = 0
        for cam_id in CAM_IDS:
            cam_dir = folder / f"camera_{cam_id}"
            if not cam_dir.exists():
                continue
            images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if len(images) > 200:
                images = images[100:-100]
            else:
                continue
            # Sample evenly from folder (max_per_folder per cam)
            if len(images) > max_per_folder:
                step = len(images) // max_per_folder
                images = images[::step][:max_per_folder]
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
                    })
                    folder_count += 1
                except Exception as e:
                    print(f"  Error: {img_path}: {e}")
                    continue
        print(f"  {folder.name}: {folder_count} images scored")
    return image_scores

def main():
    t0 = time.time()
    positions = tile_positions(IMAGE_WIDTH, IMAGE_HEIGHT, TILE_SIZE, TILE_STRIDE)
    print(f"Tiles per image: {len(positions)}")

    print("Loading backbone...")
    backbone, features_dict = load_backbone()

    memory_bank = np.load(OUTPUT_DIR / "memory_bank.npy")
    print(f"Memory bank: {memory_bank.shape}")

    with open(OUTPUT_DIR / "training_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold_mad"]
    print(f"Threshold: {threshold}")

    # Discover folders
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

    # 0630 non-defect folders (29개)
    non_defect_0630 = [f for f in all_dates.get(DEFECT_DATE, []) if DEFECT_FOLDER_PREFIX not in f.name]
    print(f"\n0630 non-defect folders: {len(non_defect_0630)}")

    # Score them (sample 20 images per folder per cam to save time)
    print("Scoring 0630 non-defect folders...")
    non_defect_scores = process_folders(non_defect_0630, DEFECT_DATE, backbone, features_dict, positions, memory_bank, max_per_folder=20)

    # Also score defect folder for comparison
    print("\nScoring defect folder...")
    defect_scores = process_folders([defect_folder] if defect_folder else [], DEFECT_DATE, backbone, features_dict, positions, memory_bank, max_per_folder=50)

    # Stats
    nd_max = [s["max_score"] for s in non_defect_scores]
    df_max = [s["max_score"] for s in defect_scores]

    print(f"\n{'='*60}")
    print(f"=== VERIFICATION RESULTS ===")
    print(f"{'='*60}")
    print(f"Defect folder ({DEFECT_FOLDER_PREFIX}):")
    print(f"  n={len(df_max)}, min={np.min(df_max):.4f}, max={np.max(df_max):.4f}, median={np.median(df_max):.4f}")
    print(f"  Anomaly rate: {sum(1 for s in df_max if s > threshold)}/{len(df_max)} ({100*sum(1 for s in df_max if s > threshold)/max(len(df_max),1):.1f}%)")

    print(f"\n0630 non-defect (29 folders):")
    print(f"  n={len(nd_max)}, min={np.min(nd_max):.4f}, max={np.max(nd_max):.4f}, median={np.median(nd_max):.4f}")
    print(f"  Anomaly rate: {sum(1 for s in nd_max if s > threshold)}/{len(nd_max)} ({100*sum(1 for s in nd_max if s > threshold)/max(len(nd_max),1):.1f}%)")

    print(f"\nThreshold: {threshold:.4f}")
    print(f"\n*** If non-defect 0630 anomaly rate is HIGH → model detects DATE difference, not defects ***")
    print(f"*** If non-defect 0630 anomaly rate is LOW → model genuinely detects defects ***")

    # Histogram: 3 groups
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(df_max, bins=40, alpha=0.6, label=f'Defect folder (160852) n={len(df_max)}', color='red', edgecolor='darkred')
    ax.hist(nd_max, bins=40, alpha=0.6, label=f'0630 non-defect (29 folders) n={len(nd_max)}', color='orange', edgecolor='darkorange')
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Max Tile Score (L2 distance)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('v6 Verification: Defect vs 0630 Non-Defect vs Threshold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "verify_0630.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved: {out_path}")

    # Per-folder breakdown
    print(f"\n=== Per-folder anomaly rates ===")
    folder_stats = defaultdict(list)
    for s in non_defect_scores:
        folder_stats[s["folder"]].append(s["max_score"])
    for fname, scores in sorted(folder_stats.items()):
        anom = sum(1 for s in scores if s > threshold)
        print(f"  {fname}: {anom}/{len(scores)} ({100*anom/max(len(scores),1):.1f}%) median={np.median(scores):.4f}")

    # Save
    results = {
        "defect_scores": defect_scores,
        "non_defect_0630_scores": non_defect_scores,
        "threshold": threshold,
    }
    with open(OUTPUT_DIR / "verify_results.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
