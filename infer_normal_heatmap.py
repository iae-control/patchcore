#!/usr/bin/env python3
"""
Inference on NORMAL date (20250831) with HEATMAP generation
Compares FULL-res vs HALF-res models (group 1 only)
"""
import sys, json, time, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TEST_DATE = "20250831"
SPEC = "596x199"
GROUP_ID = 1
CAM_IDS = [1, 10]
MIRROR_CAMS = {10}
TRIM_HEAD = 100
TRIM_TAIL = 100
MAX_FOLDERS = 3  # 3 folders to keep it manageable
MAX_IMAGES_PER_FOLDER = 30  # sample images per folder

IMAGE_WIDTH_ORIG = 1920
IMAGE_HEIGHT_ORIG = 1200
GAUSSIAN_SIGMA = 4
SCORE_PERCENTILE_CAP = 99.5
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3

MODELS = {
    "full": {
        "dir": Path("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1"),
        "width": 1920, "height": 1200,
        "label": "FULL (1920x1200)"
    },
    "half": {
        "dir": Path("/home/dk-sdd/patchcore/output_v5b_half/596x199/group_1"),
        "width": 960, "height": 600,
        "label": "HALF (960x600)"
    }
}

OUTPUT_BASE = Path("/home/dk-sdd/patchcore/normal_test_heatmaps")


# ===== BACKBONE =====
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        local_weights = cache_dir / "wide_resnet50_2-95faca4d.pth"
        backbone = wide_resnet50_2(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict)

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.spatial_pool = nn.AvgPool2d(kernel_size=SPATIAL_POOL_K, stride=SPATIAL_POOL_S)
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
            features = self.spatial_pool(features)
        return features.float()

    def extract_spatial(self, x):
        feat_map = self.forward(x)
        B, C, H, W = feat_map.shape
        features = feat_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return features.cpu().numpy(), (H, W)


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


def generate_heatmap(img_path, memory_bank, extractor, spatial_size, output_path,
                     pos_mean, pos_std, threshold_z, model_label, transform, mirror=False):
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMAGE_WIDTH_ORIG, IMAGE_HEIGHT_ORIG):
        return None, None

    img_display = img.copy()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    tensor = transform(img).unsqueeze(0)
    features, _ = extractor.extract_spatial(tensor)
    scores = score_spatial(features, memory_bank)
    z_scores = (scores - pos_mean) / (pos_std + 1e-6)

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

    plt.suptitle(f"PatchCore {model_label} | {SPEC}/group_{GROUP_ID} | {img_path.name} | NORMAL DATE ({TEST_DATE})", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return max_z, float(scores.max())


def natural_sort_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]


def main():
    device = "cuda:0"

    # Load backbone once
    print("Loading backbone...")
    extractor = SpatialFeatureExtractor(device)

    # Find normal test folders
    date_dir = NAS_ROOT / TEST_DATE
    folders = sorted([d for d in date_dir.iterdir() if d.is_dir() and SPEC in d.name])[:MAX_FOLDERS]
    print(f"Testing {len(folders)} folders from {TEST_DATE}")

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_cfg['label']}")
        print(f"{'='*60}")

        model_dir = model_cfg["dir"]
        if not model_dir.exists():
            print(f"  SKIP: {model_dir} not found")
            continue

        memory_bank = np.load(model_dir / "memory_bank.npy")
        spatial_size = np.load(model_dir / "spatial_size.npy").tolist()
        pos_mean = np.load(model_dir / "pos_mean.npy")
        pos_std = np.load(model_dir / "pos_std.npy")
        with open(model_dir / "training_meta.json") as f:
            meta = json.load(f)
        threshold = meta["threshold_mad"]

        print(f"  Bank: {memory_bank.shape}, Spatial: {spatial_size}, Threshold: {threshold:.4f}")

        transform = transforms.Compose([
            transforms.Resize((model_cfg["height"], model_cfg["width"]),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        out_dir = OUTPUT_BASE / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        total_imgs = 0
        total_anom = 0
        all_z = []

        for folder in folders:
            folder_short = folder.name
            folder_out = out_dir / folder_short
            folder_out.mkdir(parents=True, exist_ok=True)

            img_count = 0
            for cam_id in CAM_IDS:
                cam_dir = folder / f"camera_{cam_id}"
                if not cam_dir.is_dir():
                    continue
                images = sorted(
                    [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                    key=natural_sort_key
                )
                if len(images) <= TRIM_HEAD + TRIM_TAIL:
                    continue
                images = images[TRIM_HEAD:len(images) - TRIM_TAIL]

                # Evenly sample
                step = max(1, len(images) // (MAX_IMAGES_PER_FOLDER // len(CAM_IDS)))
                images = images[::step]

                mirror = cam_id in MIRROR_CAMS
                for img_path in images:
                    out_path = folder_out / f"heatmap_cam{cam_id}_{img_path.stem}.png"
                    max_z, raw_max = generate_heatmap(
                        img_path, memory_bank, extractor, spatial_size, out_path,
                        pos_mean, pos_std, threshold, model_cfg["label"], transform, mirror
                    )
                    if max_z is not None:
                        is_anom = max_z > threshold
                        all_z.append(max_z)
                        total_imgs += 1
                        if is_anom:
                            total_anom += 1
                        img_count += 1
                        tag = "ANOM" if is_anom else "ok"
                        print(f"    cam{cam_id} {img_path.name}: z={max_z:.2f} [{tag}]")

            print(f"  {folder_short}: {img_count} heatmaps saved")

        print(f"\n  SUMMARY ({model_cfg['label']}):")
        print(f"    Total: {total_imgs}, Anomaly: {total_anom} ({100*total_anom/max(total_imgs,1):.1f}% FP)")
        if all_z:
            print(f"    Z-score: {min(all_z):.2f} ~ {max(all_z):.2f}, mean={np.mean(all_z):.2f}")

    print(f"\nHeatmaps saved to: {OUTPUT_BASE}")
    print("DONE")


if __name__ == "__main__":
    main()
