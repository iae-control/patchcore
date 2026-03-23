#!/usr/bin/env python3
"""
Inference on NORMAL date (20250831) using half-res model (group 1)
To verify false positive rate
"""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import wide_resnet50_2

# ===== CONFIG =====
NAS_ROOT = Path("/home/dk-sdd/nas_storage")
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v5b_half/596x199/group_1")
TEST_DATE = "20250831"
SPEC = "596x199"
CAM_IDS = [1, 10]  # group 1
MIRROR_CAMS = {10}
TRIM_HEAD = 100
TRIM_TAIL = 100
MAX_FOLDERS = 5  # test a few folders first

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 600

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.spatial_pool = nn.AvgPool2d(kernel_size=3, stride=3)
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

def score_image(feat_flat, memory_bank, pos_mean, pos_std, spatial_h, spatial_w, threshold):
    """Score a single image features against memory bank"""
    # FP16 matmul L2
    feat_fp16 = feat_flat.half()
    bank_fp16 = memory_bank.half()

    feat_norm = (feat_fp16 ** 2).sum(dim=1, keepdim=True)
    bank_norm = (bank_fp16 ** 2).sum(dim=1, keepdim=True).t()
    dist_sq = feat_norm + bank_norm - 2.0 * feat_fp16 @ bank_fp16.t()
    dist_sq = dist_sq.clamp(min=0)
    min_dist = dist_sq.min(dim=1).values.float().sqrt()

    # Z-score normalization (pos_mean/pos_std are 1D flattened)
    z_scores = (min_dist - pos_mean) / (pos_std + 1e-6)
    max_z = z_scores.max().item()

    return max_z, max_z > threshold

def main():
    device = "cuda:0"
    print(f"Loading model from {MODEL_DIR}")

    memory_bank = torch.from_numpy(np.load(MODEL_DIR / "memory_bank.npy")).to(device)
    spatial_size = np.load(MODEL_DIR / "spatial_size.npy")
    pos_mean = torch.from_numpy(np.load(MODEL_DIR / "pos_mean.npy")).to(device)
    pos_std = torch.from_numpy(np.load(MODEL_DIR / "pos_std.npy")).to(device)

    with open(MODEL_DIR / "training_meta.json") as f:
        meta = json.load(f)
    threshold = meta["threshold_mad"]
    spatial_h, spatial_w = int(spatial_size[0]), int(spatial_size[1])

    print(f"Bank: {memory_bank.shape}, Spatial: {spatial_h}x{spatial_w}, Threshold: {threshold:.4f}")

    # Load backbone
    extractor = SpatialFeatureExtractor(device)
    print("Backbone loaded")

    # Find normal folders
    date_dir = NAS_ROOT / TEST_DATE
    folders = sorted([d for d in date_dir.iterdir() if d.is_dir() and SPEC in d.name])[:MAX_FOLDERS]
    print(f"\nTesting {len(folders)} folders from {TEST_DATE}")

    results = []
    total_anom = 0
    total_imgs = 0

    for folder in folders:
        folder_results = []
        for cam_id in CAM_IDS:
            cam_dir = folder / f"camera_{cam_id}"
            if not cam_dir.is_dir():
                continue
            images = sorted(cam_dir.iterdir())
            images = [p for p in images if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            if len(images) <= TRIM_HEAD + TRIM_TAIL:
                continue
            images = images[TRIM_HEAD:len(images) - TRIM_TAIL]

            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                if cam_id in MIRROR_CAMS:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                tensor = TRANSFORM(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = extractor(tensor)
                feat_flat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[1])

                max_z, is_anom = score_image(feat_flat, memory_bank, pos_mean, pos_std, spatial_h, spatial_w, threshold)
                folder_results.append({"file": img_path.name, "cam": cam_id, "z": max_z, "anom": is_anom})
                total_imgs += 1
                if is_anom:
                    total_anom += 1

        n_anom = sum(1 for r in folder_results if r["anom"])
        n_total = len(folder_results)
        print(f"  {folder.name}: {n_total} imgs, {n_anom} anomaly ({100*n_anom/max(n_total,1):.1f}%)")

        # Show top anomalies
        top = sorted(folder_results, key=lambda x: x["z"], reverse=True)[:3]
        for r in top:
            tag = "ANOM" if r["anom"] else "ok"
            print(f"    cam{r['cam']} {r['file']}: z={r['z']:.2f} [{tag}]")

        results.extend(folder_results)

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_imgs} images, {total_anom} anomaly ({100*total_anom/max(total_imgs,1):.1f}% FALSE POSITIVE)")

    # Z-score distribution
    zs = [r["z"] for r in results]
    print(f"Z-score: min={min(zs):.2f}, max={max(zs):.2f}, mean={np.mean(zs):.2f}, median={np.median(zs):.2f}")

    # Save results
    out_path = MODEL_DIR / "normal_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"date": TEST_DATE, "total": total_imgs, "anomaly": total_anom,
                    "fp_rate": total_anom/max(total_imgs,1), "results": results}, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
