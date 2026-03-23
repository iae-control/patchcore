#!/usr/bin/env python3
"""Profile real images with various configs on A40."""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np, re
from torchvision.models import wide_resnet50_2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1200

class SpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        p = Path.home() / ".cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
        bb = wide_resnet50_2(weights=None)
        bb.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        self.layer1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.pool = nn.AvgPool2d(3, 3)
    def forward(self, x):
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3u = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        return self.pool(torch.cat([f2, f3u], dim=1))

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def natural_sort_key(path):
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

class SimpleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                return None
            return TRANSFORM(img)
        except:
            return None

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack(batch)

# Setup
print("Loading...", flush=True)
model = SpatialBackbone().cuda(0).eval().half()

bank = torch.from_numpy(
    np.load("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1/memory_bank.npy")
).cuda(0).half()
bank_sq = (bank.float() ** 2).sum(1).half()

pos_mean = torch.from_numpy(
    np.load("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1/pos_mean.npy").flatten()
).cuda(0).half()
pos_std = torch.from_numpy(
    np.load("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1/pos_std.npy").flatten()
).cuda(0).half()

# Spatial stride=2 indices
h_idx = torch.arange(0, 50, 2)
w_idx = torch.arange(0, 80, 2)
grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
flat_idx = (grid_h * 80 + grid_w).flatten()
sub_mean = pos_mean[flat_idx]
sub_std = pos_std[flat_idx]

# Get image paths
cam_dir = Path("/home/dk-sdd/nas_storage/20250630/20250630160852_596x199/camera_1")
all_imgs = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')], key=natural_sort_key)
all_imgs = all_imgs[100:len(all_imgs)-100]  # trim
print(f"Images: {len(all_imgs)}", flush=True)

# Warmup
for _ in range(5):
    with torch.no_grad():
        model(torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16))
torch.cuda.synchronize()

# === TEST: End-to-end with real images, various batch sizes ===
for bs in [8, 16]:
    for stride in [1, 2]:
        idx = flat_idx if stride == 2 else torch.arange(4000)
        s_mean = sub_mean if stride == 2 else pos_mean
        s_std = sub_std if stride == 2 else pos_std
        n_feat = len(idx)

        ds = SimpleDataset(all_imgs)
        loader = DataLoader(ds, batch_size=bs, num_workers=8, prefetch_factor=4,
                            pin_memory=True, collate_fn=collate, persistent_workers=True)

        # warmup
        for i, batch in enumerate(loader):
            if batch is None: continue
            x = batch.to("cuda:0", dtype=torch.float16, non_blocking=True)
            with torch.no_grad():
                feat = model(x)
            B = feat.shape[0]
            feats = feat.permute(0,2,3,1).reshape(B, 4000, 1536)[:, idx, :]
            for j in range(B):
                f = feats[j]
                f_sq = (f.float()**2).sum(1, keepdim=True).half()
                inner = f @ bank.T
                dists_sq = f_sq + bank_sq.unsqueeze(0) - 2*inner
                min_d = dists_sq.min(dim=1).values.float().clamp(min=0).sqrt()
                z = (min_d - s_mean.float()) / s_std.float()
            torch.cuda.synchronize()
            if i >= 3: break

        # Timed run
        torch.cuda.synchronize()
        t0 = time.time()
        count = 0
        for batch in loader:
            if batch is None: continue
            x = batch.to("cuda:0", dtype=torch.float16, non_blocking=True)
            with torch.no_grad():
                feat = model(x)
            B = feat.shape[0]
            feats = feat.permute(0,2,3,1).reshape(B, 4000, 1536)[:, idx, :]
            for j in range(B):
                f = feats[j]
                f_sq = (f.float()**2).sum(1, keepdim=True).half()
                inner = f @ bank.T
                dists_sq = f_sq + bank_sq.unsqueeze(0) - 2*inner
                min_d = dists_sq.min(dim=1).values.float().clamp(min=0).sqrt()
                z = (min_d - s_mean.float()) / s_std.float()
                max_z = z.max().item()
            torch.cuda.synchronize()
            count += B
        elapsed = time.time() - t0
        per_img = elapsed/count*1000
        print(f"  bs={bs} stride={stride} ({n_feat}feat): {per_img:.1f}ms/img, "
              f"{count/elapsed:.0f}img/s, 8000est={8000/count*elapsed:.0f}s", flush=True)
        del loader

print("\nDONE")
