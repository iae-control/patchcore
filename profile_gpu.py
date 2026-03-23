#!/usr/bin/env python3
"""Profile backbone vs scoring on A40."""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np
from torchvision.models import wide_resnet50_2
from pathlib import Path

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

print("Loading model...")
model = SpatialBackbone().cuda().eval().half()

bank = torch.from_numpy(
    np.load("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1/memory_bank.npy")
).cuda().float()
bank_sq = (bank ** 2).sum(1)

# Warmup
for _ in range(10):
    dummy = torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = model(dummy)
torch.cuda.synchronize()

# Backbone benchmark
print("\n=== BACKBONE ===")
for bs in [4, 8, 12, 16]:
    torch.cuda.synchronize()
    t0 = time.time()
    N = 50
    for _ in range(N):
        x = torch.randn(bs, 3, 1200, 1920, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            feat = model(x)
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_img = elapsed / (N * bs) * 1000
    print(f"  batch={bs}: {per_img:.2f}ms/img  ({N*bs} imgs in {elapsed:.1f}s)")

# Scoring benchmark
print("\n=== SCORING ===")
feat = torch.randn(4000, 1536, device="cuda", dtype=torch.float32)

# matmul L2
torch.cuda.synchronize()
t0 = time.time()
N = 500
for _ in range(N):
    f_sq = (feat ** 2).sum(1, keepdim=True)
    inner = feat @ bank.T
    dists_sq = f_sq + bank_sq.unsqueeze(0) - 2 * inner
    min_d = dists_sq.min(dim=1).values
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  matmul L2 (1 img, 4000x12000): {elapsed/N*1000:.2f}ms")

# cdist
torch.cuda.synchronize()
t0 = time.time()
N = 500
for _ in range(N):
    dists = torch.cdist(feat.unsqueeze(0), bank.unsqueeze(0)).squeeze(0)
    min_d = dists.min(dim=1).values
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  cdist (1 img, 4000x12000): {elapsed/N*1000:.2f}ms")

# Batched scoring (8 images)
feats = torch.randn(8, 4000, 1536, device="cuda", dtype=torch.float32)
torch.cuda.synchronize()
t0 = time.time()
N = 100
for _ in range(N):
    for i in range(8):
        f = feats[i]
        f_sq = (f ** 2).sum(1, keepdim=True)
        inner = f @ bank.T
        dists_sq = f_sq + bank_sq.unsqueeze(0) - 2 * inner
        min_d = dists_sq.min(dim=1).values
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  matmul L2 loop (8 imgs): {elapsed/N*1000:.2f}ms/batch, {elapsed/(N*8)*1000:.2f}ms/img")

print("\nDONE")
