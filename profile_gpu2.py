#!/usr/bin/env python3
"""Profile advanced optimizations on A40."""
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

print("Loading...")
model = SpatialBackbone().cuda().eval().half()

bank_np = np.load("/home/dk-sdd/patchcore/output_v5b_full/596x199/group_1/memory_bank.npy")
bank_f32 = torch.from_numpy(bank_np).cuda()
bank_f16 = bank_f32.half()
bank_sq_f32 = (bank_f32 ** 2).sum(1)
bank_sq_f16 = (bank_f16.float() ** 2).sum(1).half()

# Warmup
for _ in range(10):
    dummy = torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        model(dummy)
torch.cuda.synchronize()

# === TEST 1: torch.compile modes ===
print("\n=== BACKBONE: torch.compile modes ===")
for mode in ["default", "reduce-overhead", "max-autotune"]:
    try:
        compiled = torch.compile(model, mode=mode)
        # warmup compile
        for _ in range(5):
            with torch.no_grad():
                compiled(torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()

        t0 = time.time()
        N = 30
        for _ in range(N):
            with torch.no_grad():
                compiled(torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16))
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"  {mode}: {elapsed/(N*8)*1000:.2f}ms/img")
    except Exception as e:
        print(f"  {mode}: FAILED ({e})")

# === TEST 2: FP16 scoring ===
print("\n=== SCORING: FP16 vs FP32 ===")
feat = torch.randn(4000, 1536, device="cuda")

# FP32
torch.cuda.synchronize()
t0 = time.time()
for _ in range(300):
    f_sq = (feat ** 2).sum(1, keepdim=True)
    inner = feat @ bank_f32.T
    dists_sq = f_sq + bank_sq_f32.unsqueeze(0) - 2 * inner
    min_d = dists_sq.min(dim=1).values
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  FP32: {elapsed/300*1000:.2f}ms/img")

# FP16
feat16 = feat.half()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(300):
    f_sq = (feat16.float() ** 2).sum(1, keepdim=True).half()
    inner = feat16 @ bank_f16.T  # FP16 tensor core matmul
    dists_sq = f_sq + bank_sq_f16.unsqueeze(0) - 2 * inner
    min_d = dists_sq.min(dim=1).values
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  FP16 (mixed): {elapsed/300*1000:.2f}ms/img")

# Pure FP16
torch.cuda.synchronize()
t0 = time.time()
for _ in range(300):
    inner = feat16 @ bank_f16.T
    # Skip explicit L2, just use negative inner product as proxy
    max_sim, _ = inner.max(dim=1)
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  FP16 (inner product only): {elapsed/300*1000:.2f}ms/img")

# === TEST 3: Subsampled spatial (every 2nd position) ===
print("\n=== SCORING: Spatial subsampling ===")
for stride in [1, 2, 4]:
    n_feat = 4000 // (stride * stride) if stride > 1 else 4000
    f_sub = torch.randn(n_feat, 1536, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(300):
        f_sq = (f_sub ** 2).sum(1, keepdim=True)
        inner = f_sub @ bank_f32.T
        dists_sq = f_sq + bank_sq_f32.unsqueeze(0) - 2 * inner
        min_d = dists_sq.min(dim=1).values
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  stride={stride} ({n_feat} features): {elapsed/300*1000:.2f}ms/img")

# === TEST 4: Half-resolution backbone ===
print("\n=== BACKBONE: Half resolution (960x600) ===")
for _ in range(5):
    with torch.no_grad():
        model(torch.randn(8, 3, 600, 960, device="cuda", dtype=torch.float16))
torch.cuda.synchronize()
t0 = time.time()
N = 50
for _ in range(N):
    with torch.no_grad():
        out = model(torch.randn(8, 3, 600, 960, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  960x600 batch=8: {elapsed/(N*8)*1000:.2f}ms/img, output={out.shape}")

# Full res for comparison
t0 = time.time()
for _ in range(N):
    with torch.no_grad():
        out = model(torch.randn(8, 3, 1200, 1920, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  1920x1200 batch=8: {elapsed/(N*8)*1000:.2f}ms/img, output={out.shape}")

print("\nDONE")
