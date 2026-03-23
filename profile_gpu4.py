#!/usr/bin/env python3
"""Test CUDA Graphs + ORT TensorRT EP on A40."""
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

print("=== TEST 1: CUDA Graphs ===", flush=True)
model = SpatialBackbone().cuda().eval().half()
BS = 8

# Warmup
for _ in range(5):
    with torch.no_grad():
        model(torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16))
torch.cuda.synchronize()

# Baseline (no graphs)
t0 = time.time()
N = 30
for _ in range(N):
    x = torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = model(x)
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  Baseline (eager FP16): {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)

# CUDA Graph capture
print("  Capturing CUDA graph...", flush=True)
static_input = torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16)
static_output = None

# Warmup for graph capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.no_grad():
            static_output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    with torch.no_grad():
        static_output = model(static_input)
print("  Graph captured!", flush=True)

# Benchmark with graph replay
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    static_input.copy_(torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16))
    graph.replay()
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  CUDA Graph replay: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)

del model, graph, static_input, static_output
torch.cuda.empty_cache()

# === TEST 2: ONNX Runtime with various options ===
print("\n=== TEST 2: ONNX Runtime ===", flush=True)
import onnxruntime as ort

onnx_path = "/home/dk-sdd/patchcore/output_v5b_full/596x199/backbone.onnx"
if not Path(onnx_path).exists():
    print("  ONNX model not found, skipping")
else:
    # Check available EPs
    print(f"  Available EPs: {ort.get_available_providers()}", flush=True)

    # Default CUDA EP
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, opts,
        providers=[("CUDAExecutionProvider", {"device_id": 0})])
    inp_name = sess.get_inputs()[0].name

    dummy = np.random.randn(BS, 3, 1200, 1920).astype(np.float32)
    for _ in range(5):
        sess.run(None, {inp_name: dummy})

    t0 = time.time()
    for _ in range(N):
        sess.run(None, {inp_name: dummy})
    elapsed = time.time() - t0
    print(f"  ORT CUDA FP32: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)

    # ORT with FP16 input
    dummy16 = dummy.astype(np.float16)
    try:
        for _ in range(3):
            sess.run(None, {inp_name: dummy16})
        t0 = time.time()
        for _ in range(N):
            sess.run(None, {inp_name: dummy16})
        elapsed = time.time() - t0
        print(f"  ORT CUDA FP16 input: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)
    except Exception as e:
        print(f"  ORT FP16 input failed: {e}", flush=True)

    # Check if TensorRT EP available
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        print("  TensorRT EP available! Testing...", flush=True)
        trt_sess = ort.InferenceSession(onnx_path, opts,
            providers=[("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
            }), "CUDAExecutionProvider"])
        for _ in range(5):
            trt_sess.run(None, {inp_name: dummy})
        t0 = time.time()
        for _ in range(N):
            trt_sess.run(None, {inp_name: dummy})
        elapsed = time.time() - t0
        print(f"  ORT TensorRT FP16: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)
    else:
        print("  TensorRT EP not available", flush=True)

# === TEST 3: torch.jit.trace ===
print("\n=== TEST 3: torch.jit.trace ===", flush=True)
model2 = SpatialBackbone().cuda().eval().half()
dummy_t = torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16)
with torch.no_grad():
    traced = torch.jit.trace(model2, dummy_t)
    traced = torch.jit.freeze(traced)
    # warmup
    for _ in range(10):
        traced(dummy_t)
    torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    x = torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = traced(x)
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  jit.trace+freeze: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)

# jit.trace + CUDA Graph
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        with torch.no_grad():
            traced(static_input := torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16))
torch.cuda.current_stream().wait_stream(s)

graph2 = torch.cuda.CUDAGraph()
static_in2 = torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16)
with torch.cuda.graph(graph2):
    with torch.no_grad():
        static_out2 = traced(static_in2)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    static_in2.copy_(torch.randn(BS, 3, 1200, 1920, device="cuda", dtype=torch.float16))
    graph2.replay()
    torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  jit.trace+freeze+CUDAGraph: {elapsed/(N*BS)*1000:.2f}ms/img", flush=True)

print("\nDONE")
