#!/usr/bin/env python3
"""Test ORT TensorRT EP."""
import os, time, numpy as np, onnxruntime as ort

onnx_path = "/home/dk-sdd/patchcore/output_v5b_full/596x199/backbone.onnx"
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

os.makedirs("/home/dk-sdd/patchcore/trt_cache", exist_ok=True)

print("Creating TRT session (engine build may take minutes)...", flush=True)
sess = ort.InferenceSession(onnx_path, opts, providers=[
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_max_workspace_size": str(4 * 1024**3),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/home/dk-sdd/patchcore/trt_cache",
    }),
    "CUDAExecutionProvider",
])

inp = sess.get_inputs()[0].name
dummy = np.random.randn(8, 3, 1200, 1920).astype(np.float32)

print("Warming up...", flush=True)
for i in range(5):
    t0 = time.time()
    sess.run(None, {inp: dummy})
    print(f"  warmup {i}: {time.time()-t0:.1f}s", flush=True)

print("\nBenchmarking...", flush=True)
t0 = time.time()
N = 30
for _ in range(N):
    sess.run(None, {inp: dummy})
elapsed = time.time() - t0
per_img = elapsed / (N * 8) * 1000
print(f"\nORT TensorRT FP16: {per_img:.2f}ms/img")
print(f"8000 img (1 GPU): {8000 * per_img / 1000:.1f}s")
print(f"8000 img (2 GPU): {8000 * per_img / 2000:.1f}s")

# Also test CUDA EP baseline for comparison
print("\n--- CUDA EP baseline ---", flush=True)
sess2 = ort.InferenceSession(onnx_path, opts, providers=[
    ("CUDAExecutionProvider", {"device_id": 0}),
])
for _ in range(5):
    sess2.run(None, {sess2.get_inputs()[0].name: dummy})
t0 = time.time()
for _ in range(N):
    sess2.run(None, {sess2.get_inputs()[0].name: dummy})
elapsed2 = time.time() - t0
per_img2 = elapsed2 / (N * 8) * 1000
print(f"ORT CUDA FP32: {per_img2:.2f}ms/img")
print(f"Speedup: {per_img2/per_img:.1f}x")
print("DONE")
