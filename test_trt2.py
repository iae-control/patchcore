#!/usr/bin/env python3
"""Test ORT TensorRT EP with all NVIDIA libs in path."""
import os, time, numpy as np

# Set LD_LIBRARY_PATH for all NVIDIA libs
nvidia_base = "/home/dk-sdd/patchcore/venv/lib/python3.10/site-packages/nvidia"
trt_libs = "/home/dk-sdd/patchcore/venv/lib/python3.10/site-packages/tensorrt_libs"
lib_dirs = [trt_libs]
for subdir in ["cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "nvjitlink",
               "cusparse", "cusolver", "cufft", "curand", "nccl", "nvtx"]:
    p = os.path.join(nvidia_base, subdir, "lib")
    if os.path.isdir(p):
        lib_dirs.append(p)

ld_path = ":".join(lib_dirs)
os.environ["LD_LIBRARY_PATH"] = ld_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Must reload shared libs
import ctypes
for d in lib_dirs:
    for f in sorted(os.listdir(d)):
        if f.endswith(".so") or ".so." in f:
            try:
                ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)
            except:
                pass

import onnxruntime as ort

onnx_path = "/home/dk-sdd/patchcore/output_v5b_full/596x199/backbone.onnx"
os.makedirs("/home/dk-sdd/patchcore/trt_cache", exist_ok=True)

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Creating TRT session...", flush=True)
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

# Check which EP is actually being used
print(f"Active providers: {sess.get_providers()}", flush=True)

inp = sess.get_inputs()[0].name
dummy = np.random.randn(8, 3, 1200, 1920).astype(np.float32)

print("Warming up (TRT engine build on first run)...", flush=True)
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
print(f"\nResult: {per_img:.2f}ms/img")
print(f"  8000 img (1 GPU): {8000 * per_img / 1000:.1f}s")
print(f"  8000 img (2 GPU): {8000 * per_img / 2000:.1f}s")
print("DONE")
