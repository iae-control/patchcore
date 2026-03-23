#!/usr/bin/env python3
"""Final benchmark: dual GPU independent workers via threading."""
import os, sys, time, json, re, threading, numpy as np
from pathlib import Path

# Setup NVIDIA libs for TensorRT
venv_sp = Path("/home/dk-sdd/patchcore/venv/lib/python3.10/site-packages")
nvidia_base = venv_sp / "nvidia"
trt_libs = venv_sp / "tensorrt_libs"
lib_dirs = [str(trt_libs)]
for sd in ["cudnn","cublas","cuda_runtime","cuda_nvrtc","nvjitlink",
           "cusparse","cusolver","cufft","curand","nccl","nvtx"]:
    p = nvidia_base / sd / "lib"
    if p.is_dir(): lib_dirs.append(str(p))
os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH","")
import ctypes
for d in lib_dirs:
    dp = Path(d)
    if not dp.is_dir(): continue
    for f in sorted(dp.iterdir()):
        if f.suffix == ".so" or ".so." in f.name:
            try: ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
            except: pass

import torch, onnxruntime as ort
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1200
TRIM_HEAD, TRIM_TAIL = 100, 100
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v5b_full/596x199")

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def natural_sort_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', path.stem)]

class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT): return None
            return TRANSFORM(img)
        except: return None

def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack(batch) if batch else None

def get_images(folder, cam_id):
    cam_dir = Path(folder) / f"camera_{cam_id}"
    if not cam_dir.is_dir(): return []
    imgs = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')],
                  key=natural_sort_key)
    if len(imgs) <= TRIM_HEAD + TRIM_TAIL: return []
    return imgs[TRIM_HEAD:len(imgs)-TRIM_TAIL]

# Stride=2 indices
h_idx = torch.arange(0, 50, 2)
w_idx = torch.arange(0, 80, 2)
grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
FLAT_IDX = (grid_h * 80 + grid_w).flatten().numpy()

def worker_fn(gpu_id, image_paths, result_holder):
    """Independent GPU worker — runs in its own thread."""
    device = torch.device(f"cuda:{gpu_id}")

    # ORT TRT session
    onnx_path = str(MODEL_DIR / "backbone.onnx")
    cache_dir = str(MODEL_DIR / f"trt_cache_gpu{gpu_id}")
    os.makedirs(cache_dir, exist_ok=True)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, opts, providers=[
        ("TensorrtExecutionProvider", {
            "device_id": gpu_id,
            "trt_fp16_enable": True,
            "trt_max_workspace_size": str(4 * 1024**3),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
        }),
        ("CUDAExecutionProvider", {"device_id": gpu_id}),
    ])
    inp_name = sess.get_inputs()[0].name

    # Load bank
    bank = np.load(MODEL_DIR / "group_1/memory_bank.npy")
    bank_t = torch.from_numpy(bank).to(device).half()
    bank_sq = (bank_t.float()**2).sum(1).half()
    pos_mean = torch.from_numpy(
        np.load(MODEL_DIR / "group_1/pos_mean.npy").flatten()
    ).to(device).half()[FLAT_IDX]
    pos_std = torch.from_numpy(
        np.load(MODEL_DIR / "group_1/pos_std.npy").flatten()
    ).to(device).half()[FLAT_IDX]

    # Warmup
    dummy = np.random.randn(8, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
    sess.run(None, {inp_name: dummy})

    # DataLoader
    ds = ImgDataset(image_paths)
    loader = DataLoader(ds, batch_size=8, num_workers=4, prefetch_factor=4,
                        pin_memory=True, collate_fn=collate, persistent_workers=True)

    # Warmup with real images
    for i, batch in enumerate(loader):
        if batch is None: continue
        sess.run(None, {inp_name: batch.numpy()})
        if i >= 2: break

    # Timed run
    count = 0
    z_scores = []
    t0 = time.time()

    loader2 = DataLoader(ds, batch_size=8, num_workers=4, prefetch_factor=4,
                         pin_memory=True, collate_fn=collate, persistent_workers=True)

    for batch in loader2:
        if batch is None: continue
        batch_np = batch.numpy()
        B = batch_np.shape[0]

        # ORT backbone
        feat_map = sess.run(None, {inp_name: batch_np})[0]  # (B,1536,50,80)
        features = feat_map.transpose(0,2,3,1).reshape(B, 4000, 1536)[:, FLAT_IDX, :]

        # FP16 scoring on this GPU
        feat_t = torch.from_numpy(features).to(device).half()
        for i in range(B):
            f = feat_t[i]
            f_sq = (f.float()**2).sum(1, keepdim=True).half()
            inner = f @ bank_t.T
            dists_sq = f_sq + bank_sq.unsqueeze(0) - 2 * inner
            min_d = dists_sq.min(dim=1).values.float().clamp(min=0).sqrt()
            z = (min_d - pos_mean.float()) / pos_std.float()
            z_scores.append(z.max().item())
        count += B

    elapsed = time.time() - t0
    result_holder[gpu_id] = {
        "count": count, "elapsed": elapsed, "z_scores": z_scores
    }
    print(f"  GPU {gpu_id}: {count} imgs, {elapsed:.2f}s, "
          f"{count/elapsed:.1f}img/s, {elapsed/count*1000:.1f}ms/img", flush=True)


# === MAIN ===
print("Getting images...", flush=True)
all_imgs = get_images("/home/dk-sdd/nas_storage/20250630/20250630160852_596x199", 1)
print(f"Total: {len(all_imgs)} images", flush=True)

n_gpus = torch.cuda.device_count()
print(f"GPUs: {n_gpus}", flush=True)

# === Single GPU baseline ===
print("\n=== SINGLE GPU ===", flush=True)
result = {}
worker_fn(0, all_imgs, result)
r = result[0]
print(f"  -> {r['elapsed']/r['count']*1000:.1f}ms/img, 8000 est: {8000*r['elapsed']/r['count']:.0f}s")

if n_gpus >= 2:
    # === Dual GPU parallel ===
    print("\n=== DUAL GPU (parallel threads) ===", flush=True)
    mid = len(all_imgs) // 2
    imgs_0 = all_imgs[:mid]
    imgs_1 = all_imgs[mid:]

    results = {}
    t0 = time.time()
    t1 = threading.Thread(target=worker_fn, args=(0, imgs_0, results))
    t2 = threading.Thread(target=worker_fn, args=(1, imgs_1, results))
    t1.start(); t2.start()
    t1.join(); t2.join()
    total_elapsed = time.time() - t0

    total_imgs = sum(r["count"] for r in results.values())
    print(f"\n  Total: {total_imgs} imgs, {total_elapsed:.2f}s wall time")
    print(f"  -> {total_elapsed/total_imgs*1000:.1f}ms/img, "
          f"8000 est: {8000*total_elapsed/total_imgs:.0f}s")

print("\nDONE")
