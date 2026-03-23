#!/usr/bin/env python3
"""PatchCore v5b-FULL — Fast Inference Pipeline v4 (Dual A40 + TensorRT)

Optimizations:
  1. ONNX Runtime TensorRT EP (FP16 engine) for backbone
  2. FP16 matmul-based scoring (Tensor Cores)
  3. Spatial stride=2 scoring (4000 → 1000 features)
  4. Dual GPU: split batches across GPU 0 & 1
  5. Async DataLoader with prefetch

Target: 8000 images in ~40 seconds
"""

import os
import sys
import time
import json
import re
import gc
import argparse
import ctypes
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ===== NVIDIA LIBS FOR ORT TENSORRT EP =====
def setup_nvidia_libs():
    """Load NVIDIA shared libs so ORT TensorRT EP can find them."""
    venv_sp = Path("/home/dk-sdd/patchcore/venv/lib/python3.10/site-packages")
    nvidia_base = venv_sp / "nvidia"
    trt_libs = venv_sp / "tensorrt_libs"

    lib_dirs = [str(trt_libs)]
    for subdir in ["cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "nvjitlink",
                   "cusparse", "cusolver", "cufft", "curand", "nccl", "nvtx"]:
        p = nvidia_base / subdir / "lib"
        if p.is_dir():
            lib_dirs.append(str(p))

    ld_path = ":".join(lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = ld_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    for d in lib_dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        for f in sorted(dp.iterdir()):
            if f.suffix == ".so" or ".so." in f.name:
                try:
                    ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                except:
                    pass

setup_nvidia_libs()
import onnxruntime as ort

# ===== CONFIG =====
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
SPATIAL_POOL_K = 3
SPATIAL_POOL_S = 3
TRIM_HEAD = 100
TRIM_TAIL = 100
Z_SCORE_THRESHOLD = 3.0
SCORE_SPATIAL_STRIDE = 2

TARGET_SPEC = "596x199"
MODEL_DIR = Path("/home/dk-sdd/patchcore/output_v5b_full") / TARGET_SPEC

GROUP_CONFIG = {
    1: {"cams": [1, 10], "mirror_cam": 10, "infer_cam": 1},
    2: {"cams": [2, 9],  "mirror_cam": 9,  "infer_cam": 2},
    3: {"cams": [3, 8],  "mirror_cam": 8,  "infer_cam": 3},
    4: {"cams": [4, 7],  "mirror_cam": 7,  "infer_cam": 4},
    5: {"cams": [5, 6],  "mirror_cam": 6,  "infer_cam": 5},
}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===== ONNX EXPORT =====
def export_onnx_if_needed(onnx_path: Path, batch_size=8):
    if onnx_path.exists():
        return
    print(f"[ONNX] Exporting backbone...", flush=True)
    from torchvision.models import wide_resnet50_2

    class SpatialBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            cache = Path.home() / ".cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
            bb = wide_resnet50_2(weights=None)
            bb.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
            self.layer1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
            self.layer2 = bb.layer2
            self.layer3 = bb.layer3
            self.pool = nn.AvgPool2d(SPATIAL_POOL_K, SPATIAL_POOL_S)

        def forward(self, x):
            h = self.layer1(x)
            f2 = self.layer2(h)
            f3 = self.layer3(f2)
            f3u = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
            return self.pool(torch.cat([f2, f3u], dim=1))

    model = SpatialBackbone().cuda().eval()
    dummy = torch.randn(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device="cuda")
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"  Saved: {onnx_path}", flush=True)
    del model, dummy
    gc.collect()
    torch.cuda.empty_cache()


# ===== DATASET =====
def natural_sort_key(path):
    parts = re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class InferenceDataset(Dataset):
    def __init__(self, image_paths, mirror=False):
        self.image_paths = image_paths
        self.mirror = mirror

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                return None, str(img_path)
            if self.mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return TRANSFORM(img), str(img_path)
        except:
            return None, str(img_path)


def collate_skip_none(batch):
    tensors, paths = [], []
    for t, p in batch:
        if t is not None:
            tensors.append(t)
            paths.append(p)
    if not tensors:
        return None, []
    return torch.stack(tensors), paths


def get_image_paths(folder, cam_id, subsample_step=1):
    cam_dir = Path(folder) / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=natural_sort_key
    )
    if len(images) <= TRIM_HEAD + TRIM_TAIL:
        return []
    images = images[TRIM_HEAD:len(images) - TRIM_TAIL]
    if subsample_step > 1:
        images = images[::subsample_step]
    return images


# ===== GPU WORKER =====
class GpuWorker:
    """Single GPU: ORT TensorRT backbone + torch FP16 scoring."""

    def __init__(self, device_id: int, model_dir: Path, groups: list,
                 score_stride: int = SCORE_SPATIAL_STRIDE):
        self.device_id = device_id
        self.torch_device = torch.device(f"cuda:{device_id}")
        self.groups = groups
        self.score_stride = score_stride

        # ORT TensorRT session
        onnx_path = model_dir / "backbone.onnx"
        export_onnx_if_needed(onnx_path)

        cache_dir = str(model_dir / f"trt_cache_gpu{device_id}")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"  GPU {device_id}: Creating ORT TensorRT session...", flush=True)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(str(onnx_path), opts, providers=[
            ("TensorrtExecutionProvider", {
                "device_id": device_id,
                "trt_fp16_enable": True,
                "trt_max_workspace_size": str(4 * 1024**3),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
            }),
            ("CUDAExecutionProvider", {"device_id": device_id}),
        ])
        self.input_name = self.ort_session.get_inputs()[0].name

        # Check if TRT is active
        active_eps = self.ort_session.get_providers()
        if "TensorrtExecutionProvider" in active_eps:
            print(f"  GPU {device_id}: TensorRT EP active", flush=True)
        else:
            print(f"  GPU {device_id}: WARNING - TRT not active, using {active_eps}", flush=True)

        # Warmup (builds TRT engine on first call)
        print(f"  GPU {device_id}: Warming up (TRT engine build)...", flush=True)
        dummy = np.random.randn(8, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        self.ort_session.run(None, {self.input_name: dummy})
        print(f"  GPU {device_id}: Engine ready", flush=True)

        # Spatial stride indices
        h_idx = torch.arange(0, 50, score_stride)
        w_idx = torch.arange(0, 80, score_stride)
        grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
        self.flat_idx = (grid_h * 80 + grid_w).flatten().numpy()
        self.n_sub = len(self.flat_idx)

        # Load memory banks + stats (on torch GPU for scoring)
        self.banks = {}
        self.bank_sqs = {}
        self.pos_means = {}
        self.pos_stds = {}

        for g in groups:
            group_dir = model_dir / f"group_{g}"
            bank = np.load(group_dir / "memory_bank.npy")
            pos_mean = np.load(group_dir / "pos_mean.npy").flatten()
            pos_std = np.load(group_dir / "pos_std.npy").flatten()

            bank_t = torch.from_numpy(bank).to(self.torch_device).half()
            self.banks[g] = bank_t
            self.bank_sqs[g] = (bank_t.float() ** 2).sum(1).half()

            pm_full = torch.from_numpy(pos_mean).to(self.torch_device).half()
            ps_full = torch.from_numpy(pos_std).to(self.torch_device).half()
            self.pos_means[g] = pm_full[self.flat_idx]
            self.pos_stds[g] = ps_full[self.flat_idx]

            print(f"  GPU {device_id}: Group {g} loaded", flush=True)

    def process_batch(self, batch_np: np.ndarray, group_id: int) -> np.ndarray:
        """
        batch_np: (B, 3, H, W) float32 numpy
        Returns: (B,) max z-scores
        """
        # ORT backbone
        feat_map = self.ort_session.run(None, {self.input_name: batch_np})[0]
        # feat_map: (B, 1536, 50, 80) float32 numpy

        B, C, H, W = feat_map.shape
        # Reshape + subsample spatial positions
        # (B, 1536, 50, 80) → (B, 4000, 1536) → (B, n_sub, 1536)
        features = feat_map.transpose(0, 2, 3, 1).reshape(B, H * W, C)
        features = features[:, self.flat_idx, :]  # (B, n_sub, C)

        # Move to torch GPU for FP16 scoring
        feat_t = torch.from_numpy(features).to(self.torch_device).half()

        bank = self.banks[group_id]
        bank_sq = self.bank_sqs[group_id]
        pos_mean = self.pos_means[group_id]
        pos_std = self.pos_stds[group_id]

        max_z_scores = torch.empty(B, device=self.torch_device)

        for i in range(B):
            f = feat_t[i]  # (n_sub, C)
            f_sq = (f.float() ** 2).sum(1, keepdim=True).half()
            inner = f @ bank.T  # FP16 Tensor Core
            dists_sq = f_sq + bank_sq.unsqueeze(0) - 2 * inner
            min_d = dists_sq.min(dim=1).values.float().clamp(min=0).sqrt()
            z = (min_d - pos_mean.float()) / pos_std.float()
            max_z_scores[i] = z.max()

        return max_z_scores.cpu().numpy()


# ===== ENGINE =====
class FastInferenceEngine:
    def __init__(self, model_dir: Path, groups=None, batch_size=8,
                 score_stride=SCORE_SPATIAL_STRIDE):
        self.model_dir = model_dir
        self.groups = groups or [1, 2, 3, 4, 5]
        self.batch_size = batch_size

        n_gpus = torch.cuda.device_count()
        print(f"\n[Engine] {n_gpus} GPUs, batch={batch_size}/gpu", flush=True)

        self.workers = []
        for i in range(min(n_gpus, 2)):
            self.workers.append(GpuWorker(i, model_dir, self.groups, score_stride))

        print(f"[Engine] Ready. {len(self.workers)} GPUs\n", flush=True)

    def _run_batch(self, batch_tensor, group_id):
        B = batch_tensor.shape[0]
        batch_np = batch_tensor.numpy()
        n_w = len(self.workers)

        if n_w >= 2 and B >= 2:
            split = B // n_w
            results = []
            for i, worker in enumerate(self.workers):
                start = i * split
                end = start + split if i < n_w - 1 else B
                z = worker.process_batch(batch_np[start:end], group_id)
                results.append(z)
            return np.concatenate(results)
        else:
            return self.workers[0].process_batch(batch_np, group_id)

    def infer_folder(self, folder_path, cam_id, group_id, mirror=False, num_workers=8):
        image_paths = get_image_paths(folder_path, cam_id)
        if not image_paths:
            return []

        effective_bs = self.batch_size * len(self.workers)
        dataset = InferenceDataset(image_paths, mirror=mirror)
        loader = DataLoader(
            dataset, batch_size=effective_bs, num_workers=num_workers,
            prefetch_factor=4, pin_memory=True,
            collate_fn=collate_skip_none, persistent_workers=True,
        )

        results = []
        t0 = time.time()
        for batch_tensors, batch_paths in loader:
            if batch_tensors is None:
                continue
            z_scores = self._run_batch(batch_tensors, group_id)
            for j, path in enumerate(batch_paths):
                results.append({
                    "file": Path(path).name,
                    "group": group_id,
                    "max_z_score": float(z_scores[j]),
                    "anomaly": bool(z_scores[j] > Z_SCORE_THRESHOLD),
                })

        elapsed = time.time() - t0
        n = len(results)
        if n > 0:
            print(f"  Cam {cam_id} G{group_id}: {n} imgs, {elapsed:.2f}s "
                  f"({n/elapsed:.1f} img/s, {elapsed/n*1000:.1f}ms/img)", flush=True)
        return results

    def infer_folder_all_groups(self, folder_path, num_workers=8):
        all_results = {}
        t0 = time.time()
        for g in self.groups:
            cfg = GROUP_CONFIG[g]
            all_results[g] = self.infer_folder(
                folder_path, cfg["infer_cam"], g, num_workers=num_workers)
        elapsed = time.time() - t0
        total = sum(len(v) for v in all_results.values())
        print(f"\n  ALL GROUPS: {total} imgs, {elapsed:.2f}s", flush=True)
        return all_results

    def benchmark(self, folder_path, cam_id=1, group_id=1):
        image_paths = get_image_paths(folder_path, cam_id)
        n = len(image_paths)
        if not n:
            print("No images"); return

        effective_bs = self.batch_size * len(self.workers)
        print(f"\n[Benchmark] {n} imgs, cam={cam_id}, group={group_id}, "
              f"bs={effective_bs}, GPUs={len(self.workers)}", flush=True)

        dataset = InferenceDataset(image_paths)

        # Warmup
        print("  Warming up...", flush=True)
        wl = DataLoader(dataset, batch_size=effective_bs, num_workers=8,
                        prefetch_factor=4, pin_memory=True,
                        collate_fn=collate_skip_none, persistent_workers=True)
        for i, (bt, _) in enumerate(wl):
            if bt is not None:
                self._run_batch(bt, group_id)
            if i >= 5: break
        del wl

        # Timed
        print("  Running...", flush=True)
        loader = DataLoader(dataset, batch_size=effective_bs, num_workers=8,
                            prefetch_factor=4, pin_memory=True,
                            collate_fn=collate_skip_none, persistent_workers=True)

        t_start = time.time()
        count = 0
        for batch_tensors, batch_paths in loader:
            if batch_tensors is None:
                continue
            self._run_batch(batch_tensors, group_id)
            count += len(batch_paths)

        t_total = time.time() - t_start
        print(f"\n  === Results ({count} images, {len(self.workers)} GPUs) ===")
        print(f"  Total:      {t_total:.3f}s ({count/t_total:.1f} img/s)")
        print(f"  Per image:  {t_total/count*1000:.2f}ms")
        print(f"  8000 img (1 group): {8000/count*t_total:.1f}s")
        print(f"  8000 img (5 groups): ~{8000/count*t_total*5:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="PatchCore v5b Fast Inference v4 (TRT)")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--cam", type=int, default=None)
    parser.add_argument("--group", type=int, default=None)
    parser.add_argument("--groups", type=str, default="1,2,3,4,5")
    parser.add_argument("--batch", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-stride", type=int, default=SCORE_SPATIAL_STRIDE)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    groups = [int(g) for g in args.groups.split(",")]

    engine = FastInferenceEngine(
        model_dir=MODEL_DIR, groups=groups,
        batch_size=args.batch, score_stride=args.score_stride,
    )

    folder = Path(args.folder)

    if args.benchmark:
        engine.benchmark(folder, cam_id=args.cam or 1, group_id=args.group or groups[0])
        return

    t0 = time.time()
    if args.cam and args.group:
        results = engine.infer_folder(folder, args.cam, args.group, num_workers=args.workers)
        output_data = {"folder": str(folder), "camera": args.cam,
                        "group": args.group, "results": results}
    else:
        results = engine.infer_folder_all_groups(folder, num_workers=args.workers)
        output_data = {"folder": str(folder), "groups": {str(g): r for g, r in results.items()}}
        for g, r in results.items():
            if r:
                n_anom = sum(1 for x in r if x["anomaly"])
                z_vals = [x["max_z_score"] for x in r]
                print(f"  Group {g}: {n_anom}/{len(r)} anomaly, "
                      f"z={min(z_vals):.2f}~{max(z_vals):.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.2f}s")

    out_path = Path(args.output) if args.output else MODEL_DIR / "fast_inference_results.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
