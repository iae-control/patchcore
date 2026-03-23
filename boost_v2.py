#!/usr/bin/env python3
"""PatchCore Boost Experiment v2 — Fast version, no spatial scoring."""
import os, sys, json, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

NAS_ROOT = Path(os.path.expanduser("~/nas_storage"))
OUTPUT_DIR = Path(os.path.expanduser("~/patchcore/output"))
MB_PATH = OUTPUT_DIR / "150x75" / "group_1" / "memory_bank.npy"
TILE_SIZE = 256
IMAGE_W, IMAGE_H = 1920, 1200

TEST_FOLDERS = [
    "20251105/20251105042235_150x75",
    "20251105/20251105063711_150x75",
    "20251105/20251105101801_150x75",
    "20251105/20251105134001_150x75",
    "20251105/20251105171148_150x75",
]
REF_FOLDER = "20251105/20251105005723_150x75"

DEFECT_TYPES = {
    "spot":          {"type": "circle", "radius": 5,  "diff": 30, "blur": 0},
    "big_spot":      {"type": "circle", "radius": 10, "diff": 30, "blur": 0},
    "stain":         {"type": "circle", "radius": 15, "diff": 25, "blur": 7},
    "scratch":       {"type": "line",   "width": 1,  "diff": 30, "zigzag": False, "count": 1},
    "crack":         {"type": "line",   "width": 1,  "diff": 25, "zigzag": True,  "count": 1},
    "multi_scratch": {"type": "line",   "width": 1,  "diff": 25, "zigzag": False, "count": 3},
    "faint_spot":    {"type": "circle", "radius": 5,  "diff": 10, "blur": 0},
    "thin_scratch":  {"type": "line",   "width": 1,  "diff": 15, "zigzag": False, "count": 1},
    "gradient_stain":{"type": "gradient", "diff": 16},
}

def inject_defect(tile, name, rng):
    tile = tile.copy(); h, w = tile.shape[:2]
    spec = DEFECT_TYPES[name]
    cx, cy = rng.randint(40, w-40), rng.randint(40, h-40)
    diff, sign = spec["diff"], rng.choice([-1, 1])
    if spec["type"] == "circle":
        mask = np.zeros((h,w), np.float32)
        cv2.circle(mask, (cx,cy), spec["radius"], 1.0, -1)
        if spec.get("blur", 0) > 0:
            k = spec["blur"]
            mask = cv2.GaussianBlur(mask, (k*2+1, k*2+1), k)
        tile = np.clip(tile.astype(np.float32) + sign*diff*mask, 0, 255).astype(np.uint8)
    elif spec["type"] == "line":
        for _ in range(spec.get("count", 1)):
            val = int(np.clip(tile[cy, cx] + sign*diff, 0, 255))
            if spec.get("zigzag"):
                pts = [(rng.randint(20,w-20), rng.randint(20,40))]
                for _ in range(5):
                    x2 = max(5, min(w-5, pts[-1][0]+rng.randint(-15,15)))
                    y2 = min(h-5, pts[-1][1]+rng.randint(20,40))
                    pts.append((x2,y2))
                for i in range(len(pts)-1):
                    cv2.line(tile, pts[i], pts[i+1], val, spec["width"])
            else:
                if rng.random() < 0.5:
                    x = rng.randint(30, w-30)
                    cv2.line(tile, (x,10), (x+rng.randint(-10,10), h-10), val, spec["width"])
                else:
                    y = rng.randint(30, h-30)
                    cv2.line(tile, (10,y), (w-10, y+rng.randint(-10,10)), val, spec["width"])
    elif spec["type"] == "gradient":
        mask = np.zeros((h,w), np.float32)
        cv2.circle(mask, (cx,cy), 30, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31,31), 10)
        mask /= (mask.max()+1e-8)
        tile = np.clip(tile.astype(np.float32) + sign*diff*mask, 0, 255).astype(np.uint8)
    return tile

def extract_tiles(img, indices):
    cols = IMAGE_W // TILE_SIZE
    return [img[(i//cols)*TILE_SIZE:(i//cols+1)*TILE_SIZE, (i%cols)*TILE_SIZE:(i%cols+1)*TILE_SIZE]
            for i in indices if img[(i//cols)*TILE_SIZE:(i//cols+1)*TILE_SIZE].shape[0] == TILE_SIZE]

def tile_to_tensor(t, clahe=False):
    if clahe:
        t = cv2.createCLAHE(2.0, (8,8)).apply(t)
    t = cv2.resize(t, (224,224))
    t3 = np.stack([t]*3, axis=2).astype(np.float32) / 255.0
    from torchvision import transforms
    return transforms.Normalize([.485,.456,.406],[.229,.224,.225])(transforms.ToTensor()(t3))

class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        bb = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
        self.l2 = bb.layer2; self.l3 = bb.layer3
        self.cuda().eval()
    @torch.no_grad()
    def forward(self, x):
        x = x.cuda()
        h = self.stem(x); f2 = self.l2(h); f3 = self.l3(f2)
        f3u = F.interpolate(f3, f2.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([f2, f3u], 1)
        return F.adaptive_avg_pool2d(feat, 1).flatten(1).cpu().numpy()

def knn_scores(feats, bank, ks=[1,3,5,9]):
    """Batch kNN scoring. feats: (N,D), bank: (M,D). Returns dict k->scores(N,)"""
    # Use torch for speed
    feats_t = torch.from_numpy(feats).cuda()
    bank_t = torch.from_numpy(bank).cuda()
    dists = torch.cdist(feats_t, bank_t)  # (N, M)
    result = {}
    max_k = max(ks)
    topk_d, _ = torch.topk(dists, max_k, dim=1, largest=False)  # (N, max_k)
    for k in ks:
        result[k] = topk_d[:, :k].mean(dim=1).cpu().numpy()
    return result

def fft_score(tile):
    mag = np.abs(np.fft.fftshift(np.fft.fft2(tile.astype(np.float32))))
    h, w = mag.shape; cy, cx = h//2, w//2; r = min(h,w)//4
    Y, X = np.ogrid[:h,:w]; mask = (Y-cy)**2+(X-cx)**2 <= r**2
    return mag[~mask].mean()

def dct_score(tile):
    d = cv2.dct(tile.astype(np.float32)); h,w=d.shape
    return np.abs(d[h//2:, w//2:]).mean()

def ssim_score(tile, ref):
    C1,C2 = 6.5025, 58.5225
    t,r = tile.astype(np.float64), ref.astype(np.float64)
    mt = cv2.GaussianBlur(t,(11,11),1.5); mr = cv2.GaussianBlur(r,(11,11),1.5)
    st2 = cv2.GaussianBlur(t**2,(11,11),1.5)-mt**2
    sr2 = cv2.GaussianBlur(r**2,(11,11),1.5)-mr**2
    str_ = cv2.GaussianBlur(t*r,(11,11),1.5)-mt*mr
    ssim = ((2*mt*mr+C1)*(2*str_+C2))/((mt**2+mr**2+C1)*(st2+sr2+C2))
    return 1.0 - float(np.mean(ssim))

def grad_score(tile):
    gx = cv2.Sobel(tile, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(tile, cv2.CV_64F, 0, 1, 3)
    return float(np.mean(np.sqrt(gx**2+gy**2)))

def main():
    t0 = time.time()
    print("="*60)
    print("PATCHCORE BOOST v2")
    print("="*60)
    
    bank = np.load(str(MB_PATH))
    print(f"Bank: {bank.shape}")
    ext = Extractor()
    print(f"CNN loaded ({time.time()-t0:.1f}s)")
    
    tile_idx = [8, 14, 20]
    
    # ── Ref stats ──
    print("\n[REF] Reference statistics...")
    ref_cam = NAS_ROOT / REF_FOLDER / "camera_1"
    ref_imgs = sorted([f for f in ref_cam.iterdir() if f.suffix.lower() in ('.jpg','.png')])[:5]
    
    ref_tiles = []
    for p in ref_imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W): img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        ref_tiles.extend(extract_tiles(img, tile_idx))
    
    ref_avg = {idx: np.mean([ref_tiles[j*3+i] for j in range(len(ref_imgs)) if j*3+i<len(ref_tiles)], 0).astype(np.uint8) 
               for i, idx in enumerate(tile_idx)}
    
    # CNN features for ref tiles (normal + CLAHE)
    ref_tensors = torch.stack([tile_to_tensor(t) for t in ref_tiles])
    ref_tensors_cl = torch.stack([tile_to_tensor(t, clahe=True) for t in ref_tiles])
    
    ref_feats = ext(ref_tensors)
    ref_feats_cl = ext(ref_tensors_cl)
    
    ref_knn = knn_scores(ref_feats, bank, [1,3,5,9])
    ref_knn_cl = knn_scores(ref_feats_cl, bank, [1,3,5,9])
    
    ref_noncnn = {"fft": [], "dct": [], "ssim": [], "grad": []}
    for t in ref_tiles:
        ref_noncnn["fft"].append(fft_score(t))
        ref_noncnn["dct"].append(dct_score(t))
        ref_noncnn["ssim"].append(ssim_score(t, ref_avg[tile_idx[0]]))
        ref_noncnn["grad"].append(grad_score(t))
    
    ref_stats = {}
    for k in [1,3,5,9]:
        ref_stats[f"k{k}"] = {"mean": ref_knn[k].mean(), "std": max(ref_knn[k].std(), 1e-6)}
        ref_stats[f"cl_k{k}"] = {"mean": ref_knn_cl[k].mean(), "std": max(ref_knn_cl[k].std(), 1e-6)}
    for m in ["fft","dct","ssim","grad"]:
        ref_stats[m] = {"mean": np.mean(ref_noncnn[m]), "std": max(np.std(ref_noncnn[m]), 1e-6)}
    
    for m, s in ref_stats.items():
        print(f"  {m}: mean={s['mean']:.4f} std={s['std']:.4f}")
    
    # ── Test ──
    print(f"\n[TEST] Testing on {len(TEST_FOLDERS)} folders...")
    # Store raw scores per method per defect
    scores_by = defaultdict(lambda: defaultdict(list))  # scores_by[method][defect] = [score1, ...]
    normal_scores_by = defaultdict(list)  # normal_scores_by[method] = [score1, ...]
    
    rng = np.random.RandomState(42)
    
    for fi, frel in enumerate(TEST_FOLDERS):
        folder = NAS_ROOT / frel
        cam1 = folder / "camera_1"
        imgs = sorted([f for f in cam1.iterdir() if f.suffix.lower() in ('.jpg','.png')])
        if not imgs: continue
        img = cv2.imread(str(imgs[len(imgs)//2]), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W): img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        
        clean_tiles = extract_tiles(img, tile_idx)
        if len(clean_tiles) < 3: continue
        print(f"  [{fi+1}/5] {frel.split('/')[-1]}")
        
        # Normal scores for this folder
        normal_tensors = torch.stack([tile_to_tensor(t) for t in clean_tiles])
        normal_tensors_cl = torch.stack([tile_to_tensor(t, clahe=True) for t in clean_tiles])
        normal_feats = ext(normal_tensors)
        normal_feats_cl = ext(normal_tensors_cl)
        normal_knn = knn_scores(normal_feats, bank, [1,3,5,9])
        normal_knn_cl = knn_scores(normal_feats_cl, bank, [1,3,5,9])
        
        for k in [1,3,5,9]:
            normal_scores_by[f"k{k}"].extend(normal_knn[k].tolist())
            normal_scores_by[f"cl_k{k}"].extend(normal_knn_cl[k].tolist())
        for ti, t in enumerate(clean_tiles):
            normal_scores_by["fft"].append(fft_score(t))
            normal_scores_by["dct"].append(dct_score(t))
            normal_scores_by["ssim"].append(ssim_score(t, ref_avg.get(tile_idx[ti], t)))
            normal_scores_by["grad"].append(grad_score(t))
        
        # Defect scores
        for dname in DEFECT_TYPES:
            defect_tensors = []
            defect_tensors_cl = []
            defect_tiles_raw = []
            for ti, (tidx, ct) in enumerate(zip(tile_idx, clean_tiles)):
                dt = inject_defect(ct, dname, rng)
                defect_tiles_raw.append((tidx, dt))
                defect_tensors.append(tile_to_tensor(dt))
                defect_tensors_cl.append(tile_to_tensor(dt, clahe=True))
            
            d_feats = ext(torch.stack(defect_tensors))
            d_feats_cl = ext(torch.stack(defect_tensors_cl))
            d_knn = knn_scores(d_feats, bank, [1,3,5,9])
            d_knn_cl = knn_scores(d_feats_cl, bank, [1,3,5,9])
            
            for k in [1,3,5,9]:
                scores_by[f"k{k}"][dname].extend(d_knn[k].tolist())
                scores_by[f"cl_k{k}"][dname].extend(d_knn_cl[k].tolist())
            
            for ti, (tidx, dt) in enumerate(defect_tiles_raw):
                scores_by["fft"][dname].append(fft_score(dt))
                scores_by["dct"][dname].append(dct_score(dt))
                scores_by["ssim"][dname].append(ssim_score(dt, ref_avg.get(tidx, dt)))
                scores_by["grad"][dname].append(grad_score(dt))
    
    # ── Evaluation ──
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    ALL_METHODS = [f"k{k}" for k in [1,3,5,9]] + [f"cl_k{k}" for k in [1,3,5,9]] + ["fft","dct","ssim","grad"]
    
    # Compute ratios (score / ref_mean)
    ratios = defaultdict(lambda: defaultdict(list))
    for m in ALL_METHODS:
        mu = ref_stats[m]["mean"]
        for dname in DEFECT_TYPES:
            for s in scores_by[m][dname]:
                ratios[m][dname].append(s / mu if mu > 0 else 0)
    
    # Compute z-scores for ensemble
    z_scores = defaultdict(lambda: defaultdict(list))
    for m in ALL_METHODS:
        mu, sig = ref_stats[m]["mean"], ref_stats[m]["std"]
        for dname in DEFECT_TYPES:
            for s in scores_by[m][dname]:
                z_scores[m][dname].append((s - mu) / sig)
    
    # Ensemble: max z-score across methods
    # ens_baseline: k1 + fft + dct + ssim + grad (current-like)
    # ens_best_cnn: k3 + cl_k3
    # ens_all: all methods
    # ens_smart: k3 + cl_k3 + fft + ssim (drop weak: dct, grad contribute less?)
    # ens_or: detected if ANY method ratio > threshold
    
    ens_configs = {
        "ens_base": ["k1", "fft", "dct", "ssim", "grad"],
        "ens_k3": ["k3", "fft", "dct", "ssim", "grad"],
        "ens_cnn": ["k3", "k5", "cl_k3"],
        "ens_smart": ["k3", "cl_k3", "fft", "ssim"],
        "ens_all": ALL_METHODS,
        "ens_no_weak": ["k3", "k5", "cl_k3", "fft", "ssim"],
    }
    
    for ens_name, methods in ens_configs.items():
        for dname in DEFECT_TYPES:
            n = len(z_scores[methods[0]][dname])
            for i in range(n):
                max_z = max(z_scores[m][dname][i] for m in methods)
                ratios[ens_name][dname].append(1.0 + max_z * 0.1)  # z->ratio approx
    
    # OR-ensemble: detected if ANY method > threshold  
    for dname in DEFECT_TYPES:
        n = len(ratios["k1"][dname])
        for i in range(n):
            # ratio > 1.1 for any of the good methods
            any_detected = any(ratios[m][dname][i] > 1.1 for m in ["k3", "k5", "cl_k3", "fft", "ssim"])
            ratios["ens_or_1.1"][dname].append(2.0 if any_detected else 0.5)
            any_detected2 = any(ratios[m][dname][i] > 1.05 for m in ["k3", "cl_k3", "fft", "ssim", "grad"])
            ratios["ens_or_1.05"][dname].append(2.0 if any_detected2 else 0.5)
    
    ALL_EVAL = ALL_METHODS + list(ens_configs.keys()) + ["ens_or_1.1", "ens_or_1.05"]
    
    # Find best threshold per method
    results = []
    for m in ALL_EVAL:
        best = (0, 1.1)
        for thr in [1.0, 1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3]:
            det = sum(1 for d in DEFECT_TYPES for r in ratios[m][d] if r > thr)
            total = sum(len(ratios[m][d]) for d in DEFECT_TYPES)
            # Also count false positives on normal
            if m in ALL_METHODS:
                normal_ratios = [s / ref_stats[m]["mean"] for s in normal_scores_by[m]] if ref_stats[m]["mean"] > 0 else []
                fp = sum(1 for r in normal_ratios if r > thr)
            else:
                fp = 0
            if det > best[0] or (det == best[0] and thr > best[1]):
                best = (det, thr)
        det, thr = best
        total = sum(len(ratios[m][d]) for d in DEFECT_TYPES)
        rate = det / total * 100 if total > 0 else 0
        results.append((m, thr, rate, det, total))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'Method':<18} {'Thr':>6} {'Rate':>8} {'Det':>6}")
    print("-"*42)
    for m, thr, rate, det, total in results:
        marker = " ★" if rate > 76.3 else ""
        print(f"{m:<18} {thr:>6.2f} {rate:>7.1f}% {det:>3}/{total}{marker}")
    
    # Defect breakdown for top methods
    top8 = [r[0] for r in results[:8]]
    print(f"\n{'Defect':<16}", end="")
    for m in top8:
        print(f" {m[:10]:>10}", end="")
    print()
    print("-"*(16+11*len(top8)))
    
    for dname in DEFECT_TYPES:
        print(f"{dname:<16}", end="")
        for m in top8:
            thr = [r[1] for r in results if r[0]==m][0]
            total = len(ratios[m][dname])
            det = sum(1 for r in ratios[m][dname] if r > thr)
            rate = det/total*100 if total > 0 else 0
            print(f" {rate:>9.0f}%", end="")
        print()
    
    # False positive analysis
    print(f"\n--- False Positives (normal tiles detected as defect) ---")
    for m in top8:
        if m not in ALL_METHODS: continue
        thr = [r[1] for r in results if r[0]==m][0]
        normal_r = [s / ref_stats[m]["mean"] for s in normal_scores_by[m]] if ref_stats[m]["mean"] > 0 else []
        fp = sum(1 for r in normal_r if r > thr)
        print(f"  {m}: {fp}/{len(normal_r)} FP at thr={thr:.2f}")
    
    # Save
    out = {"results": results, "ref_stats": {k: v for k, v in ref_stats.items()}}
    with open(str(OUTPUT_DIR / "150x75" / "boost_v2_results.json"), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("DONE!")

if __name__ == "__main__":
    main()
