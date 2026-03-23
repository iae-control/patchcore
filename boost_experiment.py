#!/usr/bin/env python3
"""PatchCore Detection Improvement Experiments
Tests 5 approaches on 150x75 group_1 with synthetic defects.
GPU-light: uses CUDA:1, batch_size=4, torch.no_grad() only.
"""
import os, sys, time, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

# ── Setup ──
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 only
sys.path.insert(0, os.path.expanduser('~/patchcore'))

from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

NAS_ROOT = Path(os.path.expanduser("~/nas_storage"))
OUTPUT_DIR = Path(os.path.expanduser("~/patchcore/output"))
MEMORY_BANK_PATH = OUTPUT_DIR / "150x75" / "group_1" / "memory_bank.npy"
TILE_SIZE = 256
IMAGE_W, IMAGE_H = 1920, 1200
SEED = 42
np.random.seed(SEED)

# Test folders (from generalization test)
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

def inject_defect(tile, defect_name, rng):
    tile = tile.copy()
    h, w = tile.shape[:2]
    spec = DEFECT_TYPES[defect_name]
    cx, cy = rng.randint(40, w-40), rng.randint(40, h-40)
    diff = spec["diff"]
    sign = rng.choice([-1, 1])
    if spec["type"] == "circle":
        r = spec["radius"]
        blur_k = spec.get("blur", 0)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        if blur_k > 0:
            mask = cv2.GaussianBlur(mask, (blur_k*2+1, blur_k*2+1), blur_k)
        tile_f = tile.astype(np.float32) + sign * diff * mask
        tile = np.clip(tile_f, 0, 255).astype(np.uint8)
    elif spec["type"] == "line":
        count = spec.get("count", 1)
        width = spec.get("width", 1)
        zigzag = spec.get("zigzag", False)
        for _ in range(count):
            if zigzag:
                pts = []
                x, y = rng.randint(20, w-20), rng.randint(20, 40)
                pts.append((x, y))
                for _ in range(5):
                    x += rng.randint(-15, 15); y += rng.randint(20, 40)
                    x = max(5, min(w-5, x)); y = min(h-5, y)
                    pts.append((x, y))
                for i in range(len(pts)-1):
                    cv2.line(tile, pts[i], pts[i+1], int(np.clip(tile[cy, cx] + sign*diff, 0, 255)), width)
            else:
                if rng.random() < 0.5:
                    x = rng.randint(30, w-30)
                    cv2.line(tile, (x, 10), (x + rng.randint(-10,10), h-10),
                             int(np.clip(tile[cy, cx] + sign*diff, 0, 255)), width)
                else:
                    y = rng.randint(30, h-30)
                    cv2.line(tile, (10, y), (w-10, y + rng.randint(-10,10)),
                             int(np.clip(tile[cy, cx] + sign*diff, 0, 255)), width)
    elif spec["type"] == "gradient":
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), 30, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = mask / (mask.max() + 1e-8)
        tile_f = tile.astype(np.float32) + sign * diff * mask
        tile = np.clip(tile_f, 0, 255).astype(np.uint8)
    return tile


def extract_tiles(img_gray, tile_indices):
    tiles = []
    cols = IMAGE_W // TILE_SIZE
    for idx in tile_indices:
        r, c = divmod(idx, cols)
        y, x = r * TILE_SIZE, c * TILE_SIZE
        tile = img_gray[y:y+TILE_SIZE, x:x+TILE_SIZE]
        if tile.shape == (TILE_SIZE, TILE_SIZE):
            tiles.append(tile)
    return tiles


class FeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)
        avg = F.adaptive_avg_pool2d(combined, 1).squeeze(-1).squeeze(-1)
        return avg, combined  # avg: (B,1536), combined: (B,1536,H,W)


def tile_to_tensor(tile_gray, clahe=False):
    """Convert grayscale tile to 3-channel normalized tensor."""
    if clahe:
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        tile_gray = cl.apply(tile_gray)
    t = cv2.resize(tile_gray, (224, 224))
    t3 = np.stack([t]*3, axis=2).astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(t3)
    tensor = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(tensor)
    return tensor


def knn_score(feat, bank, k=1):
    dists = np.linalg.norm(bank - feat, axis=1)
    return np.sort(dists)[:k].mean()


def spatial_topk_score(spatial_feat, bank, topk_spatial=3, k_nn=1):
    """Spatial scoring: for each position compute kNN, take top-K positions."""
    C, H, W = spatial_feat.shape
    positions = spatial_feat.reshape(C, -1).T  # (HW, C)
    # Batch distance computation
    pos_t = torch.from_numpy(positions).cuda()
    bank_t = torch.from_numpy(bank).cuda()
    dists = torch.cdist(pos_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)  # (HW, M)
    if k_nn == 1:
        min_dists = dists.min(dim=1).values
    else:
        min_dists = torch.topk(dists, k_nn, dim=1, largest=False).values.mean(dim=1)
    topk_vals = torch.topk(min_dists, min(topk_spatial, len(min_dists))).values
    return topk_vals.mean().item()


def compute_fft_score(tile):
    mag = np.abs(np.fft.fft2(tile.astype(np.float32)))
    mag = np.fft.fftshift(mag)
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    mask = (Y-cy)**2 + (X-cx)**2 <= r**2
    return mag[~mask].mean()

def compute_dct_score(tile):
    dct = cv2.dct(tile.astype(np.float32))
    h, w = dct.shape
    return np.abs(dct[h//2:, w//2:]).mean()

def compute_ssim_score(tile, ref_tile):
    C1, C2 = 6.5025, 58.5225
    t = tile.astype(np.float64)
    r = ref_tile.astype(np.float64)
    mu_t = cv2.GaussianBlur(t, (11,11), 1.5)
    mu_r = cv2.GaussianBlur(r, (11,11), 1.5)
    sigma_t2 = cv2.GaussianBlur(t**2, (11,11), 1.5) - mu_t**2
    sigma_r2 = cv2.GaussianBlur(r**2, (11,11), 1.5) - mu_r**2
    sigma_tr = cv2.GaussianBlur(t*r, (11,11), 1.5) - mu_t*mu_r
    ssim_map = ((2*mu_t*mu_r + C1)*(2*sigma_tr + C2)) / ((mu_t**2 + mu_r**2 + C1)*(sigma_t2 + sigma_r2 + C2))
    return 1.0 - float(np.mean(ssim_map))

def compute_gradient_score(tile):
    gx = cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def main():
    print("=" * 70)
    print("PATCHCORE BOOST EXPERIMENT")
    print("=" * 70)
    
    # Load memory bank
    memory_bank = np.load(str(MEMORY_BANK_PATH))
    print(f"Memory bank: {memory_bank.shape}")
    
    # Setup CNN
    print("Loading CNN on cuda:0...")
    extractor = FeatureExtractor('cuda')
    
    # Tile indices for testing (spread across image)
    test_tile_indices = [8, 14, 20]
    
    # ── Build reference stats from ref folder ──
    print("\n[REF] Building reference stats...")
    ref_folder = NAS_ROOT / REF_FOLDER
    ref_cam1 = ref_folder / "camera_1"
    ref_imgs = sorted([f for f in ref_cam1.iterdir() if f.suffix.lower() in ('.jpg','.png','.bmp')])[:5]
    
    # Collect reference normal tiles
    ref_tiles_all = []
    for img_path in ref_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        tiles = extract_tiles(img, test_tile_indices)
        ref_tiles_all.extend(tiles)
    
    # Ref tile averages (for SSIM)
    ref_tile_avg = {}
    for i, idx in enumerate(test_tile_indices):
        tiles_at = [ref_tiles_all[j*3+i] for j in range(len(ref_imgs)) if j*3+i < len(ref_tiles_all)]
        ref_tile_avg[idx] = np.mean(tiles_at, axis=0).astype(np.uint8)
    
    # Compute reference scores for all methods
    ref_scores = defaultdict(list)
    
    for tile in ref_tiles_all:
        # Non-CNN
        ref_scores["fft"].append(compute_fft_score(tile))
        ref_scores["dct"].append(compute_dct_score(tile))
        ref_scores["gradient"].append(compute_gradient_score(tile))
        ref_scores["ssim"].append(compute_ssim_score(tile, ref_tile_avg[test_tile_indices[0]]))
        
        # CNN - original (k=1)
        tensor = tile_to_tensor(tile).unsqueeze(0)
        avg_feat, spatial_feat = extractor(tensor)
        avg_np = avg_feat.cpu().numpy()[0]
        spatial_np = spatial_feat.cpu().numpy()[0]
        
        ref_scores["orig_k1"].append(knn_score(avg_np, memory_bank, k=1))
        ref_scores["orig_k3"].append(knn_score(avg_np, memory_bank, k=3))
        ref_scores["orig_k5"].append(knn_score(avg_np, memory_bank, k=5))
        ref_scores["orig_k9"].append(knn_score(avg_np, memory_bank, k=9))
        ref_scores["hm_top3"].append(spatial_topk_score(spatial_np, memory_bank, topk_spatial=3))
        ref_scores["hm_top5"].append(spatial_topk_score(spatial_np, memory_bank, topk_spatial=5))
        
        # CLAHE version
        tensor_cl = tile_to_tensor(tile, clahe=True).unsqueeze(0)
        avg_cl, spatial_cl = extractor(tensor_cl)
        avg_cl_np = avg_cl.cpu().numpy()[0]
        spatial_cl_np = spatial_cl.cpu().numpy()[0]
        ref_scores["clahe_orig_k1"].append(knn_score(avg_cl_np, memory_bank, k=1))
        ref_scores["clahe_orig_k3"].append(knn_score(avg_cl_np, memory_bank, k=3))
        ref_scores["clahe_hm_top3"].append(spatial_topk_score(spatial_cl_np, memory_bank, topk_spatial=3))
    
    ref_normal = {}
    for method, vals in ref_scores.items():
        ref_normal[method] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-6)}
        print(f"  {method}: mean={ref_normal[method]['mean']:.4f}, std={ref_normal[method]['std']:.4f}")
    
    # ── Define all methods to test ──
    METHODS = [
        # Baseline
        "orig_k1",       # Original PatchCore (k=1)
        # Experiment A: k-value tuning
        "orig_k3",       # k=3
        "orig_k5",       # k=5
        "orig_k9",       # k=9
        # Experiment B: Spatial scoring
        "hm_top3",       # Heatmap TopK=3
        "hm_top5",       # Heatmap TopK=5
        # Experiment C: CLAHE preprocessing
        "clahe_orig_k1", # CLAHE + Original k=1
        "clahe_orig_k3", # CLAHE + Original k=3
        "clahe_hm_top3", # CLAHE + Heatmap TopK=3
        # Non-CNN
        "fft", "dct", "gradient", "ssim",
    ]
    
    # ── Test on 5 folders ──
    print("\n[TEST] Running on 5 folders...")
    # results[method][defect_type] = list of (ratio, detected@1.1)
    all_ratios = defaultdict(lambda: defaultdict(list))
    rng = np.random.RandomState(SEED)
    
    for fi, folder_rel in enumerate(TEST_FOLDERS):
        folder = NAS_ROOT / folder_rel
        cam1 = folder / "camera_1"
        imgs = sorted([f for f in cam1.iterdir() if f.suffix.lower() in ('.jpg','.png','.bmp')])
        if not imgs: continue
        test_img_path = imgs[len(imgs)//2]
        
        img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        
        clean_tiles = extract_tiles(img, test_tile_indices)
        if len(clean_tiles) < 3: continue
        
        print(f"\n  Folder [{fi+1}/5]: {folder_rel.split('/')[-1]}")
        
        for defect_name in DEFECT_TYPES:
            for ti, (tile_idx, clean_tile) in enumerate(zip(test_tile_indices, clean_tiles)):
                defect_tile = inject_defect(clean_tile, defect_name, rng)
                
                # Extract features
                tensor = tile_to_tensor(defect_tile).unsqueeze(0)
                avg_feat, spatial_feat = extractor(tensor)
                avg_np = avg_feat.cpu().numpy()[0]
                spatial_np = spatial_feat.cpu().numpy()[0]
                
                tensor_cl = tile_to_tensor(defect_tile, clahe=True).unsqueeze(0)
                avg_cl, spatial_cl = extractor(tensor_cl)
                avg_cl_np = avg_cl.cpu().numpy()[0]
                spatial_cl_np = spatial_cl.cpu().numpy()[0]
                
                scores = {}
                scores["orig_k1"] = knn_score(avg_np, memory_bank, k=1)
                scores["orig_k3"] = knn_score(avg_np, memory_bank, k=3)
                scores["orig_k5"] = knn_score(avg_np, memory_bank, k=5)
                scores["orig_k9"] = knn_score(avg_np, memory_bank, k=9)
                scores["hm_top3"] = spatial_topk_score(spatial_np, memory_bank, topk_spatial=3)
                scores["hm_top5"] = spatial_topk_score(spatial_np, memory_bank, topk_spatial=5)
                scores["clahe_orig_k1"] = knn_score(avg_cl_np, memory_bank, k=1)
                scores["clahe_orig_k3"] = knn_score(avg_cl_np, memory_bank, k=3)
                scores["clahe_hm_top3"] = spatial_topk_score(spatial_cl_np, memory_bank, topk_spatial=3)
                scores["fft"] = compute_fft_score(defect_tile)
                scores["dct"] = compute_dct_score(defect_tile)
                scores["gradient"] = compute_gradient_score(defect_tile)
                scores["ssim"] = compute_ssim_score(defect_tile, ref_tile_avg.get(tile_idx, clean_tile))
                
                for method in METHODS:
                    ratio = scores[method] / ref_normal[method]["mean"] if ref_normal[method]["mean"] > 0 else 0
                    all_ratios[method][defect_name].append(ratio)
        
        print(f"    Done ({len(DEFECT_TYPES)} defects × {len(test_tile_indices)} tiles)")
    
    # ── Compute ensemble methods ──
    print("\n[ENSEMBLE] Computing ensemble scores...")
    
    # Ensemble methods: combine via max z-score
    n_cases = len(all_ratios["orig_k1"][list(DEFECT_TYPES.keys())[0]])
    
    # Ens_MAX_baseline: Original (k1) + all non-CNN (like current system)
    baseline_methods = ["orig_k1", "hm_top3", "fft", "dct", "gradient", "ssim"]
    # Ens_MAX_v2: improved methods only
    improved_methods = ["orig_k3", "hm_top5", "fft", "dct", "ssim", "gradient"]
    # Ens_MAX_clahe: with CLAHE
    clahe_methods = ["clahe_orig_k3", "clahe_hm_top3", "fft", "dct", "ssim", "gradient"]
    # Ens_MAX_best: best CNN + best non-CNN
    best_methods = ["orig_k3", "orig_k5", "hm_top5", "clahe_orig_k3", "fft", "ssim", "gradient"]
    # Ens_OR: detected if ANY individual method has ratio > 1.1
    
    ensemble_configs = {
        "ens_baseline": baseline_methods,
        "ens_improved": improved_methods,
        "ens_clahe": clahe_methods,
        "ens_best": best_methods,
    }
    
    for ens_name, ens_methods in ensemble_configs.items():
        for defect_name in DEFECT_TYPES:
            ratios_list = []
            for method in ens_methods:
                ratios_list.append(all_ratios[method][defect_name])
            # For each case, take max ratio
            for i in range(len(ratios_list[0])):
                max_ratio = max(r[i] for r in ratios_list)
                all_ratios[ens_name][defect_name].append(max_ratio)
    
    # Weighted ensemble: weighted average of z-scores, then check threshold
    for defect_name in DEFECT_TYPES:
        for i in range(n_cases):
            # Compute z-scores
            zs = {}
            for m in ["orig_k3", "hm_top5", "fft", "ssim", "gradient"]:
                zs[m] = (all_ratios[m][defect_name][i] * ref_normal[m]["mean"] - ref_normal[m]["mean"]) / ref_normal[m]["std"]
            # Weights: higher for CNN methods
            weighted_z = 0.35*zs["orig_k3"] + 0.25*zs["hm_top5"] + 0.15*zs["fft"] + 0.15*zs["ssim"] + 0.10*zs["gradient"]
            # Convert back to ratio-like score
            all_ratios["ens_weighted"][defect_name].append(1.0 + weighted_z * 0.1)  # scale factor
    
    # ── Evaluate all methods ──
    THRESHOLDS = [1.05, 1.1, 1.15, 1.2, 1.3]
    all_methods = METHODS + list(ensemble_configs.keys()) + ["ens_weighted"]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Find best threshold for each method
    best_per_method = {}
    for method in all_methods:
        best_thr = 1.1
        best_det = 0
        for thr in THRESHOLDS:
            total = sum(len(all_ratios[method][d]) for d in DEFECT_TYPES)
            det = sum(1 for d in DEFECT_TYPES for r in all_ratios[method][d] if r > thr)
            if det > best_det:
                best_det = det
                best_thr = thr
        best_per_method[method] = (best_thr, best_det)
    
    # Print summary
    print(f"\n{'Method':<20} {'Best_Thr':>8} {'Det_Rate':>10} {'Total':>6}")
    print("-" * 50)
    
    results_for_report = []
    for method in sorted(all_methods, key=lambda m: best_per_method[m][1], reverse=True):
        thr, det = best_per_method[method]
        total = sum(len(all_ratios[method][d]) for d in DEFECT_TYPES)
        rate = det / total * 100 if total > 0 else 0
        print(f"{method:<20} {thr:>8.2f} {rate:>9.1f}% {det:>3}/{total}")
        results_for_report.append((method, thr, rate, det, total))
    
    # Per-defect breakdown for top methods
    top_methods = [m for m, _, _ , _, _ in sorted(results_for_report, key=lambda x: x[2], reverse=True)[:8]]
    
    print(f"\n{'Defect':<16}", end="")
    for m in top_methods:
        print(f" {m[:12]:>12}", end="")
    print()
    print("-" * (16 + 13 * len(top_methods)))
    
    defect_rates = {}
    for defect in DEFECT_TYPES:
        print(f"{defect:<16}", end="")
        defect_rates[defect] = {}
        for m in top_methods:
            thr = best_per_method[m][0]
            total = len(all_ratios[m][defect])
            det = sum(1 for r in all_ratios[m][defect] if r > thr)
            rate = det / total * 100 if total > 0 else 0
            defect_rates[defect][m] = rate
            print(f" {rate:>11.0f}%", end="")
        print()
    
    # Save results
    result_data = {
        "summary": [(m, t, r, d, tot) for m, t, r, d, tot in results_for_report],
        "defect_rates": defect_rates,
        "ref_normal": {k: v for k, v in ref_normal.items()},
    }
    out_path = OUTPUT_DIR / "150x75" / "boost_experiment_results.json"
    with open(str(out_path), 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
