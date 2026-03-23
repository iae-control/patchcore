#!/usr/bin/env python3
"""194x150 Ens_MAX Generalization Test — Final Version
Uses group_1 memory_bank.npy, tests on 5 different production date folders.
Reference stats from a 6th folder (different date).
"""
import os, sys, random, time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ──
NAS_ROOT = Path(os.path.expanduser("~/nas_storage"))
OUTPUT_DIR = Path(os.path.expanduser("~/patchcore/output"))
MEMORY_BANK_PATH = OUTPUT_DIR / "194x150" / "group_1" / "memory_bank.npy"
TILE_SIZE = 256
IMAGE_W, IMAGE_H = 1920, 1200
TRIM_HEAD = 100
TRIM_TAIL = 100
DETECTION_THRESHOLD = 1.1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Defect definitions ──
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
    import cv2
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
        tile_f = tile.astype(np.float32)
        tile_f += sign * diff * mask
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
                    x += rng.randint(-15, 15)
                    y += rng.randint(20, 40)
                    x = max(5, min(w-5, x))
                    y = min(h-5, y)
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
        r = 30
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = mask / (mask.max() + 1e-8)
        tile_f = tile.astype(np.float32)
        tile_f += sign * diff * mask
        tile = np.clip(tile_f, 0, 255).astype(np.uint8)
    return tile


# Hardcoded folders with camera_1 images confirmed
# 20251103: 254 folders with images, 20251104: 340 folders with images
HARDCODED_FOLDERS = [
    # ref folder (20251103 early)
    "20251103/20251103175941_194x150",
    # test 1 (20251103 mid-early)
    "20251103/20251103200012_194x150",
    # test 2 (20251103 mid)  
    "20251103/20251103220015_194x150",
    # test 3 (20251104 early)
    "20251104/20251104000102_194x150",
    # test 4 (20251104 mid)
    "20251104/20251104060012_194x150",
    # test 5 (20251104 late)
    "20251104/20251104120012_194x150",
]

def find_194x150_folders():
    """Return hardcoded 194x150 folders, verifying camera_1 exists."""
    results = []
    for f in HARDCODED_FOLDERS:
        p = NAS_ROOT / f
        if p.exists() and (p / "camera_1").is_dir():
            results.append(p)
    
    if len(results) < 6:
        # Fallback: scan 20251103 and 20251104 for folders with camera_1
        print(f"  Only {len(results)} hardcoded folders valid, scanning for more...")
        for date in ["20251103", "20251104"]:
            date_dir = NAS_ROOT / date
            if not date_dir.exists():
                continue
            subs = sorted([s for s in date_dir.iterdir() if s.is_dir() and '194x150' in s.name])
            # Pick evenly spaced folders
            step = max(1, len(subs) // 10)
            for i in range(0, len(subs), step):
                sub = subs[i]
                if sub not in results and (sub / "camera_1").is_dir():
                    # Check if camera_1 has files
                    cam1_files = list((sub / "camera_1").iterdir())
                    if cam1_files:
                        results.append(sub)
                        print(f"    Found: {sub.name}")
                        if len(results) >= 6:
                            break
            if len(results) >= 6:
                break
    
    return results


def extract_tiles(img_gray, tile_indices):
    """Extract specific tiles from image (after trimming head/tail rows)."""
    # Apply trim
    trimmed = img_gray[TRIM_HEAD:IMAGE_H - TRIM_TAIL, :]
    th, tw = trimmed.shape
    cols = tw // TILE_SIZE
    tiles = []
    for idx in tile_indices:
        r, c = divmod(idx, cols)
        y, x = r * TILE_SIZE, c * TILE_SIZE
        if y + TILE_SIZE <= th and x + TILE_SIZE <= tw:
            tile = trimmed[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape == (TILE_SIZE, TILE_SIZE):
                tiles.append(tile)
    return tiles


def get_valid_tile_indices():
    """Return valid tile indices after trimming."""
    th = IMAGE_H - TRIM_HEAD - TRIM_TAIL  # 1000
    tw = IMAGE_W  # 1920
    rows = th // TILE_SIZE  # 3
    cols = tw // TILE_SIZE  # 7
    total = rows * cols  # 21
    # Pick 3 center-ish tiles spread across the image
    mid_row = rows // 2
    indices = [mid_row * cols + 1, mid_row * cols + 3, mid_row * cols + 5]
    return [i for i in indices if i < total]


# ── CNN feature extraction ──
def setup_cnn():
    import torch
    import torchvision.models as models
    model = models.wide_resnet50_2(weights='IMAGENET1K_V1').eval().cuda()
    features = {}
    def hook_layer2(m, inp, out):
        features['layer2'] = out
    def hook_layer3(m, inp, out):
        features['layer3'] = out
    model.layer2.register_forward_hook(hook_layer2)
    model.layer3.register_forward_hook(hook_layer3)
    return model, features

def extract_cnn_features(model, features_dict, tiles_gray, batch_size=4):
    import torch
    import torch.nn.functional as F
    import cv2
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
    
    all_avg = []
    all_spatial = []
    
    for i in range(0, len(tiles_gray), batch_size):
        batch_tiles = tiles_gray[i:i+batch_size]
        tensors = []
        for t in batch_tiles:
            t_resized = cv2.resize(t, (224, 224))
            t3 = np.stack([t_resized]*3, axis=0).astype(np.float32) / 255.0
            tensors.append(t3)
        batch = torch.tensor(np.array(tensors)).cuda()
        batch = (batch - mean) / std
        
        with torch.no_grad():
            _ = model(batch)
            l2 = features_dict['layer2']
            l3 = features_dict['layer3']
            l3_up = F.interpolate(l3, size=l2.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([l2, l3_up], dim=1)
            avg_feat = combined.mean(dim=(2,3))
            all_avg.append(avg_feat.cpu().numpy())
            all_spatial.append(combined.cpu().numpy())
    
    return np.concatenate(all_avg, axis=0), np.concatenate(all_spatial, axis=0)

_bank_gpu = None

def _get_bank_gpu(bank):
    global _bank_gpu
    import torch
    if _bank_gpu is None:
        _bank_gpu = torch.from_numpy(bank).cuda()
    return _bank_gpu

def knn_score(feat, bank, k=3):
    import torch
    bank_g = _get_bank_gpu(bank)
    feat_g = torch.from_numpy(feat).cuda().unsqueeze(0)
    dists = torch.cdist(feat_g, bank_g).squeeze(0)
    topk = torch.topk(dists, k, largest=False).values
    return topk.mean().item()

def hm_topk3_score(spatial, bank, k=3):
    import torch
    bank_g = _get_bank_gpu(bank)
    C, H, W = spatial.shape
    positions = torch.from_numpy(spatial.reshape(C, -1).T).cuda()
    dists = torch.cdist(positions, bank_g)
    topk_dists = torch.topk(dists, k, dim=1, largest=False).values
    per_pos = topk_dists.mean(dim=1)
    top3 = torch.topk(per_pos, 3, largest=True).values
    return top3.mean().item()

# ── Non-CNN methods ──
def compute_fft_score(tile):
    mag = np.abs(np.fft.fft2(tile.astype(np.float32)))
    mag = np.fft.fftshift(mag)
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = min(h, w) // 4
    mask = np.zeros_like(mag, dtype=bool)
    Y, X = np.ogrid[:h, :w]
    mask[(Y-cy)**2 + (X-cx)**2 <= r**2] = True
    high_freq = mag[~mask].mean()
    return high_freq

def compute_dct_score(tile):
    import cv2
    dct = cv2.dct(tile.astype(np.float32))
    h, w = dct.shape
    hf = np.abs(dct[h//2:, w//2:]).mean()
    return hf

def compute_lbp_score(tile):
    h, w = tile.shape
    center = tile[1:-1, 1:-1].astype(np.float32)
    patterns = []
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        neighbor = tile[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.float32)
        patterns.append((neighbor >= center).astype(np.float32))
    lbp = sum(p * (2**i) for i, p in enumerate(patterns))
    return float(np.std(lbp))

def compute_ssim_score(tile, ref_tile):
    import cv2
    C1, C2 = 6.5025, 58.5225
    t = tile.astype(np.float64)
    r = ref_tile.astype(np.float64)
    mu_t = cv2.GaussianBlur(t, (11,11), 1.5)
    mu_r = cv2.GaussianBlur(r, (11,11), 1.5)
    mu_t2 = mu_t ** 2
    mu_r2 = mu_r ** 2
    mu_tr = mu_t * mu_r
    sigma_t2 = cv2.GaussianBlur(t**2, (11,11), 1.5) - mu_t2
    sigma_r2 = cv2.GaussianBlur(r**2, (11,11), 1.5) - mu_r2
    sigma_tr = cv2.GaussianBlur(t*r, (11,11), 1.5) - mu_tr
    ssim_map = ((2*mu_tr + C1)*(2*sigma_tr + C2)) / ((mu_t2 + mu_r2 + C1)*(sigma_t2 + sigma_r2 + C2))
    return 1.0 - float(np.mean(ssim_map))

def compute_gradient_score(tile):
    import cv2
    gx = cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))

def compute_multireso_score(tile):
    import cv2
    scores = []
    t = tile.astype(np.float32)
    for scale in [1.0, 0.5, 0.25]:
        if scale < 1.0:
            sz = max(1, int(TILE_SIZE * scale))
            t_s = cv2.resize(t, (sz, sz))
        else:
            t_s = t
        scores.append(float(np.std(t_s)))
    return float(np.mean(scores))


def main():
    import cv2
    print("=" * 60)
    print("194x150 Ens_MAX Generalization Test — Final")
    print("=" * 60)
    
    # Step 1: Load memory bank
    print("\n[1] Loading memory bank...")
    if not MEMORY_BANK_PATH.exists():
        print(f"  ERROR: {MEMORY_BANK_PATH} not found!")
        sys.exit(1)
    memory_bank = np.load(str(MEMORY_BANK_PATH))
    print(f"  Memory bank shape: {memory_bank.shape}")
    
    # Step 2: Find 194x150 folders
    print("\n[2] Finding 194x150 folders...")
    all_folders = find_194x150_folders()
    print(f"  Total 194x150 folders with camera_1: {len(all_folders)}")
    
    if len(all_folders) < 6:
        print(f"  ERROR: Need at least 6 folders, found {len(all_folders)}")
        sys.exit(1)
    
    selected = all_folders[:6]
    
    ref_folder = selected[0]
    test_folders = selected[1:6]
    
    print(f"\n  Reference folder: {ref_folder.parent.name}/{ref_folder.name}")
    print(f"  Test folders:")
    for i, f in enumerate(test_folders):
        print(f"    [{i+1}] {f.parent.name}/{f.name}")
    
    # Step 3: Setup CNN
    print("\n[3] Setting up CNN model...")
    model, features_dict = setup_cnn()
    print("  WideResNet50 loaded on GPU")
    
    # Step 4: Build non-CNN reference stats from ref_folder
    print("\n[4] Building reference statistics from ref folder...")
    ref_cam1 = ref_folder / "camera_1"
    ref_imgs = sorted([f for f in ref_cam1.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.bmp')])[:5]
    
    test_tile_indices = get_valid_tile_indices()
    print(f"  Tile indices: {test_tile_indices}")
    
    ref_stats = {"fft": [], "dct": [], "lbp": [], "gradient": [], "multireso": []}
    ref_tiles_for_ssim = []
    
    for img_path in ref_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        
        tiles = extract_tiles(img, test_tile_indices)
        for tile in tiles:
            ref_stats["fft"].append(compute_fft_score(tile))
            ref_stats["dct"].append(compute_dct_score(tile))
            ref_stats["lbp"].append(compute_lbp_score(tile))
            ref_stats["gradient"].append(compute_gradient_score(tile))
            ref_stats["multireso"].append(compute_multireso_score(tile))
            ref_tiles_for_ssim.append(tile.copy())
    
    ref_normal = {}
    for method in ["fft", "dct", "lbp", "gradient", "multireso"]:
        vals = ref_stats[method]
        ref_normal[method] = {"mean": np.mean(vals), "std": max(np.std(vals), 1e-6)}
        print(f"  {method}: mean={ref_normal[method]['mean']:.4f}, std={ref_normal[method]['std']:.4f}")
    
    # SSIM ref tiles
    ref_tile_avg = {}
    n_imgs = len(ref_imgs)
    for i, idx in enumerate(test_tile_indices):
        tiles_at_pos = [ref_tiles_for_ssim[j*len(test_tile_indices)+i] 
                       for j in range(n_imgs) if j*len(test_tile_indices)+i < len(ref_tiles_for_ssim)]
        if tiles_at_pos:
            ref_tile_avg[idx] = np.mean(tiles_at_pos, axis=0).astype(np.uint8)
    
    ssim_vals = []
    for img_path in ref_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        tiles = extract_tiles(img, test_tile_indices)
        for tile, idx in zip(tiles, test_tile_indices):
            if idx in ref_tile_avg:
                ssim_vals.append(compute_ssim_score(tile, ref_tile_avg[idx]))
    ref_normal["ssim"] = {"mean": np.mean(ssim_vals), "std": max(np.std(ssim_vals), 1e-6)}
    print(f"  ssim: mean={ref_normal['ssim']['mean']:.6f}, std={ref_normal['ssim']['std']:.6f}")
    
    # CNN ref stats
    print("\n  Computing CNN reference scores...")
    cnn_ref_original = []
    cnn_ref_hmtopk3 = []
    for img_path in ref_imgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        tiles = extract_tiles(img, test_tile_indices)
        if not tiles: continue
        avg_feats, spatial_feats = extract_cnn_features(model, features_dict, tiles)
        for j in range(len(tiles)):
            cnn_ref_original.append(knn_score(avg_feats[j], memory_bank))
            cnn_ref_hmtopk3.append(hm_topk3_score(spatial_feats[j], memory_bank))
    
    ref_normal["original"] = {"mean": np.mean(cnn_ref_original), "std": max(np.std(cnn_ref_original), 1e-6)}
    ref_normal["hm_topk3"] = {"mean": np.mean(cnn_ref_hmtopk3), "std": max(np.std(cnn_ref_hmtopk3), 1e-6)}
    print(f"  original: mean={ref_normal['original']['mean']:.4f}, std={ref_normal['original']['std']:.4f}")
    print(f"  hm_topk3: mean={ref_normal['hm_topk3']['mean']:.4f}, std={ref_normal['hm_topk3']['std']:.4f}")
    
    # Step 5: Test on 5 folders
    print("\n[5] Running tests on 5 folders...")
    METHOD_NAMES = ["original", "hm_topk3", "fft", "dct", "lbp", "ssim", "gradient", "multireso", "ens_max"]
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    folder_names = []
    
    rng = np.random.RandomState(SEED)
    
    for fi, folder in enumerate(test_folders):
        fname = f"{folder.parent.name}/{folder.name}"
        folder_names.append(fname)
        print(f"\n  --- Folder [{fi+1}/5]: {fname} ---")
        
        cam1 = folder / "camera_1"
        imgs = sorted([f for f in cam1.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.bmp')])
        if not imgs:
            print(f"    SKIP: no images")
            continue
        
        test_img_path = imgs[len(imgs)//2]
        print(f"    Image: {test_img_path.name}")
        
        img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    SKIP: cannot read")
            continue
        if img.shape != (IMAGE_H, IMAGE_W):
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        
        clean_tiles = extract_tiles(img, test_tile_indices)
        if len(clean_tiles) < 3:
            print(f"    SKIP: not enough tiles ({len(clean_tiles)})")
            continue
        
        for di, defect_name in enumerate(DEFECT_TYPES):
            if di % 3 == 0:
                print(f"    defect {di+1}/{len(DEFECT_TYPES)}: {defect_name}...", flush=True)
            for ti, (tile_idx, clean_tile) in enumerate(zip(test_tile_indices, clean_tiles)):
                defect_tile = inject_defect(clean_tile, defect_name, rng)
                
                scores = {}
                scores["fft"] = compute_fft_score(defect_tile)
                scores["dct"] = compute_dct_score(defect_tile)
                scores["lbp"] = compute_lbp_score(defect_tile)
                scores["gradient"] = compute_gradient_score(defect_tile)
                scores["multireso"] = compute_multireso_score(defect_tile)
                scores["ssim"] = compute_ssim_score(defect_tile, ref_tile_avg.get(tile_idx, clean_tile))
                
                avg_feat, spatial_feat = extract_cnn_features(model, features_dict, [defect_tile])
                scores["original"] = knn_score(avg_feat[0], memory_bank)
                scores["hm_topk3"] = hm_topk3_score(spatial_feat[0], memory_bank)
                
                ratios = {}
                z_scores = {}
                for method in METHOD_NAMES[:-1]:
                    mu = ref_normal[method]["mean"]
                    sigma = ref_normal[method]["std"]
                    ratio = scores[method] / mu if mu > 0 else 0
                    ratios[method] = ratio
                    z_scores[method] = (scores[method] - mu) / sigma
                
                for method in METHOD_NAMES[:-1]:
                    detected = ratios[method] > DETECTION_THRESHOLD
                    results[fi][defect_name][method].append(detected)
                
                ens_detected = any(ratios[m] > DETECTION_THRESHOLD for m in METHOD_NAMES[:-1])
                results[fi][defect_name]["ens_max"].append(ens_detected)
        
        total = sum(len(results[fi][d]["ens_max"]) for d in DEFECT_TYPES)
        detected_ens = sum(sum(results[fi][d]["ens_max"]) for d in DEFECT_TYPES)
        detected_orig = sum(sum(results[fi][d]["original"]) for d in DEFECT_TYPES)
        print(f"    Results: Original={detected_orig}/{total}, Ens_MAX={detected_ens}/{total}")
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Folder':<40} | {'Original':>8} | {'HM_TopK3':>8} | {'Ens_MAX':>8}")
    print("-" * 75)
    
    folder_summary = []
    for fi in range(5):
        if fi not in results:
            continue
        total = sum(len(results[fi][d]["original"]) for d in DEFECT_TYPES)
        rates = {}
        for method in METHOD_NAMES:
            det = sum(sum(results[fi][d][method]) for d in DEFECT_TYPES)
            rates[method] = det / total * 100 if total > 0 else 0
        
        short_name = folder_names[fi].split("/")[-1][:35]
        print(f"  {short_name:<38} | {rates['original']:>7.1f}% | {rates['hm_topk3']:>7.1f}% | {rates['ens_max']:>7.1f}%")
        folder_summary.append((folder_names[fi], rates))
    
    print(f"\n{'Defect Type':<20} | {'Original':>8} | {'HM_TopK3':>8} | {'FFT':>6} | {'DCT':>6} | {'LBP':>6} | {'SSIM':>6} | {'Grad':>6} | {'MRes':>6} | {'Ens_MAX':>8}")
    print("-" * 120)
    
    defect_summary = []
    for defect in DEFECT_TYPES:
        row = {}
        for method in METHOD_NAMES:
            total = sum(len(results[fi][defect][method]) for fi in range(5) if fi in results)
            det = sum(sum(results[fi][defect][method]) for fi in range(5) if fi in results)
            row[method] = det / total * 100 if total > 0 else 0
        print(f"  {defect:<18} | {row['original']:>7.1f}% | {row['hm_topk3']:>7.1f}% | {row['fft']:>5.1f}% | {row['dct']:>5.1f}% | {row['lbp']:>5.1f}% | {row['ssim']:>5.1f}% | {row['gradient']:>5.1f}% | {row['multireso']:>5.1f}% | {row['ens_max']:>7.1f}%")
        defect_summary.append((defect, row))
    
    print("\n--- Overall ---")
    overall = {}
    for method in METHOD_NAMES:
        total = sum(len(results[fi][d][method]) for fi in range(5) for d in DEFECT_TYPES if fi in results)
        det = sum(sum(results[fi][d][method]) for fi in range(5) for d in DEFECT_TYPES if fi in results)
        overall[method] = det / total * 100 if total > 0 else 0
        print(f"  {method:<15}: {det}/{total} = {overall[method]:.1f}%")
    
    # Save results
    import json
    result_data = {
        "experiment": "194x150 Ens_MAX Generalization Test",
        "memory_bank": str(MEMORY_BANK_PATH),
        "memory_bank_shape": list(memory_bank.shape),
        "ref_folder": str(ref_folder),
        "test_folders": [str(f) for f in test_folders],
        "folder_summary": [(n, r) for n, r in folder_summary],
        "defect_summary": [(d, r) for d, r in defect_summary],
        "overall": overall,
    }
    out_dir = OUTPUT_DIR / "194x150"
    out_path = out_dir / "generalization_test_final.json"
    with open(str(out_path), 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    # Telegram format
    print("\n\n=== TELEGRAM FORMAT ===")
    msg = "📊 194x150 Ens_MAX 일반화 검증 결과\n\n"
    msg += f"학습된 memory bank (group_1, shape={memory_bank.shape})\n"
    msg += f"다른 생산 폴더 5개에서 테스트\n"
    msg += f"5폴더 × 27케이스 = 135 총 케이스\n\n"
    
    msg += "📋 폴더별 검출률\n"
    for name, rates in folder_summary:
        short = name.split("/")[-1][:25]
        msg += f"  {short}: Orig {rates['original']:.0f}% | HM {rates['hm_topk3']:.0f}% | Ens {rates['ens_max']:.0f}%\n"
    
    msg += f"\n🎯 Ens_MAX 결함유형별 검출률\n"
    for defect, row in defect_summary:
        msg += f"  {defect}: {row['ens_max']:.0f}%\n"
    
    msg += f"\n📊 전체 방법 비교\n"
    for method in METHOD_NAMES:
        msg += f"  {method}: {overall[method]:.1f}%\n"
    
    best = max(overall.items(), key=lambda x: x[1])
    msg += f"\n✅ 결론: {best[0]}이 {best[1]:.1f}%로 최고 성능"
    if overall['ens_max'] >= overall['original']:
        msg += f"\nEns_MAX가 Original 대비 +{overall['ens_max']-overall['original']:.1f}%p 향상"
    
    print(msg)
    
    with open(str(out_dir / "telegram_msg.txt"), 'w') as f:
        f.write(msg)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
