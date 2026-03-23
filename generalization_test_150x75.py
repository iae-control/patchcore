#!/usr/bin/env python3
"""150x75 Ens_MAX Generalization Test — group_1 (camera_1 + camera_10)
Uses pre-trained memory_bank.npy to test on different production lots.
"""
import os, sys, time, json, random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ──
NAS_ROOT = Path(os.path.expanduser("~/nas_storage"))
OUTPUT_DIR = Path(os.path.expanduser("~/patchcore/output"))
MEMORY_BANK_PATH = OUTPUT_DIR / "150x75" / "group_1" / "memory_bank.npy"
TILE_SIZE = 256
IMAGE_W, IMAGE_H = 1920, 1200
TRIM_HEAD, TRIM_TAIL = 100, 100
DETECTION_THRESHOLD = 1.1
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Tile grid ──
TILES_X = IMAGE_W // TILE_SIZE  # 7
TILES_Y = IMAGE_H // TILE_SIZE  # 4
TOTAL_TILES = TILES_X * TILES_Y  # 28

# ── Synthetic defect definitions ──
DEFECT_TYPES = {
    # Spots
    "spot":          {"type": "circle", "radius": 5,  "diff": 30, "blur": 0},
    "big_spot":      {"type": "circle", "radius": 10, "diff": 30, "blur": 0},
    "stain":         {"type": "circle", "radius": 15, "diff": 25, "blur": 7},
    # Linear
    "scratch":       {"type": "line",   "width": 1,  "diff": 30, "zigzag": False, "count": 1},
    "crack":         {"type": "line",   "width": 1,  "diff": 25, "zigzag": True,  "count": 1},
    "multi_scratch": {"type": "line",   "width": 1,  "diff": 25, "zigzag": False, "count": 3},
    # Subtle
    "faint_spot":    {"type": "circle", "radius": 5,  "diff": 10, "blur": 0},
    "thin_scratch":  {"type": "line",   "width": 1,  "diff": 15, "zigzag": False, "count": 1},
    "gradient_stain":{"type": "gradient", "diff": 16},
}

def inject_defect(tile, defect_name, rng):
    """Inject synthetic defect into a 256x256 grayscale tile. Returns modified tile."""
    import cv2
    tile = tile.copy()
    h, w = tile.shape[:2]
    spec = DEFECT_TYPES[defect_name]
    
    cx, cy = rng.randint(40, w-40), rng.randint(40, h-40)
    diff = spec["diff"]
    # Decide sign: darken or brighten
    sign = rng.choice([-1, 1])
    
    if spec["type"] == "circle":
        r = spec["radius"]
        blur = spec.get("blur", 0)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        if blur > 0:
            mask = cv2.GaussianBlur(mask, (blur*2+1, blur*2+1), blur)
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
                angle = rng.uniform(0, np.pi)
                length = rng.randint(40, 120)
                x1 = int(cx - length/2 * np.cos(angle))
                y1 = int(cy - length/2 * np.sin(angle))
                x2 = int(cx + length/2 * np.cos(angle))
                y2 = int(cy + length/2 * np.sin(angle))
                val = int(np.clip(tile[cy, cx] + sign*diff, 0, 255))
                cv2.line(tile, (x1, y1), (x2, y2), val, width)
            cx += rng.randint(-30, 30)
            cy += rng.randint(-30, 30)
            cx = max(20, min(w-20, cx))
            cy = max(20, min(h-20, cy))
            
    elif spec["type"] == "gradient":
        mask = np.zeros((h, w), dtype=np.float32)
        r = 40
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (r*2+1, r*2+1), r/2)
        mask = mask / (mask.max() + 1e-8)
        tile_f = tile.astype(np.float32)
        tile_f += sign * diff * mask
        tile = np.clip(tile_f, 0, 255).astype(np.uint8)
    
    return tile


def extract_tiles(image_gray, tile_size=256):
    """Extract non-overlapping tiles from image."""
    h, w = image_gray.shape[:2]
    tiles = []
    positions = []
    for ty in range(h // tile_size):
        for tx in range(w // tile_size):
            y0, x0 = ty * tile_size, tx * tile_size
            tile = image_gray[y0:y0+tile_size, x0:x0+tile_size]
            tiles.append(tile)
            positions.append((ty, tx))
    return tiles, positions


# ══════════════════════════════════════════════════════════
# Scoring Methods
# ══════════════════════════════════════════════════════════

class CNNFeatureExtractor:
    """WideResNet50 layer2+layer3 feature extractor."""
    def __init__(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.wide_resnet50_2(pretrained=True).eval().to(self.device)
        self.features = {}
        
        def hook_fn(name):
            def hook(module, inp, out):
                self.features[name] = out
            return hook
        
        self.model.layer2.register_forward_hook(hook_fn('layer2'))
        self.model.layer3.register_forward_hook(hook_fn('layer3'))
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, tiles_gray):
        """Extract features from list of grayscale tiles. Returns (N, 1536) array."""
        import torch
        import torch.nn.functional as F
        
        all_feats = []
        batch_size = 4
        
        for i in range(0, len(tiles_gray), batch_size):
            batch_tiles = tiles_gray[i:i+batch_size]
            tensors = []
            for t in batch_tiles:
                # grayscale → 3ch
                t3 = np.stack([t, t, t], axis=-1) if t.ndim == 2 else t
                tensors.append(self.transform(t3))
            
            batch = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                self.features.clear()
                _ = self.model(batch)
                
                l2 = self.features['layer2']  # (B, 512, 28, 28)
                l3 = self.features['layer3']  # (B, 1024, 14, 14)
                
                # Upsample l3 to l2 size
                l3_up = F.interpolate(l3, size=l2.shape[2:], mode='bilinear', align_corners=False)
                # Concat
                cat = torch.cat([l2, l3_up], dim=1)  # (B, 1536, 28, 28)
                # Global avg pool
                feat = F.adaptive_avg_pool2d(cat, 1).squeeze(-1).squeeze(-1)  # (B, 1536)
                
                all_feats.append(feat.cpu().numpy())
        
        return np.concatenate(all_feats, axis=0)
    
    def extract_spatial(self, tiles_gray):
        """Extract spatial features (B, 1536, 28, 28) for HM_TopK3."""
        import torch
        import torch.nn.functional as F
        
        all_feats = []
        batch_size = 4
        
        for i in range(0, len(tiles_gray), batch_size):
            batch_tiles = tiles_gray[i:i+batch_size]
            tensors = []
            for t in batch_tiles:
                t3 = np.stack([t, t, t], axis=-1) if t.ndim == 2 else t
                tensors.append(self.transform(t3))
            
            batch = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                self.features.clear()
                _ = self.model(batch)
                
                l2 = self.features['layer2']
                l3 = self.features['layer3']
                l3_up = F.interpolate(l3, size=l2.shape[2:], mode='bilinear', align_corners=False)
                cat = torch.cat([l2, l3_up], dim=1)  # (B, 1536, 28, 28)
                
                all_feats.append(cat.cpu().numpy())
        
        return np.concatenate(all_feats, axis=0)


def score_original(tile_feats, memory_bank, k=3):
    """Original PatchCore: avg_pool feature → kNN distance to memory bank."""
    from scipy.spatial.distance import cdist
    dists = cdist(tile_feats, memory_bank, metric='euclidean')
    return np.min(dists, axis=1)  # min distance to nearest neighbor


def score_hm_topk3(tile_spatial, memory_bank_spatial, k=3):
    """HM_TopK3: spatial patch-level scoring, top-3 patches. GPU-accelerated."""
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank_t = torch.from_numpy(memory_bank_spatial).float().to(device)  # (M, 1536)
    # Precompute bank norms
    bank_norm_sq = (bank_t ** 2).sum(dim=1)  # (M,)
    
    scores = []
    N = tile_spatial.shape[0]
    for i in range(N):
        patches = tile_spatial[i].reshape(1536, -1).T  # (784, 1536)
        patches_t = torch.from_numpy(patches).float().to(device)
        # Efficient distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        p_norm_sq = (patches_t ** 2).sum(dim=1, keepdim=True)  # (784, 1)
        dists_sq = p_norm_sq + bank_norm_sq.unsqueeze(0) - 2 * patches_t @ bank_t.T  # (784, M)
        dists_sq = torch.clamp(dists_sq, min=0)
        min_dists = torch.sqrt(dists_sq.min(dim=1).values)  # (784,)
        top_k = torch.topk(min_dists, k).values
        scores.append(top_k.mean().item())
        del patches_t, dists_sq, min_dists
    
    del bank_t
    torch.cuda.empty_cache()
    return np.array(scores)


def score_fft(tile_gray, ref_mean=None, ref_std=None):
    """FFT-based anomaly score."""
    spec = np.abs(np.fft.fft2(tile_gray.astype(np.float32)))
    spec = np.fft.fftshift(spec)
    score = np.mean(spec)
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


def score_dct(tile_gray, ref_mean=None, ref_std=None):
    """DCT-based anomaly score."""
    from scipy.fftpack import dct
    d = dct(dct(tile_gray.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
    # High-freq energy
    h, w = d.shape
    mask = np.ones((h, w))
    mask[:h//4, :w//4] = 0  # exclude low-freq
    score = np.mean(np.abs(d * mask))
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


def _lbp_manual(image, P=8, R=1):
    """Simple LBP implementation without skimage."""
    import cv2
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for p in range(P):
        angle = 2 * np.pi * p / P
        dy, dx = -R * np.cos(angle), R * np.sin(angle)
        # Bilinear interpolation coords
        fy, fx = int(round(dy)), int(round(dx))
        shifted = np.zeros_like(image)
        src_y0 = max(0, -fy)
        src_y1 = min(h, h - fy)
        src_x0 = max(0, -fx)
        src_x1 = min(w, w - fx)
        dst_y0 = max(0, fy)
        dst_y1 = min(h, h + fy)
        dst_x0 = max(0, fx)
        dst_x1 = min(w, w + fx)
        sy = min(src_y1 - src_y0, dst_y1 - dst_y0)
        sx = min(src_x1 - src_x0, dst_x1 - dst_x0)
        if sy > 0 and sx > 0:
            shifted[dst_y0:dst_y0+sy, dst_x0:dst_x0+sx] = image[src_y0:src_y0+sy, src_x0:src_x0+sx]
        lbp += ((shifted >= image).astype(np.uint8) << p)
    return lbp


def score_lbp(tile_gray, ref_mean=None, ref_std=None):
    """LBP-based anomaly score."""
    lbp = _lbp_manual(tile_gray, P=8, R=1)
    hist, _ = np.histogram(lbp, bins=10, density=True)
    score = np.std(hist)
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


def _ssim_manual(img1, img2, data_range=255):
    """Simple SSIM without skimage."""
    import cv2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def score_ssim(tile_gray, ref_tile, ref_mean=None, ref_std=None):
    """SSIM-based anomaly score (1 - SSIM)."""
    s = _ssim_manual(tile_gray, ref_tile, data_range=255)
    score = 1.0 - s
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


def score_gradient(tile_gray, ref_mean=None, ref_std=None):
    """Gradient (Sobel) based anomaly score."""
    import cv2
    gx = cv2.Sobel(tile_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    score = np.mean(mag)
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


def score_multireso(tile_gray, ref_mean=None, ref_std=None):
    """Multi-resolution anomaly score."""
    import cv2
    scores_mr = []
    current = tile_gray.astype(np.float32)
    for scale in [1.0, 0.5, 0.25]:
        if scale < 1.0:
            h, w = int(current.shape[0] * scale), int(current.shape[1] * scale)
            scaled = cv2.resize(tile_gray.astype(np.float32), (w, h))
        else:
            scaled = current
        # Use variance + high-freq energy
        spec = np.abs(np.fft.fft2(scaled))
        scores_mr.append(np.mean(spec))
    score = np.mean(scores_mr)
    if ref_mean is not None and ref_std is not None and ref_std > 1e-8:
        return (score - ref_mean) / ref_std
    return score


# ══════════════════════════════════════════════════════════
# Main Experiment
# ══════════════════════════════════════════════════════════

def main():
    import cv2
    
    print("=" * 70)
    print("150x75 Ens_MAX Generalization Test — group_1 (camera_1)")
    print("=" * 70)
    
    # Step 1: Load memory bank
    print("\n[1] Loading memory bank...")
    memory_bank = np.load(str(MEMORY_BANK_PATH))
    print(f"  Memory bank shape: {memory_bank.shape}")
    
    # Step 2: Find 150x75 folders by lot
    print("\n[2] Finding test folders...")
    base = NAS_ROOT / "20251105"
    all_folders = sorted([f for f in os.listdir(base) if "150x75" in f.lower()])
    print(f"  Total 150x75 folders: {len(all_folders)}")
    
    # Group by lot
    lot_folders = defaultdict(list)
    for f in all_folders:
        lev = base / f / "level2.txt"
        if lev.exists():
            for line in open(lev):
                if line.startswith("lot_no"):
                    lot = line.split(":")[-1].strip()
                    lot_folders[lot].append(f)
                    break
    
    lots = sorted(lot_folders.keys())
    print(f"  Unique lots: {len(lots)}")
    
    # Pick 5 evenly spaced lots
    step = max(1, len(lots) // 5)
    selected_lots = [lots[i * step] for i in range(5)]
    # For each lot, pick 1 folder from middle
    test_folders = []
    for lot in selected_lots:
        flist = lot_folders[lot]
        mid = len(flist) // 2
        test_folders.append(flist[mid])
    
    print(f"  Selected test folders:")
    for i, f in enumerate(test_folders):
        lot = selected_lots[i]
        print(f"    [{i+1}] {f} (lot: {lot})")
    
    # Step 3: Init CNN extractor
    print("\n[3] Initializing CNN feature extractor...")
    extractor = CNNFeatureExtractor()
    print("  Done.")
    
    # Step 4: For non-CNN methods, build normal statistics from OTHER lots
    print("\n[4] Building normal statistics for non-CNN methods...")
    # Pick 3 lots NOT in selected_lots for reference
    ref_lots = [l for l in lots if l not in selected_lots][:5]
    ref_folders_list = []
    for lot in ref_lots:
        flist = lot_folders[lot]
        ref_folders_list.append(flist[len(flist)//2])
    
    # Collect normal tile stats
    method_names_noncnn = ["FFT", "DCT", "LBP", "Gradient", "MultiReso"]
    # For SSIM we need a reference tile per position
    
    # Process reference tiles
    print("  Loading reference images...")
    ref_tiles_by_pos = defaultdict(list)  # position → list of tiles
    ref_scores = {m: [] for m in method_names_noncnn}
    
    for rf in ref_folders_list[:3]:
        img_dir = base / rf / "camera_1"
        if not img_dir.exists():
            continue
        imgs = sorted(os.listdir(img_dir))
        if len(imgs) < TRIM_HEAD + TRIM_TAIL + 10:
            continue
        # Pick 3 images from middle
        mid_imgs = imgs[TRIM_HEAD:len(imgs)-TRIM_TAIL]
        sample_imgs = [mid_imgs[len(mid_imgs)//4], mid_imgs[len(mid_imgs)//2], mid_imgs[3*len(mid_imgs)//4]]
        
        for img_name in sample_imgs:
            img = cv2.imread(str(img_dir / img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            tiles, positions = extract_tiles(img)
            for tile, pos in zip(tiles, positions):
                ref_tiles_by_pos[pos].append(tile)
                ref_scores["FFT"].append(score_fft(tile))
                ref_scores["DCT"].append(score_dct(tile))
                ref_scores["LBP"].append(score_lbp(tile))
                ref_scores["Gradient"].append(score_gradient(tile))
                ref_scores["MultiReso"].append(score_multireso(tile))
    
    # Compute mean/std for each method
    normal_stats = {}
    for m in method_names_noncnn:
        vals = np.array(ref_scores[m])
        normal_stats[m] = {"mean": np.mean(vals), "std": np.std(vals)}
        print(f"  {m}: mean={normal_stats[m]['mean']:.4f}, std={normal_stats[m]['std']:.4f}")
    
    # SSIM reference: average tile per position
    ssim_ref_tiles = {}
    for pos, tiles_list in ref_tiles_by_pos.items():
        ssim_ref_tiles[pos] = np.mean(tiles_list, axis=0).astype(np.uint8)
    
    # SSIM normal stats
    ssim_normal_scores = []
    for pos, tiles_list in ref_tiles_by_pos.items():
        ref = ssim_ref_tiles[pos]
        for t in tiles_list:
            ssim_normal_scores.append(score_ssim(t, ref))
    normal_stats["SSIM"] = {"mean": np.mean(ssim_normal_scores), "std": np.std(ssim_normal_scores)}
    print(f"  SSIM: mean={normal_stats['SSIM']['mean']:.4f}, std={normal_stats['SSIM']['std']:.4f}")
    
    # Step 5: Run tests
    print("\n[5] Running defect detection tests...")
    all_methods = ["Original", "HM_TopK3", "FFT", "DCT", "LBP", "SSIM", "Gradient", "MultiReso", "Ens_MAX"]
    
    # Results storage
    # results[folder_idx][defect_type][method] = list of detected (bool)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    rng = np.random.RandomState(SEED)
    tile_positions_to_test = [
        (1, 2),  # upper-middle
        (2, 4),  # center-right
        (3, 1),  # lower-left
    ]
    
    for fi, folder_name in enumerate(test_folders):
        print(f"\n  --- Folder [{fi+1}/5]: {folder_name} ---")
        img_dir = base / folder_name / "camera_1"
        if not img_dir.exists():
            print(f"    SKIP: no camera_1")
            continue
        
        imgs = sorted(os.listdir(img_dir))
        total = len(imgs)
        if total < TRIM_HEAD + TRIM_TAIL + 10:
            print(f"    SKIP: too few images ({total})")
            continue
        
        mid_imgs = imgs[TRIM_HEAD:total-TRIM_TAIL]
        # Pick 1 test image from the middle
        test_img_name = mid_imgs[len(mid_imgs)//2]
        test_img = cv2.imread(str(img_dir / test_img_name), cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            print(f"    SKIP: cannot read {test_img_name}")
            continue
        
        print(f"    Image: {test_img_name}, shape: {test_img.shape}")
        
        # Extract all tiles
        all_tiles, all_positions = extract_tiles(test_img)
        
        # Compute normal scores ONLY for the 3 test positions
        test_tile_indices = [ty * TILES_X + tx for ty, tx in tile_positions_to_test if ty * TILES_X + tx < len(all_tiles)]
        test_tiles_normal = [all_tiles[idx] for idx in test_tile_indices]
        
        print("    Computing normal tile features (3 positions)...")
        normal_feats = extractor.extract(test_tiles_normal)
        normal_spatial = extractor.extract_spatial(test_tiles_normal)
        
        normal_original_scores = score_original(normal_feats, memory_bank)
        normal_hm_scores = score_hm_topk3(normal_spatial, memory_bank)
        
        # Per-position normal scores
        pos_normal = {}
        for pi, idx in enumerate(test_tile_indices):
            tile = all_tiles[idx]
            pos = all_positions[idx]
            ref_t = ssim_ref_tiles.get(pos, tile)
            pos_normal[idx] = {
                "Original": normal_original_scores[pi],
                "HM_TopK3": normal_hm_scores[pi],
                "FFT": score_fft(tile),
                "DCT": score_dct(tile),
                "LBP": score_lbp(tile),
                "SSIM": score_ssim(tile, ref_t),
                "Gradient": score_gradient(tile),
                "MultiReso": score_multireso(tile),
            }
        
        # Average for reporting
        avg_normal = {}
        for m in ["Original", "HM_TopK3", "FFT", "DCT", "LBP", "SSIM", "Gradient", "MultiReso"]:
            avg_normal[m] = np.mean([pos_normal[idx][m] for idx in test_tile_indices])
        
        print(f"    Normal scores: Orig={avg_normal['Original']:.4f}, HM={avg_normal['HM_TopK3']:.4f}")
        
        # Test each defect type × 3 positions
        for defect_name in DEFECT_TYPES:
            for pi, (ty, tx) in enumerate(tile_positions_to_test):
                tile_idx = ty * TILES_X + tx
                if tile_idx >= len(all_tiles) or tile_idx not in pos_normal:
                    continue
                
                # Create defective tile
                original_tile = all_tiles[tile_idx].copy()
                defect_tile = inject_defect(original_tile, defect_name, rng)
                
                # Score defective tile - CNN
                def_feat = extractor.extract([defect_tile])
                def_spatial = extractor.extract_spatial([defect_tile])
                
                def_original = score_original(def_feat, memory_bank)[0]
                def_hm = score_hm_topk3(def_spatial, memory_bank)[0]
                
                # Non-CNN
                pos = all_positions[tile_idx]
                ref_t = ssim_ref_tiles.get(pos, original_tile)
                
                def_fft = score_fft(defect_tile)
                def_dct = score_dct(defect_tile)
                def_lbp = score_lbp(defect_tile)
                def_ssim = score_ssim(defect_tile, ref_t)
                def_grad = score_gradient(defect_tile)
                def_mr = score_multireso(defect_tile)
                
                # Use per-position normal score
                n = pos_normal[tile_idx]
                method_scores = {
                    "Original": def_original / (n["Original"] + 1e-8),
                    "HM_TopK3": def_hm / (n["HM_TopK3"] + 1e-8),
                    "FFT": def_fft / (n["FFT"] + 1e-8),
                    "DCT": def_dct / (n["DCT"] + 1e-8),
                    "LBP": def_lbp / (n["LBP"] + 1e-8),
                    "SSIM": def_ssim / (n["SSIM"] + 1e-8),
                    "Gradient": def_grad / (n["Gradient"] + 1e-8),
                    "MultiReso": def_mr / (n["MultiReso"] + 1e-8),
                }
                
                for m in ["Original", "HM_TopK3", "FFT", "DCT", "LBP", "SSIM", "Gradient", "MultiReso"]:
                    detected = method_scores[m] > DETECTION_THRESHOLD
                    results[fi][defect_name][m].append(detected)
                
                # Ens_MAX: detected if ANY method detects
                ens_detected = any(method_scores[m] > DETECTION_THRESHOLD 
                                  for m in ["Original", "HM_TopK3", "FFT", "DCT", "LBP", "SSIM", "Gradient", "MultiReso"])
                results[fi][defect_name]["Ens_MAX"].append(ens_detected)
        
        print(f"    Done. Tested {len(DEFECT_TYPES)} defect types × 3 positions = {len(DEFECT_TYPES)*3} cases")
    
    # Step 6: Compile results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Per-folder detection rates for Original, HM_TopK3, Ens_MAX
    folder_rates = {}
    for fi in range(5):
        folder_rates[fi] = {}
        for m in all_methods:
            all_detected = []
            for d in DEFECT_TYPES:
                all_detected.extend(results[fi][d][m])
            if all_detected:
                folder_rates[fi][m] = sum(all_detected) / len(all_detected) * 100
            else:
                folder_rates[fi][m] = 0
    
    # Print folder summary
    print(f"\n{'폴더(lot)':<25} | {'Original':>8} | {'HM_TopK3':>8} | {'Ens_MAX':>8}")
    print("-" * 60)
    for fi in range(5):
        lot = selected_lots[fi] if fi < len(selected_lots) else "?"
        o = folder_rates[fi].get("Original", 0)
        h = folder_rates[fi].get("HM_TopK3", 0)
        e = folder_rates[fi].get("Ens_MAX", 0)
        print(f"{lot:<25} | {o:>7.1f}% | {h:>7.1f}% | {e:>7.1f}%")
    
    # Per-defect Ens_MAX rates
    print(f"\n{'결함유형':<16}", end="")
    for fi in range(5):
        print(f" | {'폴더'+str(fi+1):>6}", end="")
    print(f" | {'평균':>6}")
    print("-" * 70)
    
    defect_avg = {}
    for d in DEFECT_TYPES:
        print(f"{d:<16}", end="")
        rates = []
        for fi in range(5):
            detected = results[fi][d]["Ens_MAX"]
            rate = sum(detected) / max(len(detected), 1) * 100
            rates.append(rate)
            print(f" | {rate:>5.0f}%", end="")
        avg = np.mean(rates) if rates else 0
        defect_avg[d] = avg
        print(f" | {avg:>5.1f}%")
    
    # Per-method overall rates
    print(f"\n{'방법':<15}", end="")
    for fi in range(5):
        print(f" | {'폴더'+str(fi+1):>6}", end="")
    print(f" | {'평균':>6}")
    print("-" * 70)
    
    method_overall = {}
    for m in all_methods:
        print(f"{m:<15}", end="")
        rates = []
        for fi in range(5):
            r = folder_rates[fi].get(m, 0)
            rates.append(r)
            print(f" | {r:>5.1f}%", end="")
        avg = np.mean(rates)
        method_overall[m] = avg
        print(f" | {avg:>5.1f}%")
    
    # Save JSON
    output = {
        "experiment": "150x75 Ens_MAX Generalization Test",
        "group": "group_1 (camera_1, flange_top_inner)",
        "memory_bank_shape": list(memory_bank.shape),
        "test_folders": test_folders,
        "selected_lots": selected_lots,
        "folder_detection_rates": {test_folders[fi]: folder_rates[fi] for fi in range(5)},
        "defect_avg_ens_max": defect_avg,
        "method_overall": method_overall,
        "total_cases": len(DEFECT_TYPES) * 3 * 5,
    }
    
    out_path = OUTPUT_DIR / "150x75" / "generalization_test_results.json"
    with open(str(out_path), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    
    # Print telegram-ready summary
    print("\n\n=== TELEGRAM REPORT ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
