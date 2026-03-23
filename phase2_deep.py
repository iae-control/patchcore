#!/usr/bin/env python3
"""
Phase 2: Deep Learning approaches to break PatchCore detection limits.
Tests 6 methods × 10 subtle defect types × 3 intensity levels = 180+ experiments.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
# EfficientNet skipped - no internet to download weights
# from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from src.patchcore import PatchCoreModel
from src.config import TILE_SIZE, CAMERA_GROUPS, USE_DATA_PARALLEL

SPEC = '150x75'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser('~/patchcore/phase2_results')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ============================================================
# Subtle Defect Generation (10 types × 3 levels)
# ============================================================
def make_defect(tile, dtype, level):
    """Apply subtle defect to tile. level: 1(극미세), 2(미세), 3(약한)"""
    h, w = tile.shape[:2]
    d = tile.copy().astype(np.float32)
    cx, cy = w // 2, h // 2
    
    intensity_map = {1: 0.5, 2: 0.75, 3: 1.0}
    s = intensity_map[level]
    
    if dtype == 'hair_thin_scratch':
        bright = int(10 + 5 * s)
        x1, y1 = int(w * 0.2), int(h * 0.3)
        x2, y2 = int(w * 0.8), int(h * 0.7)
        val = int(np.clip(d[y1,x1,0]-bright, 0, 255))
        cv2.line(d, (x1,y1), (x2,y2), (val,val,val), 1)
        
    elif dtype == 'faint_spot':
        r = int(3 + 2 * s)
        bright = int(8 + 4 * s)
        val = int(np.clip(d[cy,cx,0]-bright, 0, 255))
        cv2.circle(d, (cx, cy), r, (val,val,val), -1)
        
    elif dtype == 'micro_crack':
        length = int(30 + 20 * s)
        bright = int(8 + 7 * s)
        pts = []
        x, y = int(w*0.3), int(h*0.5)
        for i in range(length):
            pts.append((x, y))
            x += 1
            y += np.random.choice([-1, 0, 1])
            y = int(np.clip(y, 0, h-1))
            x = int(np.clip(x, 0, w-1))
        for i in range(len(pts)-1):
            val = int(np.clip(d[pts[i][1],pts[i][0],0]-bright, 0, 255))
            cv2.line(d, pts[i], pts[i+1], (val,val,val), 1)
            
    elif dtype == 'gradient_stain':
        r = int(15 + 10 * s)
        bright = int(10 + 8 * s)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = gaussian_filter(mask, sigma=r*0.5)
        mask = mask / (mask.max() + 1e-8) * bright
        for c in range(3):
            d[:,:,c] -= mask
            
    elif dtype == 'surface_roughness':
        sigma = 5 + 5 * s
        noise = np.random.normal(0, sigma, (h, w, 3)).astype(np.float32)
        d += noise
        
    elif dtype == 'slight_discoloration':
        shift = int(5 + 3 * s)
        d += shift
        
    elif dtype == 'pinhole':
        r = int(1 + 1 * s)
        bright = int(20 + 15 * s)
        val = int(np.clip(d[cy,cx,0]-bright, 0, 255))
        cv2.circle(d, (cx, cy), r, (val,val,val), -1)
        
    elif dtype == 'shallow_dent':
        r = int(15 + 10 * s)
        bright = int(8 + 7 * s)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
        mask = np.clip(1.0 - dist / r, 0, 1) * bright
        for c in range(3):
            d[:,:,c] -= mask
            
    elif dtype == 'oxide_scale':
        bright = int(8 + 7 * s)
        n_pts = int(15 + 10 * s)
        pts = []
        x, y = int(w*0.3), int(h*0.4)
        for i in range(n_pts):
            pts.append([x, y])
            x += np.random.randint(1, 4)
            y += np.random.randint(-3, 4)
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)
        pts = np.array(pts, dtype=np.int32)
        val = int(np.clip(d[pts[0,1],pts[0,0],0]-bright, 0, 255))
        cv2.polylines(d, [pts], False, (val,val,val), 1)
        
    elif dtype == 'rolling_mark':
        bright = int(3 + 4 * s)
        thickness = int(1 + s)
        for i in range(3):
            y_pos = int(h * (0.3 + 0.15 * i))
            val = int(np.clip(d[y_pos,w//2,0]-bright, 0, 255))
            cv2.line(d, (0, y_pos), (w, y_pos), (val,val,val), thickness)
    
    return np.clip(d, 0, 255).astype(np.uint8)


DEFECT_TYPES = [
    'hair_thin_scratch', 'faint_spot', 'micro_crack', 'gradient_stain',
    'surface_roughness', 'slight_discoloration', 'pinhole', 'shallow_dent',
    'oxide_scale', 'rolling_mark'
]
LEVELS = [1, 2, 3]
LEVEL_NAMES = {1: 'L1(극미세)', 2: 'L2(미세)', 3: 'L3(약한)'}


# ============================================================
# Feature Extractors
# ============================================================
class WRN_Layers(nn.Module):
    """WideResNet50 with configurable layer extraction."""
    def __init__(self, layers=[2,3], spatial=False, device='cuda'):
        super().__init__()
        self.device = device
        self.target_layers = layers
        self.spatial = spatial
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256, 64x64
        self.layer2 = backbone.layer2  # 512, 32x32
        self.layer3 = backbone.layer3  # 1024, 16x16
        self.layer4 = backbone.layer4  # 2048, 8x8
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        feats = {1: f1, 2: f2, 3: f3, 4: f4}
        # Determine target spatial size
        if self.spatial:
            # Use layer1 resolution (64x64) for high-res
            target_size = f1.shape[2:]
        else:
            target_size = f2.shape[2:]  # 32x32
        
        selected = []
        for li in self.target_layers:
            f = feats[li]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            selected.append(f)
        
        combined = torch.cat(selected, dim=1)
        
        if self.spatial:
            return combined  # (B, C, H, W)
        else:
            return F.adaptive_avg_pool2d(combined, 1).squeeze(-1).squeeze(-1)


# EfficientNet removed (no internet on server)


# ============================================================
# Scoring Methods
# ============================================================
def score_pooled(features, bank, batch_size=2048):
    """Standard PatchCore scoring on pooled features."""
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(bank).cuda()
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i+batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        scores[i:i+batch_size] = dists.min(dim=1).values.cpu().numpy()
    return scores


def score_spatial(feat_map, bank, topk=5, use_gaussian=True):
    """Heatmap scoring with TopK + optional Gaussian smoothing."""
    if feat_map.dim() == 4:
        feat_map = feat_map[0]  # (C, H, W)
    C, H, W = feat_map.shape
    patches = feat_map.permute(1, 2, 0).reshape(-1, C)
    bank_t = torch.from_numpy(bank).cuda()
    patches_t = patches.cuda()
    
    dists = torch.cdist(patches_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    min_dists = dists.min(dim=1).values
    score_map = min_dists.reshape(H, W).cpu().numpy()
    
    if use_gaussian:
        score_map = gaussian_filter(score_map, sigma=2)
    
    flat = score_map.flatten()
    topk_vals = np.sort(flat)[-topk:]
    return topk_vals.mean(), score_map


def score_pca_recon(features, pca_components, pca_mean):
    """PCA reconstruction error scoring."""
    centered = features - pca_mean
    projected = centered @ pca_components.T
    reconstructed = projected @ pca_components + pca_mean
    errors = np.sqrt(np.sum((features - reconstructed) ** 2, axis=1))
    return errors


# ============================================================
# Sub-patch method
# ============================================================
def extract_subpatches(tile_img, sub_size, transform, extractor):
    """Split tile into sub-patches, extract features for each."""
    h, w = tile_img.shape[:2]
    patches = []
    for y in range(0, h - sub_size + 1, sub_size):
        for x in range(0, w - sub_size + 1, sub_size):
            sub = tile_img[y:y+sub_size, x:x+sub_size]
            sub_rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            patches.append(transform(sub_rgb))
    if not patches:
        return None
    batch = torch.stack(patches)
    with torch.no_grad():
        feats = extractor(batch)
    return feats.cpu().numpy()


# ============================================================
# Main Experiment
# ============================================================
def find_img(spec):
    for e in os.listdir(NAS_DIR):
        p = os.path.join(NAS_DIR, e)
        if not os.path.isdir(p): continue
        if len(e)==8 and e.isdigit():
            for s in os.listdir(p):
                if spec in s:
                    sp = os.path.join(p, s)
                    for c in [f'camera_{i}' for i in range(1,11)]:
                        cp = os.path.join(sp, c)
                        if os.path.isdir(cp):
                            imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                            if len(imgs)>10: return os.path.join(cp, imgs[len(imgs)//2]), c
        elif spec in e:
            for c in [f'camera_{i}' for i in range(1,11)]:
                cp = os.path.join(p, c)
                if os.path.isdir(cp):
                    imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                    if len(imgs)>10: return os.path.join(cp, imgs[len(imgs)//2]), c
    return None, None


def tiles(img):
    h, w = img.shape[:2]
    ts = TILE_SIZE
    r = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            r.append((img[y:y+ts, x:x+ts], (x, y)))
    return r


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 2: Deep Learning PatchCore Detection Limit Breakthrough")
    print("=" * 70)
    
    # Load image and tiles
    path, cam = find_img(SPEC)
    if not path:
        print("No image found!")
        return
    print(f"Image: {path}\nCamera: {cam}")
    img = cv2.imread(path)
    all_t = tiles(img)
    print(f"Total tiles: {len(all_t)}")
    
    # Find memory bank
    cn = int(cam.replace('camera_', ''))
    mb_path = None
    for g, info in CAMERA_GROUPS.items():
        if cn in info['cams']:
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb
                break
    if not mb_path:
        for g in range(1, 6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb
                break
    
    memory_bank = np.load(mb_path)
    print(f"Memory bank: {memory_bank.shape}")
    
    # Select test tiles (diverse variance)
    vars_list = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    test_indices = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    test_tiles = [(all_t[i][0], all_t[i][1]) for i in test_indices]
    print(f"Test tiles: {test_indices}")
    
    # Prepare all test cases: 10 defects × 3 levels × 3 tiles = 90 per method
    print("\nGenerating 90 defect cases (10 types × 3 levels × 3 tiles)...")
    
    # ============================================================
    # Define methods to test
    # ============================================================
    methods = {}
    
    # --- Method A: Baseline (WRN L2+L3 pooled) ---
    print("\n[A] Baseline: WRN50 L2+L3 (current)")
    ext_base = WRN_Layers(layers=[2,3], spatial=False)
    # Build bank for baseline from existing memory_bank
    # Actually rebuild from tiles for fair comparison
    print("  Extracting normal tile features...")
    normal_batches = []
    for i in range(0, len(all_t), 32):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+32, len(all_t)))])
        with torch.no_grad():
            normal_batches.append(ext_base(batch).cpu().numpy())
    bank_base = np.concatenate(normal_batches)
    methods['A_baseline'] = {'extractor': ext_base, 'bank': bank_base, 'spatial': False}
    print(f"  Bank shape: {bank_base.shape}")
    
    # --- Method B: WRN L1+L2+L3 ---
    print("\n[B] WRN50 L1+L2+L3")
    ext_l123 = WRN_Layers(layers=[1,2,3], spatial=False)
    normal_batches = []
    for i in range(0, len(all_t), 32):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+32, len(all_t)))])
        with torch.no_grad():
            normal_batches.append(ext_l123(batch).cpu().numpy())
    bank_l123 = np.concatenate(normal_batches)
    methods['B_L123'] = {'extractor': ext_l123, 'bank': bank_l123, 'spatial': False}
    print(f"  Bank shape: {bank_l123.shape}")
    
    # --- Method C: WRN L1+L2+L3+L4 ---
    print("\n[C] WRN50 L1+L2+L3+L4")
    ext_l1234 = WRN_Layers(layers=[1,2,3,4], spatial=False)
    normal_batches = []
    for i in range(0, len(all_t), 32):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+32, len(all_t)))])
        with torch.no_grad():
            normal_batches.append(ext_l1234(batch).cpu().numpy())
    bank_l1234 = np.concatenate(normal_batches)
    methods['C_L1234'] = {'extractor': ext_l1234, 'bank': bank_l1234, 'spatial': False}
    print(f"  Bank shape: {bank_l1234.shape}")

    # --- Method D: WRN L1 only (highest resolution features) ---
    print("\n[D] WRN50 L1 only (256-dim, 64x64 resolution)")
    ext_l1 = WRN_Layers(layers=[1], spatial=False)
    normal_batches = []
    for i in range(0, len(all_t), 32):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+32, len(all_t)))])
        with torch.no_grad():
            normal_batches.append(ext_l1(batch).cpu().numpy())
    bank_l1 = np.concatenate(normal_batches)
    methods['D_L1only'] = {'extractor': ext_l1, 'bank': bank_l1, 'spatial': False}
    print(f"  Bank shape: {bank_l1.shape}")

    # --- Method E: Heatmap L2+L3 (spatial, TopK5+Gaussian) ---
    print("\n[E] Heatmap L2+L3 spatial TopK5+Gaussian")
    ext_hm = WRN_Layers(layers=[2,3], spatial=True)
    # Build spatial bank (subsample patches from normal tiles)
    spatial_bank_list = []
    for i in range(0, len(all_t), 16):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+16, len(all_t)))])
        with torch.no_grad():
            fm = ext_hm(batch)
        B, C, H, W = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        step = max(1, len(patches) // 1000)
        spatial_bank_list.append(patches[::step])
    bank_spatial = np.concatenate(spatial_bank_list)
    methods['E_Heatmap'] = {'extractor': ext_hm, 'bank': bank_spatial, 'spatial': True}
    print(f"  Spatial bank: {bank_spatial.shape}")

    # --- Method F: Heatmap L1+L2+L3 high-res (64x64) ---
    print("\n[F] Heatmap L1+L2+L3 high-res 64x64")
    ext_hm_hr = WRN_Layers(layers=[1,2,3], spatial=True)
    spatial_bank_hr_list = []
    for i in range(0, len(all_t), 16):
        batch = torch.stack([T(cv2.cvtColor(all_t[j][0], cv2.COLOR_BGR2RGB)) for j in range(i, min(i+16, len(all_t)))])
        with torch.no_grad():
            fm = ext_hm_hr(batch)
        B, C, H, W = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        step = max(1, len(patches) // 1000)
        spatial_bank_hr_list.append(patches[::step])
    bank_spatial_hr = np.concatenate(spatial_bank_hr_list)
    methods['F_Heatmap_HR'] = {'extractor': ext_hm_hr, 'bank': bank_spatial_hr, 'spatial': True}
    print(f"  High-res spatial bank: {bank_spatial_hr.shape}")

    # --- Method G: Sub-patch 128x128 ---
    print("\n[G] Sub-patch 128x128")
    T128 = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((128, 128)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ext_sub = WRN_Layers(layers=[2,3], spatial=False)
    # Build sub-patch bank
    sub_bank_list = []
    for j in range(0, len(all_t), 8):
        for jj in range(j, min(j+8, len(all_t))):
            spf = extract_subpatches(all_t[jj][0], 128, T128, ext_sub)
            if spf is not None:
                sub_bank_list.append(spf)
    bank_sub = np.concatenate(sub_bank_list)
    # Subsample if too large
    if len(bank_sub) > 50000:
        idx_sub = np.random.choice(len(bank_sub), 50000, replace=False)
        bank_sub = bank_sub[idx_sub]
    methods['G_SubPatch128'] = {'extractor': ext_sub, 'bank': bank_sub, 'spatial': False, 'sub_size': 128, 'transform': T128}
    print(f"  Sub-patch bank: {bank_sub.shape}")

    # --- Method H: PCA Reconstruction ---
    print("\n[H] PCA Reconstruction Error")
    # Use baseline features, compute PCA
    n_components = min(128, bank_base.shape[1], bank_base.shape[0])
    pca_mean = bank_base.mean(axis=0)
    centered = bank_base - pca_mean
    # Compute top-k eigenvectors via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:n_components]  # (K, D)
    methods['H_PCA'] = {'pca_components': pca_components, 'pca_mean': pca_mean, 'extractor': ext_base}
    print(f"  PCA components: {pca_components.shape}, explained var top-10: {(S[:10]**2/np.sum(S**2)*100).round(1)}")

    # ============================================================
    # Run all experiments
    # ============================================================
    print("\n" + "=" * 70)
    print("Running experiments: 10 defects × 3 levels × 3 tiles × 8 methods")
    print("=" * 70)
    
    # Results structure: results[method][defect_type][level] = [detected_bools]
    results = {}
    scores_detail = {}  # For analysis
    method_names = list(methods.keys()) + ['H_PCA']
    method_names = list(dict.fromkeys(method_names))  # unique
    
    for mname in method_names:
        results[mname] = {}
        scores_detail[mname] = {}
        for dt in DEFECT_TYPES:
            results[mname][dt] = {1: [], 2: [], 3: []}
            scores_detail[mname][dt] = {1: [], 2: [], 3: []}
    
    for ti, (tile, pos) in enumerate(test_tiles):
        print(f"\n--- Test tile {ti+1}/3, pos={pos} ---")
        
        for dt in DEFECT_TYPES:
            for lv in LEVELS:
                defect_tile = make_defect(tile, dt, lv)
                
                # Process each method
                for mname, minfo in methods.items():
                    ext = minfo.get('extractor')
                    bank = minfo.get('bank')
                    
                    if mname == 'H_PCA':
                        # PCA reconstruction
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            feats = ext(batch).cpu().numpy()
                        errors = score_pca_recon(feats, minfo['pca_components'], minfo['pca_mean'])
                        normal_err = errors[0]
                        defect_err = errors[1]
                        ratio = defect_err / (normal_err + 1e-8)
                        detected = ratio > 1.3
                        
                    elif minfo.get('spatial'):
                        # Heatmap method
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            fmaps = ext(batch)
                        s_normal, _ = score_spatial(fmaps[0], bank)
                        s_defect, _ = score_spatial(fmaps[1], bank)
                        ratio = s_defect / (s_normal + 1e-8)
                        detected = ratio > 1.3
                        
                    elif minfo.get('sub_size'):
                        # Sub-patch method
                        sub_size = minfo['sub_size']
                        trans = minfo['transform']
                        normal_feats = extract_subpatches(tile, sub_size, trans, ext)
                        defect_feats = extract_subpatches(defect_tile, sub_size, trans, ext)
                        if normal_feats is None or defect_feats is None:
                            ratio = 1.0
                            detected = False
                        else:
                            s_normal = score_pooled(normal_feats, bank).max()
                            s_defect = score_pooled(defect_feats, bank).max()
                            ratio = s_defect / (s_normal + 1e-8)
                            detected = ratio > 1.3
                    else:
                        # Standard pooled method
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            feats = ext(batch).cpu().numpy()
                        scores = score_pooled(feats, bank)
                        s_normal = scores[0]
                        s_defect = scores[1]
                        ratio = s_defect / (s_normal + 1e-8)
                        detected = ratio > 1.3
                    
                    results[mname][dt][lv].append(detected)
                    scores_detail[mname][dt][lv].append(ratio)
        
        print(f"  Tile {ti+1} done.")
    
    # ============================================================
    # Compile Results
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Overall detection rate per method
    print("\n1. Overall Detection Rate (90 cases each):")
    print(f"{'Method':<20} {'Total':>6} {'Rate':>8}")
    print("-" * 36)
    method_totals = {}
    for mname in method_names:
        total_det = 0
        total_cases = 0
        for dt in DEFECT_TYPES:
            for lv in LEVELS:
                total_cases += len(results[mname][dt][lv])
                total_det += sum(results[mname][dt][lv])
        rate = total_det / total_cases * 100 if total_cases > 0 else 0
        method_totals[mname] = (total_det, total_cases, rate)
        print(f"{mname:<20} {total_det:>3}/{total_cases:<3} {rate:>6.1f}%")
    
    # Per defect type × method
    print("\n2. Detection Rate by Defect Type:")
    header = f"{'Defect':<22}"
    for mname in method_names:
        short = mname.split('_', 1)[1] if '_' in mname else mname
        header += f" {short:>10}"
    print(header)
    print("-" * len(header))
    
    for dt in DEFECT_TYPES:
        row = f"{dt:<22}"
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            total = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            pct = det / total * 100 if total > 0 else 0
            row += f" {det}/{total}({pct:3.0f}%)"
        print(row)
    
    # Per level
    print("\n3. Detection Rate by Level:")
    for lv in LEVELS:
        print(f"\n  Level {lv} ({LEVEL_NAMES[lv]}):")
        header = f"  {'Method':<20}"
        header += f" {'Det':>4}/{'Tot':>3} {'Rate':>6}"
        print(header)
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            total = sum(len(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            rate = det / total * 100 if total > 0 else 0
            print(f"  {mname:<20} {det:>4}/{total:<3} {rate:>5.1f}%")
    
    # Best method per defect type
    print("\n4. Best Method per Defect Type:")
    print(f"{'Defect':<22} {'Best Method':<20} {'Rate':>6} {'Avg Ratio':>10}")
    print("-" * 60)
    best_mapping = {}
    for dt in DEFECT_TYPES:
        best_rate = -1
        best_method = None
        best_ratio = 0
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            total = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            rate = det / total * 100 if total > 0 else 0
            avg_ratio = np.mean([r for lv in LEVELS for r in scores_detail[mname][dt][lv]])
            if rate > best_rate or (rate == best_rate and avg_ratio > best_ratio):
                best_rate = rate
                best_method = mname
                best_ratio = avg_ratio
        best_mapping[dt] = (best_method, best_rate, best_ratio)
        print(f"{dt:<22} {best_method:<20} {best_rate:>5.1f}% {best_ratio:>9.3f}")
    
    # ============================================================
    # Save results to JSON
    # ============================================================
    json_results = {
        'method_totals': {k: {'detected': v[0], 'total': v[1], 'rate': v[2]} for k, v in method_totals.items()},
        'per_defect': {},
        'per_level': {},
        'best_mapping': {k: {'method': v[0], 'rate': v[1], 'ratio': v[2]} for k, v in best_mapping.items()},
        'scores_detail': {}
    }
    for dt in DEFECT_TYPES:
        json_results['per_defect'][dt] = {}
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            total = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            json_results['per_defect'][dt][mname] = {'detected': det, 'total': total, 'rate': det/total*100 if total>0 else 0}
    
    for mname in method_names:
        json_results['scores_detail'][mname] = {}
        for dt in DEFECT_TYPES:
            json_results['scores_detail'][mname][dt] = {}
            for lv in LEVELS:
                json_results['scores_detail'][mname][dt][str(lv)] = [float(r) for r in scores_detail[mname][dt][lv]]
    
    with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # ============================================================
    # Create comparison visualization
    # ============================================================
    print("\nGenerating comparison images...")
    
    # Create summary chart image
    n_methods = len(method_names)
    n_defects = len(DEFECT_TYPES)
    cell_w, cell_h = 80, 25
    margin_left = 180
    margin_top = 100
    img_w = margin_left + n_methods * cell_w + 20
    img_h = margin_top + n_defects * cell_h + 120  # extra for level breakdown
    
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(canvas, 'Phase 2: Detection Rate by Method x Defect', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Column headers (methods)
    for mi, mname in enumerate(method_names):
        short = mname.split('_', 1)[1] if '_' in mname else mname
        x = margin_left + mi * cell_w + 5
        cv2.putText(canvas, short[:10], (x, margin_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Rows (defect types)
    for di, dt in enumerate(DEFECT_TYPES):
        y = margin_top + di * cell_h
        cv2.putText(canvas, dt[:20], (5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        for mi, mname in enumerate(method_names):
            x = margin_left + mi * cell_w
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            total = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            rate = det / total * 100 if total > 0 else 0
            
            # Color: green if high, red if low
            g = int(min(255, rate * 2.55))
            r = int(min(255, (100 - rate) * 2.55))
            cv2.rectangle(canvas, (x+2, y+2), (x+cell_w-2, y+cell_h-2), (0, g, r), -1)
            cv2.putText(canvas, f'{rate:.0f}%', (x+15, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Total row
    y_total = margin_top + n_defects * cell_h + 10
    cv2.putText(canvas, 'TOTAL', (5, y_total + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    for mi, mname in enumerate(method_names):
        x = margin_left + mi * cell_w
        rate = method_totals[mname][2]
        g = int(min(255, rate * 2.55))
        r = int(min(255, (100 - rate) * 2.55))
        cv2.rectangle(canvas, (x+2, y_total+2), (x+cell_w-2, y_total+cell_h-2), (0, g, r), -1)
        cv2.putText(canvas, f'{rate:.0f}%', (x+15, y_total+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    chart_path = os.path.join(RESULT_DIR, 'detection_heatmap.jpg')
    cv2.imwrite(chart_path, canvas)
    print(f"Saved: {chart_path}")
    
    # Create defect sample visualization for best vs worst
    for ti in range(min(1, len(test_tiles))):
        tile = test_tiles[ti][0]
        sample_h = 80
        n_cols = len(DEFECT_TYPES) + 1
        sample_w = sample_h
        pad = 2
        canvas2_w = n_cols * (sample_w + pad) + pad
        canvas2_h = (len(LEVELS) + 1) * (sample_h + pad + 15) + 30
        canvas2 = np.ones((canvas2_h, canvas2_w, 3), dtype=np.uint8) * 240
        
        # Header
        cv2.putText(canvas2, 'Normal', (pad, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1)
        thumb = cv2.resize(tile, (sample_w, sample_h))
        canvas2[15:15+sample_h, pad:pad+sample_w] = thumb
        
        for di, dt in enumerate(DEFECT_TYPES):
            x = pad + (di+1) * (sample_w + pad)
            cv2.putText(canvas2, dt[:12], (x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1)
            
            for li, lv in enumerate(LEVELS):
                y = 15 + (li+1) * (sample_h + pad + 15)
                dtile = make_defect(tile, dt, lv)
                thumb = cv2.resize(dtile, (sample_w, sample_h))
                canvas2[y:y+sample_h, x:x+sample_w] = thumb
                cv2.putText(canvas2, f'L{lv}', (pad, y + sample_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1)
        
        samples_path = os.path.join(RESULT_DIR, 'defect_samples.jpg')
        cv2.imwrite(samples_path, canvas2)
        print(f"Saved: {samples_path}")
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("\n=== EXPERIMENT COMPLETE ===")
    
    # Print final recommendation
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    best_method = max(method_totals.items(), key=lambda x: x[1][2])
    print(f"Best overall: {best_method[0]} ({best_method[1][2]:.1f}%)")
    print(f"vs Baseline: {method_totals['A_baseline'][2]:.1f}%")
    print(f"\nPer-defect optimal mapping:")
    for dt, (meth, rate, ratio) in best_mapping.items():
        print(f"  {dt:<22} → {meth} ({rate:.0f}%, ratio={ratio:.3f})")


if __name__ == '__main__':
    main()
