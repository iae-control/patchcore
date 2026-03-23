#!/usr/bin/env python3
"""
Phase 2 v2: Deep Learning PatchCore Detection Limit Breakthrough.
Fixed: separate bank images from test images (no self-referencing).
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
# Subtle Defect Generation
# ============================================================
def make_defect(tile, dtype, level):
    h, w = tile.shape[:2]
    d = tile.copy().astype(np.float32)
    cx, cy = w // 2, h // 2
    s = {1: 0.5, 2: 0.75, 3: 1.0}[level]
    
    if dtype == 'hair_thin_scratch':
        bright = int(10 + 5 * s)
        x1, y1, x2, y2 = int(w*.2), int(h*.3), int(w*.8), int(h*.7)
        val = int(np.clip(d[y1,x1,0]-bright, 0, 255))
        cv2.line(d, (x1,y1), (x2,y2), (val,val,val), 1)
    elif dtype == 'faint_spot':
        r, bright = int(3+2*s), int(8+4*s)
        val = int(np.clip(d[cy,cx,0]-bright, 0, 255))
        cv2.circle(d, (cx,cy), r, (val,val,val), -1)
    elif dtype == 'micro_crack':
        length, bright = int(30+20*s), int(8+7*s)
        x, y = int(w*0.3), int(h*0.5)
        for i in range(length):
            nx = min(x+1, w-1)
            ny = int(np.clip(y + np.random.choice([-1,0,1]), 0, h-1))
            val = int(np.clip(d[y,x,0]-bright, 0, 255))
            cv2.line(d, (x,y), (nx,ny), (val,val,val), 1)
            x, y = nx, ny
    elif dtype == 'gradient_stain':
        r, bright = int(15+10*s), int(10+8*s)
        mask = np.zeros((h,w), dtype=np.float32)
        cv2.circle(mask, (cx,cy), r, 1.0, -1)
        mask = gaussian_filter(mask, sigma=r*0.5)
        mask = mask / (mask.max()+1e-8) * bright
        for c in range(3): d[:,:,c] -= mask
    elif dtype == 'surface_roughness':
        sigma = 5 + 5*s
        d += np.random.normal(0, sigma, (h,w,3)).astype(np.float32)
    elif dtype == 'slight_discoloration':
        d += int(5 + 3*s)
    elif dtype == 'pinhole':
        r, bright = int(1+1*s), int(20+15*s)
        val = int(np.clip(d[cy,cx,0]-bright, 0, 255))
        cv2.circle(d, (cx,cy), r, (val,val,val), -1)
    elif dtype == 'shallow_dent':
        r, bright = int(15+10*s), int(8+7*s)
        Y, X = np.ogrid[:h,:w]
        mask = np.clip(1.0 - np.sqrt((X-cx)**2+(Y-cy)**2).astype(np.float32)/r, 0, 1) * bright
        for c in range(3): d[:,:,c] -= mask
    elif dtype == 'oxide_scale':
        bright = int(8+7*s)
        x, y = int(w*0.3), int(h*0.4)
        pts = []
        for i in range(int(15+10*s)):
            pts.append([x,y])
            x = int(np.clip(x+np.random.randint(1,4), 0, w-1))
            y = int(np.clip(y+np.random.randint(-3,4), 0, h-1))
        pts = np.array(pts, dtype=np.int32)
        val = int(np.clip(d[pts[0,1],pts[0,0],0]-bright, 0, 255))
        cv2.polylines(d, [pts], False, (val,val,val), 1)
    elif dtype == 'rolling_mark':
        bright, thickness = int(3+4*s), int(1+s)
        for i in range(3):
            yp = int(h*(0.3+0.15*i))
            val = int(np.clip(d[yp,w//2,0]-bright, 0, 255))
            cv2.line(d, (0,yp), (w,yp), (val,val,val), thickness)
    return np.clip(d, 0, 255).astype(np.uint8)

DEFECT_TYPES = ['hair_thin_scratch','faint_spot','micro_crack','gradient_stain',
    'surface_roughness','slight_discoloration','pinhole','shallow_dent','oxide_scale','rolling_mark']
LEVELS = [1, 2, 3]
LEVEL_NAMES = {1:'L1(극미세)', 2:'L2(미세)', 3:'L3(약한)'}

# ============================================================
# Feature Extractors
# ============================================================
class WRN_Layers(nn.Module):
    def __init__(self, layers=[2,3], spatial=False, device='cuda'):
        super().__init__()
        self.device = device
        self.target_layers = layers
        self.spatial = spatial
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.to(device); self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        feats = {1:f1, 2:f2, 3:f3, 4:f4}
        target_size = f1.shape[2:] if self.spatial else f2.shape[2:]
        selected = []
        for li in self.target_layers:
            f = feats[li]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            selected.append(f)
        combined = torch.cat(selected, dim=1)
        if self.spatial:
            return combined
        return F.adaptive_avg_pool2d(combined, 1).squeeze(-1).squeeze(-1)


def tiles(img):
    h, w = img.shape[:2]; ts = TILE_SIZE; r = []
    for y in range(0, h-ts+1, ts):
        for x in range(0, w-ts+1, ts):
            r.append((img[y:y+ts, x:x+ts], (x,y)))
    return r


def extract_tiles_features(tile_list, extractor, batch_size=32):
    all_feats = []
    for i in range(0, len(tile_list), batch_size):
        batch = torch.stack([T(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tile_list[i:i+batch_size]])
        with torch.no_grad():
            f = extractor(batch)
            if f.dim() == 4:  # spatial
                B, C, H, W = f.shape
                f = f.permute(0,2,3,1).reshape(-1, C)
            all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats)


def score_pooled(features, bank):
    scores = np.zeros(len(features), dtype=np.float32)
    bank_t = torch.from_numpy(bank).cuda()
    for i in range(0, len(features), 2048):
        batch = torch.from_numpy(features[i:i+2048]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        scores[i:i+2048] = dists.min(dim=1).values.cpu().numpy()
    return scores


def score_spatial_topk(feat_map, bank, topk=5):
    if feat_map.dim() == 3:
        C, H, W = feat_map.shape
    else:
        feat_map = feat_map[0]
        C, H, W = feat_map.shape
    patches = feat_map.permute(1,2,0).reshape(-1, C)
    bank_t = torch.from_numpy(bank).cuda()
    dists = torch.cdist(patches.cuda().unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    min_dists = dists.min(dim=1).values
    score_map = min_dists.reshape(H, W).cpu().numpy()
    score_map = gaussian_filter(score_map, sigma=2)
    topk_vals = np.sort(score_map.flatten())[-topk:]
    return topk_vals.mean(), score_map


def main():
    t0 = time.time()
    print("="*70)
    print("Phase 2 v2: PatchCore Detection Limit Breakthrough")
    print("="*70)
    
    # Find all 150x75 images for camera_1
    all_dirs = []
    for entry in sorted(os.listdir(NAS_DIR)):
        p = os.path.join(NAS_DIR, entry)
        if not os.path.isdir(p): continue
        if len(entry) == 8 and entry.isdigit():
            for sub in os.listdir(p):
                if SPEC in sub:
                    cam_path = os.path.join(p, sub, 'camera_1')
                    if os.path.isdir(cam_path):
                        imgs = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
                        if imgs:
                            all_dirs.append((cam_path, imgs))
        elif SPEC in entry:
            cam_path = os.path.join(p, 'camera_1')
            if os.path.isdir(cam_path):
                imgs = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
                if imgs:
                    all_dirs.append((cam_path, imgs))
    
    print(f"Found {len(all_dirs)} image directories with {SPEC}")
    if len(all_dirs) < 2:
        print("Need at least 2 directories!"); return
    
    # Use first N-1 dirs for bank, last dir for test
    bank_dirs = all_dirs[:-1]
    test_dir = all_dirs[-1]
    print(f"Bank dirs: {len(bank_dirs)}, Test dir: {test_dir[0]}")
    
    # Load test image
    test_img_path = os.path.join(test_dir[0], test_dir[1][len(test_dir[1])//2])
    test_img = cv2.imread(test_img_path)
    all_test_tiles = tiles(test_img)
    print(f"Test image: {test_img_path}")
    print(f"Test tiles: {len(all_test_tiles)}")
    
    # Select 3 diverse test tiles
    vars_list = [np.var(t[0]) for t in all_test_tiles]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    test_indices = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    test_tiles = [all_test_tiles[i][0] for i in test_indices]
    print(f"Selected test tile indices: {test_indices}")
    
    # Load bank tiles (subsample from bank dirs)
    print("\nLoading bank tiles from other images...")
    bank_tile_imgs = []
    max_per_dir = 5  # images per dir
    for cam_path, imgs in bank_dirs[:10]:  # limit dirs
        sample_imgs = imgs[::max(1, len(imgs)//max_per_dir)][:max_per_dir]
        for img_name in sample_imgs:
            img = cv2.imread(os.path.join(cam_path, img_name))
            if img is None: continue
            for t, _ in tiles(img):
                bank_tile_imgs.append(t)
    print(f"Bank tiles: {len(bank_tile_imgs)}")
    
    # Subsample bank if too large
    if len(bank_tile_imgs) > 2000:
        np.random.seed(42)
        sel = np.random.choice(len(bank_tile_imgs), 2000, replace=False)
        bank_tile_imgs = [bank_tile_imgs[i] for i in sel]
        print(f"Subsampled to {len(bank_tile_imgs)} bank tiles")
    
    # Find existing memory bank for baseline comparison
    mb_path = None
    cn = 1
    for g, info in CAMERA_GROUPS.items():
        if cn in info['cams']:
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb; break
    if not mb_path:
        for g in range(1,6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb; break
    
    existing_bank = np.load(mb_path)
    print(f"Existing memory bank: {existing_bank.shape}")
    
    # ============================================================
    # Build feature banks for each method
    # ============================================================
    methods = {}  # name -> {bank, extractor, spatial, desc}
    
    # A: Baseline with existing bank
    print("\n[A] Baseline: WRN50 L2+L3 (existing memory bank)")
    ext_base = WRN_Layers(layers=[2,3], spatial=False)
    methods['A_baseline_existing'] = {
        'extractor': ext_base, 'bank': existing_bank, 'spatial': False,
        'desc': 'WRN L2+L3 기존bank'
    }
    
    # A2: Baseline with fresh bank (for fair comparison with others)
    print("[A2] Baseline: WRN50 L2+L3 (fresh bank)")
    bank_a2 = extract_tiles_features(bank_tile_imgs, ext_base)
    methods['A2_baseline_fresh'] = {
        'extractor': ext_base, 'bank': bank_a2, 'spatial': False,
        'desc': 'WRN L2+L3 새bank'
    }
    print(f"  Bank: {bank_a2.shape}")
    
    # B: L1+L2+L3
    print("[B] WRN50 L1+L2+L3")
    ext_b = WRN_Layers(layers=[1,2,3], spatial=False)
    bank_b = extract_tiles_features(bank_tile_imgs, ext_b)
    methods['B_L123'] = {'extractor': ext_b, 'bank': bank_b, 'spatial': False, 'desc': 'WRN L1+L2+L3'}
    print(f"  Bank: {bank_b.shape}")
    
    # C: L1+L2+L3+L4
    print("[C] WRN50 L1+L2+L3+L4")
    ext_c = WRN_Layers(layers=[1,2,3,4], spatial=False)
    bank_c = extract_tiles_features(bank_tile_imgs, ext_c)
    methods['C_L1234'] = {'extractor': ext_c, 'bank': bank_c, 'spatial': False, 'desc': 'WRN L1234'}
    print(f"  Bank: {bank_c.shape}")
    
    # D: L1 only
    print("[D] WRN50 L1 only")
    ext_d = WRN_Layers(layers=[1], spatial=False)
    bank_d = extract_tiles_features(bank_tile_imgs, ext_d)
    methods['D_L1only'] = {'extractor': ext_d, 'bank': bank_d, 'spatial': False, 'desc': 'WRN L1만'}
    print(f"  Bank: {bank_d.shape}")
    
    # E: Heatmap L2+L3 spatial TopK5+Gaussian
    print("[E] Heatmap L2+L3 spatial")
    ext_e = WRN_Layers(layers=[2,3], spatial=True)
    bank_e_list = []
    for i in range(0, len(bank_tile_imgs), 16):
        batch = torch.stack([T(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in bank_tile_imgs[i:i+16]])
        with torch.no_grad():
            fm = ext_e(batch)
        B, C, H, W = fm.shape
        p = fm.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
        step = max(1, len(p)//500)
        bank_e_list.append(p[::step])
    bank_e = np.concatenate(bank_e_list)
    # Cap at 50k patches
    if len(bank_e) > 50000:
        bank_e = bank_e[np.random.choice(len(bank_e), 50000, replace=False)]
    methods['E_Heatmap'] = {'extractor': ext_e, 'bank': bank_e, 'spatial': True, 'desc': 'Heatmap L2+L3'}
    print(f"  Spatial bank: {bank_e.shape}")
    
    # F: Heatmap L1+L2+L3 high-res
    print("[F] Heatmap L1+L2+L3 high-res 64x64")
    ext_f = WRN_Layers(layers=[1,2,3], spatial=True)
    bank_f_list = []
    for i in range(0, len(bank_tile_imgs), 8):
        batch = torch.stack([T(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in bank_tile_imgs[i:i+8]])
        with torch.no_grad():
            fm = ext_f(batch)
        B, C, H, W = fm.shape
        p = fm.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
        step = max(1, len(p)//200)
        bank_f_list.append(p[::step])
    bank_f = np.concatenate(bank_f_list)
    if len(bank_f) > 50000:
        bank_f = bank_f[np.random.choice(len(bank_f), 50000, replace=False)]
    methods['F_Heatmap_HR'] = {'extractor': ext_f, 'bank': bank_f, 'spatial': True, 'desc': 'Heatmap L123 64x64'}
    print(f"  HR spatial bank: {bank_f.shape}")
    
    # G: Sub-patch 128x128
    print("[G] Sub-patch 128x128")
    T128 = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((128,128)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ext_g = WRN_Layers(layers=[2,3], spatial=False)
    sub_bank_list = []
    for tile_img in bank_tile_imgs[::4]:  # subsample
        h, w = tile_img.shape[:2]
        for y in range(0, h-128+1, 128):
            for x in range(0, w-128+1, 128):
                sub = tile_img[y:y+128, x:x+128]
                sub_bank_list.append(sub)
    # Extract features
    sub_feats = []
    for i in range(0, len(sub_bank_list), 32):
        batch = torch.stack([T128(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)) for s in sub_bank_list[i:i+32]])
        with torch.no_grad():
            sub_feats.append(ext_g(batch).cpu().numpy())
    bank_g = np.concatenate(sub_feats)
    methods['G_SubPatch128'] = {
        'extractor': ext_g, 'bank': bank_g, 'spatial': False,
        'sub_size': 128, 'transform': T128, 'desc': 'SubPatch 128x128'
    }
    print(f"  Sub-patch bank: {bank_g.shape}")
    
    # H: PCA Reconstruction
    print("[H] PCA Reconstruction Error")
    n_comp = min(64, bank_a2.shape[0]-1, bank_a2.shape[1])
    pca_mean = bank_a2.mean(axis=0)
    centered = bank_a2 - pca_mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:n_comp]
    methods['H_PCA'] = {
        'extractor': ext_base, 'pca_components': pca_components, 'pca_mean': pca_mean,
        'spatial': False, 'is_pca': True, 'desc': 'PCA Recon (L2+L3)'
    }
    print(f"  PCA: {n_comp} components")
    
    # ============================================================
    # Run experiments
    # ============================================================
    print("\n" + "="*70)
    print("Running experiments...")
    print("="*70)
    
    results = {}  # method -> defect -> level -> [detected_bools]
    ratios = {}   # method -> defect -> level -> [ratio_values]
    for mname in methods:
        results[mname] = {dt: {lv: [] for lv in LEVELS} for dt in DEFECT_TYPES}
        ratios[mname] = {dt: {lv: [] for lv in LEVELS} for dt in DEFECT_TYPES}
    
    for ti, tile in enumerate(test_tiles):
        print(f"\n--- Test tile {ti+1}/3 ---")
        
        for dt in DEFECT_TYPES:
            for lv in LEVELS:
                np.random.seed(42 + ti * 100 + DEFECT_TYPES.index(dt) * 10 + lv)
                defect_tile = make_defect(tile, dt, lv)
                
                for mname, minfo in methods.items():
                    ext = minfo['extractor']
                    
                    if minfo.get('is_pca'):
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            feats = ext(batch).cpu().numpy()
                        pc = minfo['pca_components']
                        pm = minfo['pca_mean']
                        cent = feats - pm
                        proj = cent @ pc.T
                        recon = proj @ pc + pm
                        errors = np.sqrt(np.sum((feats - recon)**2, axis=1))
                        ratio = float(errors[1] / (errors[0] + 1e-8))
                        detected = ratio > 1.3
                        
                    elif minfo.get('spatial'):
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            fmaps = ext(batch)
                        s_n, _ = score_spatial_topk(fmaps[0], minfo['bank'])
                        s_d, _ = score_spatial_topk(fmaps[1], minfo['bank'])
                        ratio = float(s_d / (s_n + 1e-8))
                        detected = ratio > 1.3
                        
                    elif minfo.get('sub_size'):
                        sub_size = minfo['sub_size']
                        trans = minfo['transform']
                        h, w = tile.shape[:2]
                        n_subs, d_subs = [], []
                        for y in range(0, h-sub_size+1, sub_size):
                            for x in range(0, w-sub_size+1, sub_size):
                                n_subs.append(tile[y:y+sub_size, x:x+sub_size])
                                d_subs.append(defect_tile[y:y+sub_size, x:x+sub_size])
                        n_batch = torch.stack([trans(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)) for s in n_subs])
                        d_batch = torch.stack([trans(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)) for s in d_subs])
                        with torch.no_grad():
                            nf = ext(n_batch).cpu().numpy()
                            df = ext(d_batch).cpu().numpy()
                        s_n = score_pooled(nf, minfo['bank']).max()
                        s_d = score_pooled(df, minfo['bank']).max()
                        ratio = float(s_d / (s_n + 1e-8))
                        detected = ratio > 1.3
                    else:
                        batch = torch.stack([
                            T(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                            T(cv2.cvtColor(defect_tile, cv2.COLOR_BGR2RGB))
                        ])
                        with torch.no_grad():
                            feats = ext(batch).cpu().numpy()
                        sc = score_pooled(feats, minfo['bank'])
                        ratio = float(sc[1] / (sc[0] + 1e-8))
                        detected = ratio > 1.3
                    
                    results[mname][dt][lv].append(detected)
                    ratios[mname][dt][lv].append(ratio)
        
        print(f"  Tile {ti+1} complete.")
    
    # ============================================================
    # Results
    # ============================================================
    method_names = list(methods.keys())
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # 1. Overall
    print("\n1. 전체 검출률 (각 방법당 90건):")
    print(f"{'Method':<25} {'검출':>5}/{'전체':>3} {'Rate':>7} {'AvgRatio':>10}")
    print("-"*55)
    method_totals = {}
    for mname in method_names:
        det = sum(sum(results[mname][dt][lv]) for dt in DEFECT_TYPES for lv in LEVELS)
        tot = sum(len(results[mname][dt][lv]) for dt in DEFECT_TYPES for lv in LEVELS)
        avg_r = np.mean([r for dt in DEFECT_TYPES for lv in LEVELS for r in ratios[mname][dt][lv]])
        rate = det/tot*100 if tot>0 else 0
        method_totals[mname] = {'det': int(det), 'tot': int(tot), 'rate': float(rate), 'avg_ratio': float(avg_r)}
        desc = methods[mname]['desc']
        print(f"{desc:<25} {det:>5}/{tot:<3} {rate:>6.1f}% {avg_r:>9.3f}")
    
    # 2. Per defect type
    print("\n2. 결함 유형별 검출률:")
    header = f"{'결함유형':<22}"
    short_names = []
    for mname in method_names:
        sn = mname.split('_',1)[1][:8]
        short_names.append(sn)
        header += f" {sn:>10}"
    print(header)
    print("-"*len(header))
    
    defect_results = {}
    for dt in DEFECT_TYPES:
        row = f"{dt:<22}"
        defect_results[dt] = {}
        for mi, mname in enumerate(method_names):
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            tot = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            pct = det/tot*100 if tot>0 else 0
            defect_results[dt][mname] = {'det': int(det), 'tot': int(tot), 'rate': float(pct)}
            row += f" {det}/{tot}({pct:3.0f}%)"
        print(row)
    
    # 3. Per level
    print("\n3. 레벨별 검출률:")
    for lv in LEVELS:
        print(f"\n  {LEVEL_NAMES[lv]}:")
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            tot = sum(len(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            rate = det/tot*100 if tot>0 else 0
            desc = methods[mname]['desc']
            print(f"    {desc:<25} {det:>3}/{tot:<3} {rate:>5.1f}%")
    
    # 4. Best method per defect
    print("\n4. 결함별 최적 방법:")
    print(f"{'결함':<22} {'최적방법':<25} {'Rate':>6} {'AvgRatio':>10}")
    print("-"*65)
    best_map = {}
    for dt in DEFECT_TYPES:
        best_rate, best_m, best_r = -1, '', 0
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for lv in LEVELS)
            tot = sum(len(results[mname][dt][lv]) for lv in LEVELS)
            rate = det/tot*100 if tot>0 else 0
            avg_r = np.mean([r for lv in LEVELS for r in ratios[mname][dt][lv]])
            if rate > best_rate or (rate == best_rate and avg_r > best_r):
                best_rate, best_m, best_r = rate, mname, avg_r
        best_map[dt] = {'method': best_m, 'rate': float(best_rate), 'ratio': float(best_r)}
        print(f"{dt:<22} {methods[best_m]['desc']:<25} {best_rate:>5.1f}% {best_r:>9.3f}")
    
    # 5. Previously undetectable defects focus
    hard_defects = ['gradient_stain', 'shallow_dent', 'oxide_scale', 'surface_roughness']
    print("\n5. 이전 미검출 결함 상세 (L1 레벨):")
    for dt in hard_defects:
        print(f"\n  {dt}:")
        for mname in method_names:
            if results[mname][dt][1]:
                r = ratios[mname][dt][1][0]
                d = results[mname][dt][1][0]
                desc = methods[mname]['desc']
                mark = '✓' if d else '✗'
                print(f"    {mark} {desc:<25} ratio={r:.3f}")
    
    # ============================================================
    # Save results
    # ============================================================
    json_out = {
        'method_totals': method_totals,
        'defect_results': defect_results,
        'best_mapping': best_map,
        'level_detail': {},
        'all_ratios': {}
    }
    for lv in LEVELS:
        json_out['level_detail'][str(lv)] = {}
        for mname in method_names:
            det = sum(sum(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            tot = sum(len(results[mname][dt][lv]) for dt in DEFECT_TYPES)
            json_out['level_detail'][str(lv)][mname] = {'det': int(det), 'tot': int(tot)}
    
    for mname in method_names:
        json_out['all_ratios'][mname] = {}
        for dt in DEFECT_TYPES:
            json_out['all_ratios'][mname][dt] = {}
            for lv in LEVELS:
                json_out['all_ratios'][mname][dt][str(lv)] = [float(x) for x in ratios[mname][dt][lv]]
    
    with open(os.path.join(RESULT_DIR, 'results_v2.json'), 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    
    # ============================================================
    # Create visualization
    # ============================================================
    print("\nGenerating visualization...")
    n_m = len(method_names)
    n_d = len(DEFECT_TYPES)
    cw, ch = 85, 28
    ml, mt = 180, 80
    img_w = ml + n_m * cw + 20
    img_h = mt + n_d * ch + 80
    
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, 'Phase 2: Detection Rate (%) - Method x Defect', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    cv2.putText(canvas, 'Green=high, Red=low. Threshold: ratio>1.3', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
    
    for mi, mname in enumerate(method_names):
        sn = mname.split('_',1)[1][:10]
        x = ml + mi * cw + 5
        cv2.putText(canvas, sn, (x, mt-10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 1)
    
    for di, dt in enumerate(DEFECT_TYPES):
        y = mt + di * ch
        cv2.putText(canvas, dt[:22], (5, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,0), 1)
        for mi, mname in enumerate(method_names):
            x = ml + mi * cw
            rate = defect_results[dt][mname]['rate']
            g = int(min(255, rate * 2.55))
            r = int(min(255, (100-rate) * 2.55))
            cv2.rectangle(canvas, (x+2,y+2), (x+cw-2,y+ch-2), (0,g,r), -1)
            cv2.putText(canvas, f'{rate:.0f}%', (x+20, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    
    # Total row
    yt = mt + n_d * ch + 10
    cv2.putText(canvas, 'TOTAL', (5, yt+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
    for mi, mname in enumerate(method_names):
        x = ml + mi * cw
        rate = method_totals[mname]['rate']
        g = int(min(255, rate * 2.55))
        r = int(min(255, (100-rate) * 2.55))
        cv2.rectangle(canvas, (x+2,yt+2), (x+cw-2,yt+ch-2), (0,g,r), -1)
        cv2.putText(canvas, f'{rate:.0f}%', (x+20, yt+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    chart_path = os.path.join(RESULT_DIR, 'detection_heatmap_v2.jpg')
    cv2.imwrite(chart_path, canvas)
    print(f"Saved: {chart_path}")
    
    # Defect samples image
    tile = test_tiles[0]
    sh, sw = 64, 64
    pad = 2
    n_cols = len(DEFECT_TYPES) + 1
    n_rows = len(LEVELS) + 1
    cw2 = n_cols * (sw + pad) + pad
    ch2 = n_rows * (sh + pad + 12) + 20
    canvas2 = np.ones((ch2, cw2, 3), dtype=np.uint8) * 240
    
    cv2.putText(canvas2, 'Norm', (pad, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1)
    thumb = cv2.resize(tile, (sw, sh))
    canvas2[15:15+sh, pad:pad+sw] = thumb
    
    for di, dt in enumerate(DEFECT_TYPES):
        x = pad + (di+1) * (sw + pad)
        cv2.putText(canvas2, dt[:8], (x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1)
        for li, lv in enumerate(LEVELS):
            y = 15 + (li+1) * (sh + pad + 12)
            np.random.seed(42)
            dtile = make_defect(tile, dt, lv)
            thumb = cv2.resize(dtile, (sw, sh))
            canvas2[y:y+sh, x:x+sw] = thumb
            cv2.putText(canvas2, f'L{lv}', (pad, y+sh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1)
    
    samples_path = os.path.join(RESULT_DIR, 'defect_samples_v2.jpg')
    cv2.imwrite(samples_path, canvas2)
    print(f"Saved: {samples_path}")
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    best = max(method_totals.items(), key=lambda x: x[1]['rate'])
    print(f"Best overall: {best[0]} - {methods[best[0]]['desc']} ({best[1]['rate']:.1f}%)")
    print(f"Avg ratio: {best[1]['avg_ratio']:.3f}")
    
    # Print recommendation for hard defects
    print("\nHard defect recommendations:")
    for dt in hard_defects:
        bm = best_map[dt]
        print(f"  {dt}: {methods[bm['method']]['desc']} ({bm['rate']:.0f}%, ratio={bm['ratio']:.3f})")
    
    print("\n=== COMPLETE ===")


if __name__ == '__main__':
    main()
