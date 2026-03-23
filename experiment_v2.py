#!/usr/bin/env python3
"""
Comprehensive PatchCore scoring experiment.
Tests multiple scoring strategies to maximize both spot and line defect detection.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from src.patchcore import PatchCoreModel, FeatureExtractor
from src.config import TILE_SIZE, CAMERA_GROUPS, USE_DATA_PARALLEL

SPEC = '150x75'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser('~/patchcore/experiment_v2')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

DKINDS = ['scratch','thick_scr','bright','spot','big_spot','crack','stain','multi_scr']
SPOT_TYPES = ['spot','big_spot','stain','bright']
LINE_TYPES = ['scratch','thick_scr','crack','multi_scr']


class HeatmapExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device)
        if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
            self.layer1 = nn.DataParallel(self.layer1)
            self.layer2 = nn.DataParallel(self.layer2)
            self.layer3 = nn.DataParallel(self.layer3)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        return f2, f3, f3_up, torch.cat([f2, f3_up], dim=1)


def defect(t, k):
    h,w=t.shape[:2]; d=t.copy()
    if k=='scratch': cv2.line(d,(int(w*.1),int(h*.1)),(int(w*.9),int(h*.9)),(30,30,30),2)
    elif k=='thick_scr': cv2.line(d,(int(w*.2),int(h*.1)),(int(w*.8),int(h*.9)),(20,20,20),4)
    elif k=='bright': cv2.line(d,(int(w*.15),int(h*.5)),(int(w*.85),int(h*.5)),(220,220,220),3)
    elif k=='spot': cv2.circle(d,(w//2,h//2),8,(25,25,25),-1)
    elif k=='big_spot': cv2.circle(d,(w//2,h//2),15,(30,30,30),-1)
    elif k=='crack':
        pts=[(int(w*(.1+.16*i)),int(h*(.4+.1*((-1)**i)))) for i in range(6)]
        for i in range(len(pts)-1): cv2.line(d,pts[i],pts[i+1],(25,25,25),2)
    elif k=='stain': cv2.ellipse(d,(w//2,h//2),(18,10),30,0,360,(40,40,40),-1)
    elif k=='multi_scr':
        for i in range(3): x=int(w*.15*(i+1)); cv2.line(d,(x,0),(x+5,h),(30,30,30),2)
    return d


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
    h,w = img.shape[:2]; ts=TILE_SIZE; r=[]
    for y in range(0,h-ts+1,ts):
        for x in range(0,w-ts+1,ts): r.append((img[y:y+ts,x:x+ts],(x,y)))
    return r


def spatial_score_map(feat_map, bank_t):
    """Returns min-distance score map. feat_map: (C,H,W), bank_t: (M,C) tensor on GPU."""
    C, H, W = feat_map.shape
    patches = feat_map.permute(1,2,0).reshape(-1, C).cuda()
    dists = torch.cdist(patches.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)
    return min_dists.reshape(H, W), min_dists


def main():
    print("="*70)
    print("COMPREHENSIVE SCORING EXPERIMENT v2")
    print("="*70)
    
    path, cam = find_img(SPEC)
    if not path: print("No image!"); return
    print(f"Image: {path}\nCamera: {cam}")
    img = cv2.imread(path)
    all_t = tiles(img)
    print(f"Tiles: {len(all_t)}")
    
    vars_list = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    print(f"Test tiles: {picks}")
    
    cn = int(cam.replace('camera_', ''))
    mb_path = None
    for g, info in CAMERA_GROUPS.items():
        if cn in info['cams']:
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb; break
    if not mb_path:
        for g in range(1, 6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb; break
    
    memory_bank = np.load(mb_path)
    print(f"Memory bank: {memory_bank.shape}")
    
    # ── Setup extractors ──
    ext_orig = FeatureExtractor('cuda')
    mdl_orig = PatchCoreModel(); mdl_orig.load(mb_path)
    hm_ext = HeatmapExtractor('cuda')
    
    # Build spatial bank
    print("Building spatial memory bank...")
    all_batch = torch.stack([T(cv2.cvtColor(t[0], cv2.COLOR_BGR2RGB)) for t in all_t])
    patch_bank_list = []
    for i in range(0, len(all_batch), 8):
        with torch.no_grad():
            _, _, _, fm = hm_ext(all_batch[i:i+8])
        B, C, H, W = fm.shape
        patches = fm.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
        step = max(1, len(patches) // 2000)
        patch_bank_list.append(patches[::step])
    patch_bank = np.concatenate(patch_bank_list)
    bank_t = torch.from_numpy(patch_bank).cuda()
    print(f"Spatial bank: {patch_bank.shape}")
    
    # Build per-layer banks (layer2-only, layer3-only)
    print("Building per-layer banks...")
    l2_bank_list, l3_bank_list = [], []
    for i in range(0, len(all_batch), 8):
        with torch.no_grad():
            f2, f3, f3_up, _ = hm_ext(all_batch[i:i+8])
        B2, C2, H2, W2 = f2.shape
        B3, C3, H3, W3 = f3_up.shape
        p2 = f2.permute(0,2,3,1).reshape(-1, C2).cpu().numpy()
        p3 = f3_up.permute(0,2,3,1).reshape(-1, C3).cpu().numpy()
        step2 = max(1, len(p2) // 2000)
        step3 = max(1, len(p3) // 2000)
        l2_bank_list.append(p2[::step2])
        l3_bank_list.append(p3[::step3])
    l2_bank = np.concatenate(l2_bank_list)
    l3_bank = np.concatenate(l3_bank_list)
    l2_bank_t = torch.from_numpy(l2_bank).cuda()
    l3_bank_t = torch.from_numpy(l3_bank).cuda()
    print(f"L2 bank: {l2_bank.shape}, L3 bank: {l3_bank.shape}")
    
    # Compute baseline normal score distribution for adaptive threshold
    print("Computing baseline score distribution...")
    normal_scores_orig = []
    normal_scores_hm = {k: [] for k in [1,3,5,10,20]}
    for i in range(0, len(all_batch), 32):
        batch_slice = all_batch[i:i+32]
        with torch.no_grad():
            f_orig = ext_orig(batch_slice).cpu().numpy()
        sc = mdl_orig.score(f_orig)
        normal_scores_orig.extend(sc.tolist())
        
        with torch.no_grad():
            _, _, _, fm = hm_ext(batch_slice)
        for j in range(fm.shape[0]):
            _, min_d = spatial_score_map(fm[j], bank_t)
            for k in [1,3,5,10,20]:
                topk_vals = torch.topk(min_d, min(k, len(min_d))).values
                normal_scores_hm[k].append(topk_vals.mean().item())
    
    normal_orig = np.array(normal_scores_orig)
    normal_hm = {k: np.array(v) for k, v in normal_scores_hm.items()}
    print(f"  Original: mean={normal_orig.mean():.4f}, std={normal_orig.std():.4f}, p95={np.percentile(normal_orig, 95):.4f}")
    for k in [1,3,5]:
        print(f"  HM topk={k}: mean={normal_hm[k].mean():.4f}, std={normal_hm[k].std():.4f}, p95={np.percentile(normal_hm[k], 95):.4f}")
    
    # ── Results storage ──
    # results[method_name] = {defect_type: [(tile_i, score, normal_score, detected)]}
    ALL = {}
    
    def add_result(method, dk, ti, score, normal_score):
        if method not in ALL:
            ALL[method] = {d: [] for d in DKINDS}
        ALL[method][dk].append((ti, score, normal_score))
    
    # ── Run all experiments ──
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = {dk: defect(tl, dk) for dk in DKINDS}
        all_pieces = [tl] + [dts[dk] for dk in DKINDS]
        batch = torch.stack([T(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in all_pieces])
        
        # Original PatchCore
        with torch.no_grad():
            f_orig = ext_orig(batch).cpu().numpy()
        sc_orig = mdl_orig.score(f_orig)
        s0_orig = sc_orig[0]
        for ki, dk in enumerate(DKINDS):
            add_result('Original', dk, ti, sc_orig[ki+1], s0_orig)
        
        # Heatmap features
        with torch.no_grad():
            f2_all, f3_all, f3up_all, fm_all = hm_ext(batch)
        
        # Heatmap TopK for various K
        smap0, md0 = spatial_score_map(fm_all[0], bank_t)
        for K in [1, 3, 5, 10, 20]:
            s0_k = torch.topk(md0, min(K, len(md0))).values.mean().item()
            for ki, dk in enumerate(DKINDS):
                smap, md = spatial_score_map(fm_all[ki+1], bank_t)
                sk = torch.topk(md, min(K, len(md))).values.mean().item()
                add_result(f'HM_TopK={K}', dk, ti, sk, s0_k)
        
        # Heatmap percentile scoring
        for pct in [99, 95, 90, 80, 70]:
            s0_p = torch.quantile(md0, pct/100.0).item()
            for ki, dk in enumerate(DKINDS):
                _, md = spatial_score_map(fm_all[ki+1], bank_t)
                sp = torch.quantile(md, pct/100.0).item()
                add_result(f'HM_P{pct}', dk, ti, sp, s0_p)
        
        # Layer2-only TopK=3
        smap0_l2, md0_l2 = spatial_score_map(f2_all[0], l2_bank_t)
        s0_l2 = torch.topk(md0_l2, 3).values.mean().item()
        for ki, dk in enumerate(DKINDS):
            _, md_l2 = spatial_score_map(f2_all[ki+1], l2_bank_t)
            s_l2 = torch.topk(md_l2, 3).values.mean().item()
            add_result('L2_TopK3', dk, ti, s_l2, s0_l2)
        
        # Layer3-only TopK=3
        smap0_l3, md0_l3 = spatial_score_map(f3up_all[0], l3_bank_t)
        s0_l3 = torch.topk(md0_l3, 3).values.mean().item()
        for ki, dk in enumerate(DKINDS):
            _, md_l3 = spatial_score_map(f3up_all[ki+1], l3_bank_t)
            s_l3 = torch.topk(md_l3, 3).values.mean().item()
            add_result('L3_TopK3', dk, ti, s_l3, s0_l3)
        
        # Gaussian-weighted heatmap (center-weighted)
        H, W = smap0.shape
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        gauss = np.exp(-((gx - W/2)**2 + (gy - H/2)**2) / (2 * (W/4)**2))
        gauss_t = torch.from_numpy(gauss).cuda()
        
        weighted0 = (smap0 * gauss_t).sum().item() / gauss_t.sum().item()
        for ki, dk in enumerate(DKINDS):
            smap_d, _ = spatial_score_map(fm_all[ki+1], bank_t)
            weighted_d = (smap_d * gauss_t).sum().item() / gauss_t.sum().item()
            add_result('Gauss_Center', dk, ti, weighted_d, weighted0)
        
        # Gaussian-weighted TopK (top patches * gaussian weight)
        for ki, dk in enumerate(DKINDS):
            smap_d, _ = spatial_score_map(fm_all[ki+1], bank_t)
            # Top 5 patches by gaussian-weighted score
            gw_scores = smap_d * gauss_t
            topk_gw = torch.topk(gw_scores.flatten(), 5).values.mean().item()
            add_result('Gauss_TopK5', dk, ti, topk_gw, torch.topk((smap0 * gauss_t).flatten(), 5).values.mean().item())
        
        # ── Ensemble methods ──
        # MAX(original, heatmap_topk3)
        s0_hm3 = torch.topk(md0, 3).values.mean().item()
        for ki, dk in enumerate(DKINDS):
            _, md_d = spatial_score_map(fm_all[ki+1], bank_t)
            s_hm3 = torch.topk(md_d, 3).values.mean().item()
            # Normalize to ratio
            r_orig = sc_orig[ki+1] / s0_orig if s0_orig > 0 else 1
            r_hm = s_hm3 / s0_hm3 if s0_hm3 > 0 else 1
            add_result('Ens_MAX_ratio', dk, ti, max(r_orig, r_hm), 1.0)
            add_result('Ens_SUM_ratio', dk, ti, r_orig + r_hm, 2.0)
            # Weighted: 0.3*orig + 0.7*hm
            add_result('Ens_W37', dk, ti, 0.3*r_orig + 0.7*r_hm, 1.0)
            # OR logic: detected if either > threshold
            # Will compute detection later
        
        # Adaptive threshold (z-score based)
        for ki, dk in enumerate(DKINDS):
            # Original z-score
            z_orig = (sc_orig[ki+1] - normal_orig.mean()) / max(normal_orig.std(), 1e-6)
            add_result('Adaptive_Orig', dk, ti, z_orig, (s0_orig - normal_orig.mean()) / max(normal_orig.std(), 1e-6))
            # HM topk=3 z-score
            _, md_d = spatial_score_map(fm_all[ki+1], bank_t)
            s_hm3 = torch.topk(md_d, 3).values.mean().item()
            z_hm = (s_hm3 - normal_hm[3].mean()) / max(normal_hm[3].std(), 1e-6)
            add_result('Adaptive_HM3', dk, ti, z_hm, (s0_hm3 - normal_hm[3].mean()) / max(normal_hm[3].std(), 1e-6))
            # Combined z-score
            add_result('Adaptive_Combined', dk, ti, max(z_orig, z_hm), max(
                (s0_orig - normal_orig.mean()) / max(normal_orig.std(), 1e-6),
                (s0_hm3 - normal_hm[3].mean()) / max(normal_hm[3].std(), 1e-6)
            ))
        
        print(f"  Tile {ti+1}/{len(picks)} done")
    
    # ── Detection evaluation ──
    # For ratio-based methods: detected if score/normal > threshold
    # For z-score methods: detected if z > threshold
    # Try multiple thresholds and find best
    
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    ratio_methods = [m for m in ALL if not m.startswith('Adaptive') and not m.startswith('Ens_')]
    zscore_methods = [m for m in ALL if m.startswith('Adaptive')]
    ensemble_ratio_methods = [m for m in ALL if m.startswith('Ens_')]
    
    def eval_method(method, threshold):
        spot_det, spot_tot = 0, 0
        line_det, line_tot = 0, 0
        for dk in DKINDS:
            for (ti, score, norm) in ALL[method][dk]:
                ratio = score / norm if norm > 0 else score
                detected = ratio > threshold
                if dk in SPOT_TYPES:
                    spot_det += int(detected); spot_tot += 1
                else:
                    line_det += int(detected); line_tot += 1
        return spot_det, spot_tot, line_det, line_tot
    
    def eval_zscore(method, threshold):
        spot_det, spot_tot = 0, 0
        line_det, line_tot = 0, 0
        for dk in DKINDS:
            for (ti, score, norm) in ALL[method][dk]:
                detected = score > threshold
                if dk in SPOT_TYPES:
                    spot_det += int(detected); spot_tot += 1
                else:
                    line_det += int(detected); line_tot += 1
        return spot_det, spot_tot, line_det, line_tot
    
    # Find best threshold for each method
    best_results = []
    
    for method in ratio_methods:
        best = (0, 0, 0, 0, 0, 0)
        for thr in [1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
            sd, st, ld, lt = eval_method(method, thr)
            total = sd + ld
            if total > best[0] or (total == best[0] and sd > best[1]):
                best = (total, sd, st, ld, lt, thr)
        best_results.append((method, best[5], best[1], best[2], best[3], best[4]))
    
    for method in ensemble_ratio_methods:
        best = (0, 0, 0, 0, 0, 0)
        for thr in [1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
            sd, st, ld, lt = eval_method(method, thr)
            total = sd + ld
            if total > best[0] or (total == best[0] and sd > best[1]):
                best = (total, sd, st, ld, lt, thr)
        best_results.append((method, best[5], best[1], best[2], best[3], best[4]))
    
    for method in zscore_methods:
        best = (0, 0, 0, 0, 0, 0)
        for thr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            sd, st, ld, lt = eval_zscore(method, thr)
            total = sd + ld
            if total > best[0] or (total == best[0] and sd > best[1]):
                best = (total, sd, st, ld, lt, thr)
        best_results.append((method, best[5], best[1], best[2], best[3], best[4]))
    
    # Sort by total detection
    best_results.sort(key=lambda x: (x[2]+x[4], x[2]), reverse=True)
    
    print(f"\n{'Method':<25} {'Thr':>5} {'Spot':>8} {'Line':>8} {'Total':>8}")
    print("-" * 60)
    for method, thr, sd, st, ld, lt in best_results:
        total_d = sd + ld
        total_t = st + lt
        print(f"{method:<25} {thr:>5.1f} {sd:>3}/{st:<3} {ld:>3}/{lt:<3} {total_d:>3}/{total_t:<3}")
    
    # ── Detailed breakdown for top 5 methods ──
    print("\n" + "="*70)
    print("TOP 5 DETAILED BREAKDOWN")
    print("="*70)
    
    for method, thr, sd, st, ld, lt in best_results[:5]:
        print(f"\n{method} (threshold={thr}):")
        print(f"  {'Defect':<14} {'T1':>8} {'T2':>8} {'T3':>8} {'Det':>5}")
        for dk in DKINDS:
            entries = ALL[method][dk]
            ratios = []
            for (ti, score, norm) in entries:
                if method.startswith('Adaptive'):
                    ratios.append(score)
                else:
                    ratios.append(score / norm if norm > 0 else score)
            det_count = sum(1 for r in ratios if (r > thr))
            r_strs = [f"{r:.2f}" for r in ratios]
            while len(r_strs) < 3: r_strs.append("N/A")
            marker = "✓" if det_count == len(ratios) else f"{det_count}/{len(ratios)}"
            print(f"  {dk:<14} {r_strs[0]:>8} {r_strs[1]:>8} {r_strs[2]:>8} {marker:>5}")
    
    # ── OR-logic ensemble: combine Original + HM_TopK=3 ──
    print("\n" + "="*70)
    print("OR-LOGIC ENSEMBLE (detected if EITHER method detects)")
    print("="*70)
    
    or_combos = [
        ('Original@2.0 OR HM_TopK=3@1.3', 'Original', 2.0, 'HM_TopK=3', 1.3),
        ('Original@2.0 OR HM_TopK=3@1.5', 'Original', 2.0, 'HM_TopK=3', 1.5),
        ('Original@1.5 OR HM_TopK=3@1.3', 'Original', 1.5, 'HM_TopK=3', 1.3),
        ('Original@1.5 OR HM_TopK=5@1.3', 'Original', 1.5, 'HM_TopK=5', 1.3),
        ('Original@2.0 OR HM_TopK=5@1.3', 'Original', 2.0, 'HM_TopK=5', 1.3),
        ('Original@2.0 OR HM_P95@1.3', 'Original', 2.0, 'HM_P95', 1.3),
        ('Original@2.0 OR L2_TopK3@1.3', 'Original', 2.0, 'L2_TopK3', 1.3),
        ('Adaptive_Combined@2.0', None, None, None, None),
    ]
    
    print(f"\n{'Combo':<40} {'Spot':>8} {'Line':>8} {'Total':>8}")
    print("-" * 70)
    
    for combo_name, m1, t1, m2, t2 in or_combos:
        if m1 is None:
            # Single adaptive
            sd, st, ld, lt = eval_zscore('Adaptive_Combined', 2.0)
        else:
            spot_det, spot_tot, line_det, line_tot = 0, 0, 0, 0
            for dk in DKINDS:
                for i in range(len(ALL[m1][dk])):
                    ti1, s1, n1 = ALL[m1][dk][i]
                    ti2, s2, n2 = ALL[m2][dk][i]
                    r1 = s1/n1 if n1 > 0 else s1
                    r2 = s2/n2 if n2 > 0 else s2
                    detected = (r1 > t1) or (r2 > t2)
                    if dk in SPOT_TYPES:
                        spot_det += int(detected); spot_tot += 1
                    else:
                        line_det += int(detected); line_tot += 1
            sd, st, ld, lt = spot_det, spot_tot, line_det, line_tot
        total_d = sd + ld
        total_t = st + lt
        print(f"{combo_name:<40} {sd:>3}/{st:<3} {ld:>3}/{lt:<3} {total_d:>3}/{total_t:<3}")
    
    # ── Save detailed JSON ──
    json_out = {}
    for method in ALL:
        json_out[method] = {}
        for dk in DKINDS:
            json_out[method][dk] = [(ti, float(s), float(n)) for ti, s, n in ALL[method][dk]]
    
    with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
        json.dump(json_out, f, indent=2)
    
    print(f"\nResults saved to {RESULT_DIR}/results.json")
    print("DONE!")


if __name__ == '__main__':
    main()
