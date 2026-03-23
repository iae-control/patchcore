#!/usr/bin/env python3
"""
Spot defect detection improvement test.
Tests 3 approaches:
1. Layer1 추가 (layer1+layer2+layer3 feature concat)
2. Heatmap 기반 (spatial anomaly map, no avg pool)
3. Hybrid (PatchCore + pixel-level difference)
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from src.patchcore import PatchCoreModel, FeatureExtractor
from src.config import TILE_SIZE, CAMERA_GROUPS, USE_DATA_PARALLEL

SPEC = '150x75'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser('~/patchcore/test_synthetic')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE,TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

DKINDS = ['scratch','thick_scr','bright','spot','big_spot','crack','stain','multi_scr']


# ============================================================
# Method 1: Layer1 included Feature Extractor
# ============================================================
class FeatureExtractorL123(nn.Module):
    """layer1+layer2+layer3 concat -> bigger feature vector."""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
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
        f1 = self.layer1(x)    # (B, 256, 64, 64)
        f2 = self.layer2(f1)   # (B, 512, 32, 32)
        f3 = self.layer3(f2)   # (B, 1024, 16, 16)
        # upsample all to f2 size (32x32)
        f1_d = F.interpolate(f1, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f3_u = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([f1_d, f2, f3_u], dim=1)  # (B, 256+512+1024=1792, 32, 32)
        feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        return feat


# ============================================================
# Method 2: Heatmap-based scoring (patch-level, not tile-level)
# ============================================================
class HeatmapExtractor(nn.Module):
    """Returns spatial feature map instead of pooled vector."""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
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
        """Returns (B, 1536, H, W) spatial features."""
        x = x.to(self.device)
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([f2, f3_up], dim=1)  # (B, 1536, 32, 32)


def heatmap_score(feat_map, memory_bank, topk=3):
    """
    Score each spatial position against memory bank.
    feat_map: (1536, H, W)
    memory_bank: (M, 1536)
    Returns: max_score, mean_top_score, hot_area_ratio
    """
    C, H, W = feat_map.shape
    # reshape to (H*W, C)
    patches = feat_map.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
    bank_t = torch.from_numpy(memory_bank).cuda()
    patches_t = patches.cuda()
    
    # distance to nearest neighbor for each spatial position
    dists = torch.cdist(patches_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)  # (H*W, M)
    min_dists, _ = dists.min(dim=1)  # (H*W,)
    
    score_map = min_dists.reshape(H, W)
    max_score = score_map.max().item()
    # top-k mean
    topk_vals = torch.topk(min_dists, min(topk, len(min_dists))).values
    mean_top = topk_vals.mean().item()
    # hot area: positions above 80th percentile of the score map
    thresh80 = torch.quantile(min_dists, 0.8)
    hot_ratio = (min_dists > thresh80).float().mean().item()
    
    return max_score, mean_top, hot_ratio, score_map.cpu().numpy()


# ============================================================
# Method 3: Pixel-level difference (hybrid)
# ============================================================
def pixel_diff_score(tile, ref_mean, ref_std):
    """
    Compute pixel-level anomaly score.
    tile: (H,W,3) uint8
    ref_mean: (H,W,3) float mean of normal tiles
    ref_std: (H,W,3) float std of normal tiles (clipped min 1)
    Returns: max_zscore, mean_zscore, anomaly_area_ratio
    """
    t = tile.astype(np.float32)
    diff = np.abs(t - ref_mean)
    z = diff / np.clip(ref_std, 1.0, None)
    z_gray = z.mean(axis=2)  # average across channels
    max_z = z_gray.max()
    mean_z = z_gray.mean()
    area_ratio = (z_gray > 3.0).mean()  # ratio of pixels > 3 sigma
    return max_z, mean_z, area_ratio, z_gray


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


def main():
    print("="*60)
    print("Spot Detection Improvement Test")
    print("="*60)
    
    path, cam = find_img(SPEC)
    if not path: print("No image!"); return
    print(f"Image: {path}\nCamera: {cam}")
    img = cv2.imread(path)
    all_t = tiles(img)
    print(f"Tiles: {len(all_t)}")
    
    # Pick test tiles (same logic as before)
    vars_list = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    print(f"Test tiles: {picks}")
    
    cn = int(cam.replace('camera_', ''))
    gid = None
    gi = None
    mb_path = None
    for g, info in CAMERA_GROUPS.items():
        if cn in info['cams']:
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                gid, gi, mb_path = g, info, mb
                break
    if not mb_path:
        for g in range(1, 6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                gid, gi, mb_path = g, CAMERA_GROUPS.get(g, {}), mb
                break
    
    print(f"Group {gid}: {gi.get('desc','')}")
    memory_bank = np.load(mb_path)
    print(f"Memory bank: {memory_bank.shape}")
    
    # Build reference stats from all normal tiles (for pixel-level method)
    print("\nBuilding pixel reference stats...")
    all_tile_imgs = [t[0] for t in all_t]
    ref_stack = np.stack(all_tile_imgs).astype(np.float32)
    ref_mean = ref_stack.mean(axis=0)
    ref_std = ref_stack.std(axis=0)
    print(f"  ref_mean range: [{ref_mean.min():.1f}, {ref_mean.max():.1f}]")
    print(f"  ref_std range: [{ref_std.min():.1f}, {ref_std.max():.1f}]")
    
    # ============================================================
    # Method 0: Original PatchCore (baseline)
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 0: Original PatchCore (layer2+layer3)")
    print("="*60)
    ext_orig = FeatureExtractor('cuda')
    mdl_orig = PatchCoreModel(); mdl_orig.load(mb_path)
    
    results = {}  # results[method][defect] = [(tile_i, score, ratio, det)]
    for m in ['orig', 'l123', 'heatmap_max', 'heatmap_topk', 'pixel_maxz', 'pixel_area', 'hybrid']:
        results[m] = {dk: [] for dk in DKINDS}
    
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = [defect(tl, k) for k in DKINDS]
        all_pieces = [tl] + dts
        batch = torch.stack([T(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in all_pieces])
        
        with torch.no_grad():
            f_orig = ext_orig(batch).cpu().numpy()
        sc = mdl_orig.score(f_orig)
        s0 = sc[0]
        print(f"\n  Tile {ti+1} pos={pos} | Original score: {s0:.4f}")
        print(f"  {'Defect':<14} {'Score':>8} {'Ratio':>7}")
        print(f"  {'-'*32}")
        for ki, dk in enumerate(DKINDS):
            ds = sc[ki+1]; rat = ds/s0 if s0 > 0 else 0
            print(f"  {dk:<14} {ds:>8.4f} {rat:>6.2f}x")
            results['orig'][dk].append((ti, ds, rat, rat > 2.0))
    
    # ============================================================
    # Method 1: Layer1+2+3
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 1: Layer1+Layer2+Layer3 (1792-dim)")
    print("="*60)
    ext_l123 = FeatureExtractorL123('cuda')
    
    # Build new memory bank from all tiles with l123
    print("Building L123 memory bank from all tiles...")
    all_batch = torch.stack([T(cv2.cvtColor(t[0], cv2.COLOR_BGR2RGB)) for t in all_t])
    with torch.no_grad():
        # Process in mini-batches
        all_feats_l123 = []
        bs = 32
        for i in range(0, len(all_batch), bs):
            f = ext_l123(all_batch[i:i+bs]).cpu().numpy()
            all_feats_l123.append(f)
        all_feats_l123 = np.concatenate(all_feats_l123)
    
    # Simple: use all features as bank (no coreset for test)
    bank_l123 = all_feats_l123
    print(f"L123 bank: {bank_l123.shape}")
    
    mdl_l123 = PatchCoreModel()
    mdl_l123.memory_bank = bank_l123
    
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = [defect(tl, k) for k in DKINDS]
        all_pieces = [tl] + dts
        batch = torch.stack([T(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in all_pieces])
        with torch.no_grad():
            f = []
            for i in range(0, len(batch), 32):
                f.append(ext_l123(batch[i:i+32]).cpu().numpy())
            f = np.concatenate(f)
        sc = mdl_l123.score(f)
        s0 = sc[0]
        print(f"\n  Tile {ti+1} | Original: {s0:.4f}")
        print(f"  {'Defect':<14} {'Score':>8} {'Ratio':>7}")
        print(f"  {'-'*32}")
        for ki, dk in enumerate(DKINDS):
            ds = sc[ki+1]; rat = ds/s0 if s0 > 0 else 0
            print(f"  {dk:<14} {ds:>8.4f} {rat:>6.2f}x")
            results['l123'][dk].append((ti, ds, rat, rat > 2.0))
    
    # ============================================================
    # Method 2: Heatmap-based scoring
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 2: Heatmap (spatial patch-level scoring)")
    print("="*60)
    hm_ext = HeatmapExtractor('cuda')
    
    # Build patch-level memory bank from all normal tiles
    print("Building spatial memory bank...")
    patch_bank_list = []
    for i in range(0, len(all_batch), 8):
        with torch.no_grad():
            fm = hm_ext(all_batch[i:i+8])  # (B, 1536, 32, 32)
        B, C, H, W = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()  # (B*H*W, C)
        # Subsample to keep manageable
        step = max(1, len(patches) // 2000)
        patch_bank_list.append(patches[::step])
    
    patch_bank = np.concatenate(patch_bank_list)
    print(f"Spatial bank: {patch_bank.shape}")
    
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = [defect(tl, k) for k in DKINDS]
        all_pieces = [tl] + dts
        batch = torch.stack([T(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in all_pieces])
        
        with torch.no_grad():
            fmaps = hm_ext(batch)  # (N, 1536, 32, 32)
        
        # Score original
        ms0, mt0, hr0, _ = heatmap_score(fmaps[0], patch_bank)
        print(f"\n  Tile {ti+1} | Orig max={ms0:.4f} topk={mt0:.4f}")
        print(f"  {'Defect':<14} {'Max':>8} {'Ratio':>7} {'TopK':>8} {'Ratio':>7}")
        print(f"  {'-'*48}")
        for ki, dk in enumerate(DKINDS):
            ms, mt, hr, _ = heatmap_score(fmaps[ki+1], patch_bank)
            r_max = ms/ms0 if ms0 > 0 else 0
            r_topk = mt/mt0 if mt0 > 0 else 0
            print(f"  {dk:<14} {ms:>8.4f} {r_max:>6.2f}x {mt:>8.4f} {r_topk:>6.2f}x")
            results['heatmap_max'][dk].append((ti, ms, r_max, r_max > 1.5))
            results['heatmap_topk'][dk].append((ti, mt, r_topk, r_topk > 1.5))
    
    # ============================================================
    # Method 3: Pixel-level difference
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 3: Pixel-level difference")
    print("="*60)
    
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = [defect(tl, k) for k in DKINDS]
        
        mz0, _, ar0, _ = pixel_diff_score(tl, ref_mean, ref_std)
        print(f"\n  Tile {ti+1} | Orig max_z={mz0:.2f} area={ar0:.4f}")
        print(f"  {'Defect':<14} {'MaxZ':>8} {'Ratio':>7} {'Area':>8} {'ARatio':>7}")
        print(f"  {'-'*48}")
        for ki, dk in enumerate(DKINDS):
            mz, _, ar, _ = pixel_diff_score(dts[ki], ref_mean, ref_std)
            rz = mz/mz0 if mz0 > 0 else 0
            ra = ar/ar0 if ar0 > 0 else 0
            print(f"  {dk:<14} {mz:>8.2f} {rz:>6.2f}x {ar:>8.4f} {ra:>6.2f}x")
            results['pixel_maxz'][dk].append((ti, mz, rz, rz > 1.5))
            results['pixel_area'][dk].append((ti, ar, ra, ra > 1.5))
    
    # ============================================================
    # Method 4: Hybrid (PatchCore score + pixel area combined)
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 4: Hybrid (PatchCore + Pixel combined)")
    print("="*60)
    print("  Combined score = PatchCore_ratio + Pixel_area_ratio")
    print("  Detection if combined > 3.0")
    
    for ti in range(len(picks)):
        print(f"\n  Tile {ti+1}:")
        print(f"  {'Defect':<14} {'PC_rat':>8} {'Px_rat':>8} {'Combined':>10} {'Det':>5}")
        print(f"  {'-'*50}")
        for dk in DKINDS:
            pc_rat = results['orig'][dk][ti][2]
            px_rat = results['pixel_area'][dk][ti][2]
            combined = pc_rat + px_rat
            det = combined > 3.0
            results['hybrid'][dk].append((ti, combined, combined, det))
            mark = ' ✓' if det else '  '
            print(f"  {dk:<14} {pc_rat:>8.2f} {px_rat:>8.2f} {combined:>10.2f} {mark}")
    
    # ============================================================
    # SUMMARY: Compare all methods
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: Detection rate by method (across all test tiles)")
    print("="*60)
    methods = {
        'Original (L2+L3)': 'orig',
        'L1+L2+L3': 'l123',
        'Heatmap Max': 'heatmap_max',
        'Heatmap TopK': 'heatmap_topk',
        'Pixel MaxZ': 'pixel_maxz',
        'Pixel Area': 'pixel_area',
        'Hybrid': 'hybrid',
    }
    
    header = f"{'Defect':<14}"
    for mname in methods: header += f" {mname:>16}"
    print(header)
    print("-" * len(header))
    
    for dk in DKINDS:
        row = f"{dk:<14}"
        for mname, mkey in methods.items():
            det_count = sum(1 for r in results[mkey][dk] if r[3])
            total = len(results[mkey][dk])
            avg_rat = np.mean([r[2] for r in results[mkey][dk]])
            row += f" {det_count}/{total} ({avg_rat:.2f}x)"
            # pad to 16
            needed = 16 - len(f"{det_count}/{total} ({avg_rat:.2f}x)")
            if needed > 0: row = row[:-len(f"{det_count}/{total} ({avg_rat:.2f}x)")] + " " * needed + f"{det_count}/{total} ({avg_rat:.2f}x)"
        print(row)
    
    # Spot-specific summary
    spot_types = ['spot', 'big_spot', 'stain', 'bright']
    line_types = ['scratch', 'thick_scr', 'crack', 'multi_scr']
    
    print(f"\n--- Spot-type detection rates ---")
    for mname, mkey in methods.items():
        spot_det = sum(1 for dk in spot_types for r in results[mkey][dk] if r[3])
        spot_tot = sum(len(results[mkey][dk]) for dk in spot_types)
        line_det = sum(1 for dk in line_types for r in results[mkey][dk] if r[3])
        line_tot = sum(len(results[mkey][dk]) for dk in line_types)
        print(f"  {mname:<20}: spot={spot_det}/{spot_tot}, line={line_det}/{line_tot}")
    
    # Save comparison images for best method
    print(f"\nSaving comparison images...")
    
    # Create comprehensive comparison image
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        dts = [defect(tl, k) for k in DKINDS]
        h, w = tl.shape[:2]
        n = len(DKINDS) + 1
        pad = 3
        
        # 5 rows: orig, l123, heatmap, pixel, hybrid
        method_names = ['Original', 'L1+L2+L3', 'Heatmap', 'Pixel', 'Hybrid']
        nrows = 5
        cw = n * (w + pad) + pad
        row_h = h + 25
        ch = nrows * row_h + 40
        canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 240
        
        # Column headers
        labels = ['Original'] + DKINDS
        for j, lb in enumerate(labels):
            xo = pad + j * (w + pad)
            cv2.putText(canvas, lb, (xo, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        
        for row_i, (mname, mkey) in enumerate(zip(method_names, ['orig','l123','heatmap_max','pixel_area','hybrid'])):
            yo = 20 + row_i * row_h
            # Row label
            cv2.putText(canvas, mname, (2, yo + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100,0,0), 1)
            
            # Images
            canvas[yo:yo+h, pad:pad+w] = tl
            for j, dt in enumerate(dts):
                xo = pad + (j+1) * (w + pad)
                canvas[yo:yo+h, xo:xo+w] = dt
                # Score text
                rat = results[mkey][DKINDS[j]][ti][2]
                det = results[mkey][DKINDS[j]][ti][3]
                col = (0,0,200) if det else (0,100,0)
                cv2.putText(canvas, f'{rat:.2f}x', (xo, yo+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        
        cv2.imwrite(os.path.join(RESULT_DIR, f'methods_t{ti}.jpg'), canvas)
    
    print(f"\nResults saved to: {RESULT_DIR}")
    print("DONE!")


if __name__ == '__main__':
    main()
