#!/usr/bin/env python3
"""
PatchCore Overfitting Verification & Grayscale Optimization Experiment
=====================================================================
Part A: Cross-validation with unseen images
Part B: Grayscale verification & 1ch optimization  
Part C: Fair ensemble evaluation
"""
import os, sys, json, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
try:
    from skimage.feature import local_binary_pattern
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    def ssim(a, b, **kw):
        """Simple SSIM approximation."""
        a = a.astype(float); b = b.astype(float)
        mu_a = a.mean(); mu_b = b.mean()
        sig_a = a.std(); sig_b = b.std()
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        C1 = 6.5025; C2 = 58.5225
        return float(((2*mu_a*mu_b+C1)*(2*sig_ab+C2)) / ((mu_a**2+mu_b**2+C1)*(sig_a**2+sig_b**2+C2)))
    def local_binary_pattern(img, P, R, method='uniform'):
        """Simple LBP approximation."""
        h, w = img.shape
        result = np.zeros_like(img, dtype=float)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i,j]
                code = 0
                for k, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                    if img[i+dy, j+dx] >= center: code |= (1 << k)
                result[i,j] = code
        return result

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from src.patchcore import PatchCoreModel, FeatureExtractor
from src.config import TILE_SIZE, CAMERA_GROUPS, USE_DATA_PARALLEL

# Use 200x200 spec (has trained model + NAS data)
SPEC = '200x200'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser('~/patchcore/overfit_experiment')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

DKINDS_OBVIOUS = ['scratch','thick_scr','bright','spot','big_spot','crack','stain','multi_scr']
SPOT_TYPES = ['spot','big_spot','stain','bright']
LINE_TYPES = ['scratch','thick_scr','crack','multi_scr']

# Subtle defect types with 3 intensity levels
SUBTLE_KINDS = ['faint_scratch','micro_spot','slight_stain','hairline_crack','tiny_pit',
                'surface_wave','light_scuff','dim_spot','thin_line','edge_chip']


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


def defect_obvious(t, k):
    """8 obvious defect types."""
    h, w = t.shape[:2]; d = t.copy()
    if k == 'scratch': cv2.line(d, (int(w*.1),int(h*.1)), (int(w*.9),int(h*.9)), (30,30,30), 2)
    elif k == 'thick_scr': cv2.line(d, (int(w*.2),int(h*.1)), (int(w*.8),int(h*.9)), (20,20,20), 4)
    elif k == 'bright': cv2.line(d, (int(w*.15),int(h*.5)), (int(w*.85),int(h*.5)), (220,220,220), 3)
    elif k == 'spot': cv2.circle(d, (w//2,h//2), 8, (25,25,25), -1)
    elif k == 'big_spot': cv2.circle(d, (w//2,h//2), 15, (30,30,30), -1)
    elif k == 'crack':
        pts = [(int(w*(.1+.16*i)), int(h*(.4+.1*((-1)**i)))) for i in range(6)]
        for i in range(len(pts)-1): cv2.line(d, pts[i], pts[i+1], (25,25,25), 2)
    elif k == 'stain': cv2.ellipse(d, (w//2,h//2), (18,10), 30, 0, 360, (40,40,40), -1)
    elif k == 'multi_scr':
        for i in range(3): x = int(w*.15*(i+1)); cv2.line(d, (x,0), (x+5,h), (30,30,30), 2)
    return d


def defect_subtle(t, kind, intensity):
    """10 subtle defect types × 3 intensity levels (1=faintest, 3=most visible)."""
    h, w = t.shape[:2]; d = t.copy()
    base_alpha = intensity * 0.15  # 0.15, 0.30, 0.45
    delta = int(15 * intensity)  # pixel value change
    
    if kind == 'faint_scratch':
        thickness = max(1, intensity)
        cv2.line(d, (int(w*.2),int(h*.2)), (int(w*.8),int(h*.8)), 
                 (max(0, 128-delta),)*3, thickness)
    elif kind == 'micro_spot':
        r = 3 + intensity
        cv2.circle(d, (w//2, h//2), r, (max(0, 128-delta),)*3, -1)
    elif kind == 'slight_stain':
        overlay = d.copy()
        cv2.ellipse(overlay, (w//2, h//2), (10+intensity*3, 6+intensity*2), 
                    45, 0, 360, (max(0, 128-delta),)*3, -1)
        cv2.addWeighted(overlay, base_alpha, d, 1-base_alpha, 0, d)
    elif kind == 'hairline_crack':
        pts = [(int(w*(.2+.12*i)), int(h*(.45+.05*((-1)**i)))) for i in range(5)]
        for i in range(len(pts)-1): cv2.line(d, pts[i], pts[i+1], (max(0, 128-delta),)*3, 1)
    elif kind == 'tiny_pit':
        r = 2 + intensity
        cv2.circle(d, (w//3, h//3), r, (max(0, 100-delta),)*3, -1)
    elif kind == 'surface_wave':
        for y_off in range(0, h, max(1, 20-intensity*5)):
            amplitude = intensity * 2
            for x in range(w):
                ny = min(h-1, max(0, y_off + int(amplitude * np.sin(x * 0.1))))
                if ny < h and x < w:
                    val = max(0, int(d[ny, x, 0]) - delta//2)
                    d[ny, x] = [val]*3
    elif kind == 'light_scuff':
        overlay = d.copy()
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), 
                      (max(0, 140-delta),)*3, -1)
        cv2.addWeighted(overlay, base_alpha*0.5, d, 1-base_alpha*0.5, 0, d)
    elif kind == 'dim_spot':
        overlay = d.copy()
        cv2.circle(overlay, (w//2, h//2), 5+intensity*2, (max(0, 120-delta),)*3, -1)
        cv2.addWeighted(overlay, base_alpha, d, 1-base_alpha, 0, d)
    elif kind == 'thin_line':
        cv2.line(d, (0, h//2), (w, h//2), (max(0, 128-delta),)*3, 1)
    elif kind == 'edge_chip':
        pts = np.array([[0,0], [intensity*5,0], [0,intensity*5]], np.int32)
        cv2.fillPoly(d, [pts], (max(0, 100-delta),)*3)
    return d


def spatial_score_map(feat_map, bank_t):
    C, H, W = feat_map.shape
    patches = feat_map.permute(1,2,0).reshape(-1, C).cuda()
    dists = torch.cdist(patches.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)
    return min_dists.reshape(H, W), min_dists


def tiles(img):
    h, w = img.shape[:2]; ts = TILE_SIZE; r = []
    for y in range(0, h-ts+1, ts):
        for x in range(0, w-ts+1, ts):
            r.append((img[y:y+ts, x:x+ts], (x,y)))
    return r


def find_all_images(spec, n=5):
    """Find multiple images from DIFFERENT folders for cross-validation."""
    results = []
    for e in sorted(os.listdir(NAS_DIR)):
        p = os.path.join(NAS_DIR, e)
        if not os.path.isdir(p): continue
        if spec.lower() not in e.lower() and spec.upper() not in e.upper():
            continue
        cp = os.path.join(p, 'camera_1')
        if not os.path.isdir(cp): continue
        imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
        if len(imgs) < 5: continue
        # Pick image from middle
        mid_img = os.path.join(cp, imgs[len(imgs)//2])
        # Also pick from different positions
        q1_img = os.path.join(cp, imgs[len(imgs)//4])
        q3_img = os.path.join(cp, imgs[3*len(imgs)//4])
        results.append({
            'folder': e, 
            'images': [q1_img, mid_img, q3_img],
            'total_imgs': len(imgs)
        })
        if len(results) >= n:
            break
    return results


def main():
    print("=" * 70)
    print("OVERFITTING VERIFICATION & GRAYSCALE OPTIMIZATION")
    print("=" * 70)
    
    results_summary = {}
    
    # ═══════════════════════════════════════════════════════
    # PART B: Grayscale Verification
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART B: GRAYSCALE VERIFICATION")
    print("=" * 70)
    
    # Find test images
    img_sources = find_all_images(SPEC, n=5)
    if not img_sources:
        # Try with different naming
        for e in sorted(os.listdir(NAS_DIR)):
            if '200' in e:
                cp = os.path.join(NAS_DIR, e, 'camera_1')
                if os.path.isdir(cp):
                    imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                    if len(imgs) >= 5:
                        img_sources.append({
                            'folder': e,
                            'images': [os.path.join(cp, imgs[i]) for i in [len(imgs)//4, len(imgs)//2, 3*len(imgs)//4]],
                            'total_imgs': len(imgs)
                        })
                        if len(img_sources) >= 5: break

    if not img_sources:
        print("ERROR: No images found for spec", SPEC)
        print("Available folders:")
        for e in sorted(os.listdir(NAS_DIR))[:20]:
            print(f"  {e}")
        return
    
    print(f"Found {len(img_sources)} image sources:")
    for src in img_sources:
        print(f"  {src['folder']}: {src['total_imgs']} images")
    
    # Check grayscale
    test_img = cv2.imread(img_sources[0]['images'][0])
    if test_img is not None:
        b, g, r = cv2.split(test_img)
        diff_bg = np.abs(b.astype(float) - g.astype(float)).mean()
        diff_br = np.abs(b.astype(float) - r.astype(float)).mean()
        diff_gr = np.abs(g.astype(float) - r.astype(float)).mean()
        is_gray = max(diff_bg, diff_br, diff_gr) < 1.0
        print(f"\nGrayscale check:")
        print(f"  B-G diff: {diff_bg:.4f}")
        print(f"  B-R diff: {diff_br:.4f}")
        print(f"  G-R diff: {diff_gr:.4f}")
        print(f"  Is grayscale: {is_gray}")
        print(f"  Image shape: {test_img.shape}")
        print(f"  Pixel range: [{test_img.min()}, {test_img.max()}]")
        print(f"  Mean: {test_img.mean():.1f}")
        
        results_summary['grayscale'] = {
            'is_grayscale': bool(is_gray),
            'channel_diffs': {'BG': float(diff_bg), 'BR': float(diff_br), 'GR': float(diff_gr)},
            'image_shape': list(test_img.shape),
            'pixel_range': [int(test_img.min()), int(test_img.max())],
        }
    else:
        print(f"ERROR: Cannot read image {img_sources[0]['images'][0]}")
        return
    
    # ═══════════════════════════════════════════════════════
    # PART A: Cross-validation Overfitting Test
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: CROSS-VALIDATION (OVERFITTING TEST)")
    print("=" * 70)
    
    # Load memory bank (trained model)
    mb_path = os.path.join(OUTPUT_DIR, 'group_1', 'memory_bank.npy')
    if not os.path.exists(mb_path):
        # fallback
        for g in range(1, 6):
            p = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(p): mb_path = p; break
    
    memory_bank = np.load(mb_path)
    print(f"Memory bank: {memory_bank.shape} from {mb_path}")
    
    # Setup extractors
    ext = FeatureExtractor('cuda')
    mdl = PatchCoreModel(); mdl.load(mb_path)
    hm_ext = HeatmapExtractor('cuda')
    
    # ── Test with multiple images from different folders ──
    all_cv_results = {}  # {method: {defect: [(img_idx, score, normal_score)]}}
    
    for img_idx, src in enumerate(img_sources):
        img_path = src['images'][1]  # middle image
        print(f"\n--- Image {img_idx+1}/{len(img_sources)}: {src['folder']} ---")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  SKIP: cannot read {img_path}")
            continue
        
        all_t = tiles(img)
        print(f"  Tiles: {len(all_t)}")
        
        if len(all_t) < 3:
            print(f"  SKIP: too few tiles")
            continue
        
        # Pick 3 test tiles by variance
        vars_list = [np.var(t[0]) for t in all_t]
        idx_sorted = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
        picks = [idx_sorted[0], idx_sorted[len(idx_sorted)//4], idx_sorted[len(idx_sorted)//2]]
        
        # Build spatial bank from THIS image's tiles (simulating "same image" scenario)
        all_batch = torch.stack([T(cv2.cvtColor(t[0], cv2.COLOR_BGR2RGB)) for t in all_t])
        
        # For FAIR test: build spatial bank from OTHER images (not this one)
        other_tiles_for_bank = []
        for other_idx, other_src in enumerate(img_sources):
            if other_idx == img_idx: continue  # skip current image
            other_img = cv2.imread(other_src['images'][1])
            if other_img is None: continue
            ot = tiles(other_img)
            other_tiles_for_bank.extend([t[0] for t in ot])
            if len(other_tiles_for_bank) > 200: break
        
        if len(other_tiles_for_bank) < 10:
            # If not enough other images, use same image but different tiles
            other_tiles_for_bank = [all_t[i][0] for i in range(len(all_t)) if i not in picks]
        
        # Build spatial bank from OTHER images
        other_batch = torch.stack([T(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in other_tiles_for_bank[:100]])
        patch_bank_list = []
        for i in range(0, len(other_batch), 8):
            with torch.no_grad():
                _, _, _, fm = hm_ext(other_batch[i:i+8])
            B, C, H, W = fm.shape
            patches = fm.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
            step = max(1, len(patches) // 2000)
            patch_bank_list.append(patches[::step])
        fair_bank = np.concatenate(patch_bank_list)
        fair_bank_t = torch.from_numpy(fair_bank).cuda()
        
        # Also build "SAME image" spatial bank (for comparison - this is the overfitting scenario)
        same_bank_list = []
        for i in range(0, len(all_batch), 8):
            with torch.no_grad():
                _, _, _, fm = hm_ext(all_batch[i:i+8])
            B, C, H, W = fm.shape
            patches = fm.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
            step = max(1, len(patches) // 2000)
            same_bank_list.append(patches[::step])
        same_bank = np.concatenate(same_bank_list)
        same_bank_t = torch.from_numpy(same_bank).cuda()
        
        # Compute normal score distributions (from the bank tiles)
        normal_orig_scores = []
        normal_hm_fair_scores = {k: [] for k in [3, 5]}
        normal_hm_same_scores = {k: [] for k in [3, 5]}
        
        for i in range(0, len(all_batch), 32):
            batch_slice = all_batch[i:i+32]
            with torch.no_grad():
                f = ext(batch_slice).cpu().numpy()
            sc = mdl.score(f)
            normal_orig_scores.extend(sc.tolist())
            
            with torch.no_grad():
                _, _, _, fm = hm_ext(batch_slice)
            for j in range(fm.shape[0]):
                _, md_fair = spatial_score_map(fm[j], fair_bank_t)
                _, md_same = spatial_score_map(fm[j], same_bank_t)
                for k in [3, 5]:
                    normal_hm_fair_scores[k].append(torch.topk(md_fair, min(k, len(md_fair))).values.mean().item())
                    normal_hm_same_scores[k].append(torch.topk(md_same, min(k, len(md_same))).values.mean().item())
        
        norm_orig = np.array(normal_orig_scores)
        norm_hm_fair = {k: np.array(v) for k, v in normal_hm_fair_scores.items()}
        norm_hm_same = {k: np.array(v) for k, v in normal_hm_same_scores.items()}
        
        print(f"  Normal stats - Orig: mean={norm_orig.mean():.4f} std={norm_orig.std():.4f}")
        print(f"  Normal stats - HM fair TopK3: mean={norm_hm_fair[3].mean():.4f} std={norm_hm_fair[3].std():.4f}")
        print(f"  Normal stats - HM same TopK3: mean={norm_hm_same[3].mean():.4f} std={norm_hm_same[3].std():.4f}")
        
        # Test each picked tile with defects
        for ti, pi in enumerate(picks):
            tl, pos = all_t[pi]
            
            # --- Obvious defects ---
            for dk in DKINDS_OBVIOUS:
                dt = defect_obvious(tl, dk)
                batch = torch.stack([T(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB)), 
                                      T(cv2.cvtColor(dt, cv2.COLOR_BGR2RGB))])
                
                # Original PatchCore
                with torch.no_grad():
                    f = ext(batch).cpu().numpy()
                sc = mdl.score(f)
                _add(all_cv_results, 'Original', dk, img_idx, sc[1], sc[0])
                
                # Heatmap methods
                with torch.no_grad():
                    _, _, _, fm = hm_ext(batch)
                
                # Fair bank
                _, md0_fair = spatial_score_map(fm[0], fair_bank_t)
                _, md1_fair = spatial_score_map(fm[1], fair_bank_t)
                for K in [3, 5]:
                    s0 = torch.topk(md0_fair, min(K, len(md0_fair))).values.mean().item()
                    s1 = torch.topk(md1_fair, min(K, len(md1_fair))).values.mean().item()
                    _add(all_cv_results, f'HM_TopK{K}_fair', dk, img_idx, s1, s0)
                
                # Same bank (overfitting scenario)
                _, md0_same = spatial_score_map(fm[0], same_bank_t)
                _, md1_same = spatial_score_map(fm[1], same_bank_t)
                for K in [3, 5]:
                    s0 = torch.topk(md0_same, min(K, len(md0_same))).values.mean().item()
                    s1 = torch.topk(md1_same, min(K, len(md1_same))).values.mean().item()
                    _add(all_cv_results, f'HM_TopK{K}_same', dk, img_idx, s1, s0)
                
                # Adaptive (z-score) with fair bank
                z_orig = (sc[1] - norm_orig.mean()) / max(norm_orig.std(), 1e-6)
                z_hm_fair = (torch.topk(md1_fair, 3).values.mean().item() - norm_hm_fair[3].mean()) / max(norm_hm_fair[3].std(), 1e-6)
                z_hm_same = (torch.topk(md1_same, 3).values.mean().item() - norm_hm_same[3].mean()) / max(norm_hm_same[3].std(), 1e-6)
                _add(all_cv_results, 'Adaptive_fair', dk, img_idx, max(z_orig, z_hm_fair), 0)
                _add(all_cv_results, 'Adaptive_same', dk, img_idx, max(z_orig, z_hm_same), 0)
            
            # --- Subtle defects ---
            for sk in SUBTLE_KINDS:
                for intensity in [1, 2, 3]:
                    dt = defect_subtle(tl, sk, intensity)
                    batch = torch.stack([T(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB)),
                                          T(cv2.cvtColor(dt, cv2.COLOR_BGR2RGB))])
                    
                    with torch.no_grad():
                        f = ext(batch).cpu().numpy()
                    sc = mdl.score(f)
                    dk_name = f'{sk}_i{intensity}'
                    _add(all_cv_results, 'Original', dk_name, img_idx, sc[1], sc[0])
                    
                    with torch.no_grad():
                        _, _, _, fm = hm_ext(batch)
                    _, md0_fair = spatial_score_map(fm[0], fair_bank_t)
                    _, md1_fair = spatial_score_map(fm[1], fair_bank_t)
                    s0 = torch.topk(md0_fair, 3).values.mean().item()
                    s1 = torch.topk(md1_fair, 3).values.mean().item()
                    _add(all_cv_results, 'HM_TopK3_fair', dk_name, img_idx, s1, s0)
                    
                    _, md0_same = spatial_score_map(fm[0], same_bank_t)
                    _, md1_same = spatial_score_map(fm[1], same_bank_t)
                    s0s = torch.topk(md0_same, 3).values.mean().item()
                    s1s = torch.topk(md1_same, 3).values.mean().item()
                    _add(all_cv_results, 'HM_TopK3_same', dk_name, img_idx, s1s, s0s)
        
        print(f"  Done processing image {img_idx+1}")
    
    # ═══════════════════════════════════════════════════════
    # PART B continued: 1ch vs 3ch pixel analysis
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART B: 1CH vs 3CH PIXEL ANALYSIS")
    print("=" * 70)
    
    gray_results = {'1ch': {}, '3ch': {}}
    
    # Use first image for pixel-level analysis
    test_img_path = img_sources[0]['images'][1]
    img = cv2.imread(test_img_path)
    all_t = tiles(img)
    picks = [0, len(all_t)//4, len(all_t)//2]
    
    # Build reference stats from other tiles
    ref_tiles_3ch = [all_t[i][0] for i in range(len(all_t)) if i not in picks]
    ref_gray = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in ref_tiles_3ch]
    
    ref_mean_1ch = np.mean([t.mean() for t in ref_gray])
    ref_std_1ch = np.mean([t.std() for t in ref_gray])
    ref_mean_3ch = np.mean([t.mean() for t in ref_tiles_3ch])
    ref_std_3ch = np.mean([t.std() for t in ref_tiles_3ch])
    
    # FFT reference
    ref_fft_1ch = np.mean([np.abs(np.fft.fft2(t.astype(float))).mean() for t in ref_gray])
    ref_fft_3ch = np.mean([np.mean([np.abs(np.fft.fft2(t[:,:,c].astype(float))).mean() for c in range(3)]) for t in ref_tiles_3ch])
    
    # LBP reference
    ref_lbp = np.mean([local_binary_pattern(t, 8, 1, method='uniform').mean() for t in ref_gray])
    
    for pi in picks:
        tl = all_t[pi][0]
        tl_gray = cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)
        
        for dk in DKINDS_OBVIOUS[:4]:  # Test subset
            dt = defect_obvious(tl, dk)
            dt_gray = cv2.cvtColor(dt, cv2.COLOR_BGR2GRAY)
            
            # 1ch metrics
            diff_1ch = np.abs(tl_gray.astype(float) - dt_gray.astype(float)).mean()
            ssim_1ch = ssim(tl_gray, dt_gray)
            fft_diff_1ch = abs(np.abs(np.fft.fft2(dt_gray.astype(float))).mean() - ref_fft_1ch)
            lbp_diff = abs(local_binary_pattern(dt_gray, 8, 1, method='uniform').mean() - ref_lbp)
            
            # 3ch metrics
            diff_3ch = np.abs(tl.astype(float) - dt.astype(float)).mean()
            ssim_3ch = ssim(tl, dt, channel_axis=2)
            fft_diff_3ch = abs(np.mean([np.abs(np.fft.fft2(dt[:,:,c].astype(float))).mean() for c in range(3)]) - ref_fft_3ch)
            
            key = f'{dk}_tile{pi}'
            gray_results['1ch'][key] = {
                'pixel_diff': float(diff_1ch),
                'ssim': float(ssim_1ch),
                'fft_diff': float(fft_diff_1ch),
                'lbp_diff': float(lbp_diff),
            }
            gray_results['3ch'][key] = {
                'pixel_diff': float(diff_3ch),
                'ssim': float(ssim_3ch),
                'fft_diff': float(fft_diff_3ch),
            }
    
    print("\n1ch vs 3ch comparison:")
    print(f"{'Key':<30} {'1ch_pdiff':>10} {'3ch_pdiff':>10} {'1ch_ssim':>10} {'3ch_ssim':>10}")
    for key in list(gray_results['1ch'].keys())[:8]:
        r1 = gray_results['1ch'][key]
        r3 = gray_results['3ch'][key]
        print(f"{key:<30} {r1['pixel_diff']:>10.4f} {r3['pixel_diff']:>10.4f} {r1['ssim']:>10.4f} {r3['ssim']:>10.4f}")
    
    # ═══════════════════════════════════════════════════════
    # ANALYSIS & REPORTING
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS: OVERFITTING COMPARISON")
    print("=" * 70)
    
    methods_to_compare = ['Original', 'HM_TopK3_fair', 'HM_TopK3_same', 'HM_TopK5_fair', 'HM_TopK5_same', 
                           'Adaptive_fair', 'Adaptive_same']
    
    # Detection rates for obvious defects
    print(f"\n{'Method':<25} {'Obvious Det Rate':>18} {'Subtle i3':>12} {'Subtle i2':>12} {'Subtle i1':>12}")
    print("-" * 85)
    
    report_data = {}
    
    for method in methods_to_compare:
        if method not in all_cv_results: continue
        
        # Obvious
        obvious_total = 0
        obvious_detected = 0
        for dk in DKINDS_OBVIOUS:
            if dk not in all_cv_results[method]: continue
            for (img_i, score, norm) in all_cv_results[method][dk]:
                obvious_total += 1
                if method.startswith('Adaptive'):
                    if score > 2.0: obvious_detected += 1
                else:
                    ratio = score / norm if norm > 0 else score
                    if ratio > 1.3: obvious_detected += 1
        
        # Subtle by intensity
        subtle_rates = {}
        for intensity in [1, 2, 3]:
            s_total = 0; s_detected = 0
            for sk in SUBTLE_KINDS:
                dk_name = f'{sk}_i{intensity}'
                if dk_name not in all_cv_results.get(method, {}): continue
                for (img_i, score, norm) in all_cv_results[method][dk_name]:
                    s_total += 1
                    if method.startswith('Adaptive'):
                        if score > 2.0: s_detected += 1
                    else:
                        ratio = score / norm if norm > 0 else score
                        if ratio > 1.3: s_detected += 1
            subtle_rates[intensity] = (s_detected, s_total)
        
        obv_rate = f"{obvious_detected}/{obvious_total}" if obvious_total > 0 else "N/A"
        s3 = f"{subtle_rates.get(3,(0,0))[0]}/{subtle_rates.get(3,(0,0))[1]}" if subtle_rates.get(3,(0,0))[1] > 0 else "N/A"
        s2 = f"{subtle_rates.get(2,(0,0))[0]}/{subtle_rates.get(2,(0,0))[1]}" if subtle_rates.get(2,(0,0))[1] > 0 else "N/A"
        s1 = f"{subtle_rates.get(1,(0,0))[0]}/{subtle_rates.get(1,(0,0))[1]}" if subtle_rates.get(1,(0,0))[1] > 0 else "N/A"
        
        print(f"{method:<25} {obv_rate:>18} {s3:>12} {s2:>12} {s1:>12}")
        
        report_data[method] = {
            'obvious': {'detected': obvious_detected, 'total': obvious_total},
            'subtle_i3': subtle_rates.get(3, (0,0)),
            'subtle_i2': subtle_rates.get(2, (0,0)),
            'subtle_i1': subtle_rates.get(1, (0,0)),
        }
    
    # Overfitting comparison: same vs fair
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS: SAME vs FAIR bank")
    print("=" * 70)
    
    for pair in [('HM_TopK3_same', 'HM_TopK3_fair'), ('HM_TopK5_same', 'HM_TopK5_fair'), 
                 ('Adaptive_same', 'Adaptive_fair')]:
        same_m, fair_m = pair
        if same_m not in all_cv_results or fair_m not in all_cv_results:
            continue
        
        print(f"\n{same_m} vs {fair_m}:")
        # Compare average ratios
        same_ratios = []
        fair_ratios = []
        for dk in DKINDS_OBVIOUS:
            if dk in all_cv_results[same_m]:
                for (_, s, n) in all_cv_results[same_m][dk]:
                    if same_m.startswith('Adaptive'):
                        same_ratios.append(s)
                    else:
                        same_ratios.append(s/n if n > 0 else s)
            if dk in all_cv_results[fair_m]:
                for (_, s, n) in all_cv_results[fair_m][dk]:
                    if fair_m.startswith('Adaptive'):
                        fair_ratios.append(s)
                    else:
                        fair_ratios.append(s/n if n > 0 else s)
        
        if same_ratios and fair_ratios:
            print(f"  Same bank avg score/ratio: {np.mean(same_ratios):.4f} ± {np.std(same_ratios):.4f}")
            print(f"  Fair bank avg score/ratio: {np.mean(fair_ratios):.4f} ± {np.std(fair_ratios):.4f}")
            overfit_gap = abs(np.mean(same_ratios) - np.mean(fair_ratios)) / max(np.mean(fair_ratios), 1e-6) * 100
            print(f"  Overfitting gap: {overfit_gap:.1f}%")
    
    # Save all results
    save_data = {
        'grayscale': results_summary.get('grayscale', {}),
        'cv_results': {},
        'gray_analysis': gray_results,
        'report': report_data,
    }
    for method in all_cv_results:
        save_data['cv_results'][method] = {}
        for dk in all_cv_results[method]:
            save_data['cv_results'][method][dk] = [
                (int(i), float(s), float(n)) for i, s, n in all_cv_results[method][dk]
            ]
    
    with open(os.path.join(RESULT_DIR, 'overfit_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {RESULT_DIR}/overfit_results.json")
    
    # ═══════════════════════════════════════════════════════
    # Generate telegram report text
    # ═══════════════════════════════════════════════════════
    report_lines = []
    report_lines.append("📊 PatchCore 과적합 검증 + 흑백 최적화 실험 결과")
    report_lines.append("=" * 40)
    
    # Grayscale
    gs = results_summary.get('grayscale', {})
    report_lines.append(f"\n🔍 Part B: 흑백 확인")
    report_lines.append(f"• 이미지 grayscale 여부: {'✅ 예' if gs.get('is_grayscale') else '❌ 아니오'}")
    if gs.get('channel_diffs'):
        report_lines.append(f"• 채널간 차이: B-G={gs['channel_diffs']['BG']:.4f}, B-R={gs['channel_diffs']['BR']:.4f}")
    
    # 1ch vs 3ch
    report_lines.append(f"\n📐 1ch vs 3ch 비교:")
    for key in list(gray_results['1ch'].keys())[:4]:
        r1 = gray_results['1ch'][key]
        r3 = gray_results['3ch'][key]
        report_lines.append(f"  {key}: 1ch SSIM={r1['ssim']:.4f}, 3ch SSIM={r3['ssim']:.4f}")
    
    # Overfitting
    report_lines.append(f"\n⚠️ Part A: 과적합 분석")
    report_lines.append(f"• 테스트 이미지: {len(img_sources)}개 (서로 다른 폴더)")
    
    for method in methods_to_compare:
        if method not in report_data: continue
        rd = report_data[method]
        obv = rd['obvious']
        obv_pct = obv['detected']/obv['total']*100 if obv['total'] > 0 else 0
        report_lines.append(f"• {method}: 뚜렷한결함 {obv['detected']}/{obv['total']} ({obv_pct:.0f}%)")
    
    # Key finding
    report_lines.append(f"\n🎯 핵심 발견:")
    
    # Compare same vs fair
    for pair_name, same_m, fair_m in [('HM_TopK3', 'HM_TopK3_same', 'HM_TopK3_fair')]:
        if same_m in report_data and fair_m in report_data:
            same_obv = report_data[same_m]['obvious']
            fair_obv = report_data[fair_m]['obvious']
            s_pct = same_obv['detected']/same_obv['total']*100 if same_obv['total'] > 0 else 0
            f_pct = fair_obv['detected']/fair_obv['total']*100 if fair_obv['total'] > 0 else 0
            gap = s_pct - f_pct
            report_lines.append(f"• {pair_name} 과적합 격차: same={s_pct:.0f}% vs fair={f_pct:.0f}% (차이 {gap:.0f}%p)")
    
    if 'Original' in report_data:
        orig = report_data['Original']['obvious']
        orig_pct = orig['detected']/orig['total']*100 if orig['total'] > 0 else 0
        report_lines.append(f"• Original PatchCore (공정): {orig_pct:.0f}%")
    
    report_text = '\n'.join(report_lines)
    
    with open(os.path.join(RESULT_DIR, 'telegram_report.txt'), 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to {RESULT_DIR}/telegram_report.txt")


def _add(results, method, dk, img_idx, score, normal):
    if method not in results:
        results[method] = {}
    if dk not in results[method]:
        results[method][dk] = []
    results[method][dk].append((img_idx, score, normal))


if __name__ == '__main__':
    main()
