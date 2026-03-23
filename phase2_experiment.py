#!/usr/bin/env python3
"""
Phase 2: PatchCore 검출 한계 돌파 연구
CNN 구조적 한계를 보완하는 새로운 접근법 6종 + 앙상블
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from src.patchcore import PatchCoreModel, FeatureExtractor
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

DEFECT_NAMES = [
    'hair_scratch', 'faint_spot', 'micro_crack', 'gradient_stain',
    'roughness', 'discoloration', 'pinhole', 'dent_shadow',
    'oxide_edge', 'rolling_mark'
]
DEFECT_LABELS_KR = {
    'hair_scratch': '극세 스크래치', 'faint_spot': '희미한 점', 'micro_crack': '미세 크랙',
    'gradient_stain': '그라데이션 얼룩', 'roughness': '표면 거칠기 변화',
    'discoloration': '미세 변색', 'pinhole': '핀홀', 'dent_shadow': '얕은 덴트 그림자',
    'oxide_edge': '산화스케일 경계', 'rolling_mark': '압연 마크'
}
LEVELS = [1, 2, 3]
INTENSITY = {1: 1.0, 2: 2.0, 3: 3.0}


def apply_subtle_defect(tile, defect_type, level):
    h, w = tile.shape[:2]
    d = tile.copy().astype(np.float32)
    mult = INTENSITY[level]
    rng = np.random.RandomState(42 + hash(defect_type) % 1000 + level)
    cx, cy = w // 2, h // 2

    if defect_type == 'hair_scratch':
        delta = 5 * mult
        angle = rng.uniform(20, 70)
        length = int(min(w, h) * 0.6)
        rad = math.radians(angle)
        x1 = int(cx - length/2 * math.cos(rad)); y1 = int(cy - length/2 * math.sin(rad))
        x2 = int(cx + length/2 * math.cos(rad)); y2 = int(cy + length/2 * math.sin(rad))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (x1,y1), (x2,y2), 1.0, 1)
        for c in range(3): d[:,:,c] -= delta * mask
    elif defect_type == 'faint_spot':
        radius = int(3 + level); delta = 3 * mult + 2
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        for c in range(3): d[:,:,c] -= delta * mask
    elif defect_type == 'micro_crack':
        delta = 5 * mult; length = int(30 + 10 * level)
        pts = []; x, y = cx - length//2, cy
        for i in range(length):
            x += 1; y += rng.choice([-1, 0, 1])
            y = max(0, min(h-1, y)); x = max(0, min(w-1, x))
            pts.append((x, y))
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in pts: mask[py, px] = 1.0
        for c in range(3): d[:,:,c] -= delta * mask
    elif defect_type == 'gradient_stain':
        radius = 20; delta = 5 * mult
        Y, X = np.mgrid[0:h, 0:w]
        gauss = np.exp(-((X-cx)**2 + (Y-cy)**2) / (2 * radius**2))
        for c in range(3): d[:,:,c] += delta * gauss
    elif defect_type == 'roughness':
        sigma = 3 * mult; rs = 40
        y1, y2 = max(0, cy-rs//2), min(h, cy+rs//2)
        x1, x2 = max(0, cx-rs//2), min(w, cx+rs//2)
        noise = rng.randn(y2-y1, x2-x1, 3) * sigma
        d[y1:y2, x1:x2] += noise
    elif defect_type == 'discoloration':
        delta = 2 * mult + 2; rs = 30
        y1, y2 = max(0, cy-rs//2), min(h, cy+rs//2)
        x1, x2 = max(0, cx-rs//2), min(w, cx+rs//2)
        d[y1:y2, x1:x2] += delta
    elif defect_type == 'pinhole':
        radius = 2; delta = 10 * mult
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        for c in range(3): d[:,:,c] -= delta * mask
    elif defect_type == 'dent_shadow':
        radius = 15; delta = 4 * mult
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        gauss = np.exp(-dist**2 / (2 * radius**2))
        gradient = (X - cx).astype(np.float32) / radius
        shadow = gauss * gradient * delta
        for c in range(3): d[:,:,c] += shadow
    elif defect_type == 'oxide_edge':
        delta = 5 * mult
        pts = []
        y_start = cy - 30
        for i in range(60):
            y = y_start + i; x = cx + int(5 * math.sin(i * 0.5) + rng.randint(-2, 3))
            if 0 <= x < w and 0 <= y < h: pts.append((x, y))
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in pts: mask[py, px] = 1.0
        if level >= 2: mask = cv2.GaussianBlur(mask, (3,3), 0.5); mask = np.clip(mask, 0, 1)
        d[:,:,0] -= delta * mask; d[:,:,1] -= delta * mask * 0.8; d[:,:,2] -= delta * mask * 0.6
    elif defect_type == 'rolling_mark':
        delta = 2 * mult + 1
        mask = np.zeros((h, w), dtype=np.float32)
        for stripe_y in range(cy-20, cy+20, 4):
            if 0 <= stripe_y < h: mask[stripe_y, :] = 1.0
            if 0 <= stripe_y+1 < h: mask[stripe_y+1, :] = 0.5
        for c in range(3): d[:,:,c] += delta * mask
    return np.clip(d, 0, 255).astype(np.uint8)


class HeatmapExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2; self.layer3 = backbone.layer3
        self.to(device)
        if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
            self.layer1 = nn.DataParallel(self.layer1)
            self.layer2 = nn.DataParallel(self.layer2)
            self.layer3 = nn.DataParallel(self.layer3)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        h = self.layer1(x); f2 = self.layer2(h); f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([f2, f3_up], dim=1)


def heatmap_topk_score(feat_map, patch_bank_t, topk=5, gaussian_weight=True):
    C, H, W = feat_map.shape
    patches = feat_map.permute(1, 2, 0).reshape(-1, C).cuda()
    dists = torch.cdist(patches.unsqueeze(0), patch_bank_t.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)
    if gaussian_weight:
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        cy, cx = H/2, W/2
        gdist = ((yy.float()-cy)**2 + (xx.float()-cx)**2).sqrt()
        weights = torch.exp(-gdist / (H * 0.5)).reshape(-1).cuda()
        min_dists = min_dists * weights
    topk_vals = torch.topk(min_dists, min(topk, len(min_dists))).values
    return topk_vals.mean().item()


# ============ NEW METHODS ============

def to_gray(tile):
    if len(tile.shape) == 3:
        return cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return tile.astype(np.float64)


def fft_spectrum(gray):
    """Get magnitude spectrum (log scale, shifted)."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return mag


def dct_spectrum(gray):
    """DCT spectrum."""
    return cv2.dct(gray.astype(np.float32))


def method_fft(tile, normal_stats):
    """FFT spectrum distance from normal mean spectrum."""
    gray = to_gray(tile)
    mag = fft_spectrum(gray)
    diff = np.abs(mag - normal_stats['fft_mean'])
    # Weight low frequencies more (center of spectrum)
    h, w = mag.shape
    Y, X = np.mgrid[0:h, 0:w]
    center_weight = np.exp(-((X - w//2)**2 + (Y - h//2)**2) / (2 * (min(h,w)*0.3)**2))
    weighted_diff = diff * (1 + 2 * center_weight)  # boost low-freq diffs
    return np.mean(np.sort(weighted_diff.ravel())[-50:])  # top-50 mean


def method_dct(tile, normal_stats):
    """DCT low-frequency coefficient distance."""
    gray = to_gray(tile)
    dct = dct_spectrum(gray)
    # Focus on low-frequency block (top-left 16x16)
    block = 16
    low = dct[:block, :block]
    diff = np.abs(low - normal_stats['dct_low_mean'])
    return np.mean(diff)


def compute_lbp(gray, radius=1, n_points=8):
    """Simple LBP implementation."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.int32)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dx = round(radius * np.cos(angle))
        dy = -round(radius * np.sin(angle))
        # Shift image
        shifted = np.zeros_like(gray)
        sy, sx = max(0, dy), max(0, dx)
        ey, ex = min(h, h+dy), min(w, w+dx)
        ssy, ssx = max(0, -dy), max(0, -dx)
        eey, eex = min(h, h-dy), min(w, w-dx)
        shifted[sy:ey, sx:ex] = gray[ssy:eey, ssx:eex]
        lbp += ((shifted >= gray).astype(np.int32)) << i
    return lbp

def method_lbp(tile, normal_stats):
    """LBP histogram distance from normal."""
    gray = to_gray(tile)
    lbp = compute_lbp(gray.astype(np.float64), radius=1, n_points=8)
    n_bins = 256
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    ref = normal_stats['lbp_hist_mean']
    chi2 = np.sum((hist - ref)**2 / (ref + 1e-10))
    return chi2


def compute_ssim(img1, img2, win_size=7):
    """Manual SSIM computation."""
    C1 = (0.01 * 255)**2; C2 = (0.03 * 255)**2
    k = cv2.getGaussianKernel(win_size, 1.5)
    window = k @ k.T
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1**2; mu2_sq = mu2**2; mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def method_ssim(tile, normal_stats):
    """1 - SSIM with normal mean image."""
    gray = to_gray(tile).astype(np.float64)
    ref = normal_stats['gray_mean']
    if gray.shape != ref.shape:
        gray = cv2.resize(gray, (ref.shape[1], ref.shape[0]))
    s = compute_ssim(ref, gray)
    return 1.0 - s


def method_gradient(tile, normal_stats):
    """Gradient magnitude distribution distance."""
    gray = to_gray(tile).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    # Compare statistics
    feat = np.array([np.mean(mag), np.std(mag), np.percentile(mag, 95)])
    ref = normal_stats['grad_feat_mean']
    ref_std = normal_stats['grad_feat_std'] + 1e-6
    return np.max(np.abs(feat - ref) / ref_std)


def method_multireso(tile, normal_stats):
    """Multi-resolution analysis: check at 1x, 1/2, 1/4."""
    gray = to_gray(tile)
    scores = []
    for scale_idx, scale in enumerate([1, 2, 4]):
        if scale > 1:
            g = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
        else:
            g = gray
        # Simple pixel-level MAE from normal mean at that scale
        ref = normal_stats[f'gray_mean_s{scale}']
        if g.shape != ref.shape:
            g = cv2.resize(g, (ref.shape[1], ref.shape[0]))
        diff = np.abs(g - ref)
        # Top-percentile difference
        scores.append(np.mean(np.sort(diff.ravel())[-max(1, int(diff.size*0.05)):]))
    return max(scores)  # worst scale


def find_img(spec):
    for e in os.listdir(NAS_DIR):
        p = os.path.join(NAS_DIR, e)
        if not os.path.isdir(p): continue
        if len(e) == 8 and e.isdigit():
            for s in os.listdir(p):
                if spec in s:
                    sp = os.path.join(p, s)
                    for c in [f'camera_{i}' for i in range(1, 11)]:
                        cp = os.path.join(sp, c)
                        if os.path.isdir(cp):
                            imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                            if len(imgs) > 10: return os.path.join(cp, imgs[len(imgs)//2]), c
        elif spec in e:
            for c in [f'camera_{i}' for i in range(1, 11)]:
                cp = os.path.join(p, c)
                if os.path.isdir(cp):
                    imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                    if len(imgs) > 10: return os.path.join(cp, imgs[len(imgs)//2]), c
    return None, None


def tiles(img):
    h, w = img.shape[:2]; ts = TILE_SIZE; r = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts): r.append((img[y:y+ts, x:x+ts], (x, y)))
    return r


def main():
    t0 = time.time()
    print("=" * 80)
    print("PHASE 2: PatchCore Detection Limit Breakthrough")
    print("6 new methods + PatchCore baseline + ensemble")
    print("=" * 80)

    # Load image
    path, cam = find_img(SPEC)
    if not path: print("No image!"); return
    print(f"Image: {path}\nCamera: {cam}")
    img = cv2.imread(path)
    all_t = tiles(img)
    print(f"Total tiles: {len(all_t)}")

    # Pick 3 test tiles
    vars_list = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    print(f"Test tile indices: {picks}")

    # ========== Build normal statistics for new methods ==========
    print("\nBuilding normal statistics for all methods...")
    # Sample normal tiles
    sample_step = max(1, len(all_t) // 50)
    sample_tiles = [all_t[i][0] for i in range(0, len(all_t), sample_step)]
    print(f"  Using {len(sample_tiles)} normal tiles for stats")

    normal_stats = {}

    # FFT stats
    fft_mags = [fft_spectrum(to_gray(t)) for t in sample_tiles]
    # Ensure all same shape
    min_h = min(m.shape[0] for m in fft_mags)
    min_w = min(m.shape[1] for m in fft_mags)
    fft_mags = [m[:min_h, :min_w] for m in fft_mags]
    normal_stats['fft_mean'] = np.mean(fft_mags, axis=0)
    normal_stats['fft_std'] = np.std(fft_mags, axis=0) + 1e-6

    # DCT low-freq stats
    dct_lows = []
    for t in sample_tiles:
        gray = to_gray(t)
        dct = dct_spectrum(gray)
        dct_lows.append(dct[:16, :16])
    normal_stats['dct_low_mean'] = np.mean(dct_lows, axis=0)
    normal_stats['dct_low_std'] = np.std(dct_lows, axis=0) + 1e-6

    # LBP stats
    n_bins = 256
    lbp_hists = []
    for t in sample_tiles:
        gray = to_gray(t).astype(np.float64)
        lbp = compute_lbp(gray, radius=1, n_points=8)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_hists.append(hist)
    normal_stats['lbp_hist_mean'] = np.mean(lbp_hists, axis=0)
    normal_stats['lbp_hist_std'] = np.std(lbp_hists, axis=0) + 1e-6

    # Gray mean image (for SSIM)
    grays = [to_gray(t) for t in sample_tiles]
    min_h2 = min(g.shape[0] for g in grays)
    min_w2 = min(g.shape[1] for g in grays)
    grays = [g[:min_h2, :min_w2] for g in grays]
    normal_stats['gray_mean'] = np.mean(grays, axis=0)

    # Gradient stats
    grad_feats = []
    for t in sample_tiles:
        gray = to_gray(t).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        grad_feats.append([np.mean(mag), np.std(mag), np.percentile(mag, 95)])
    grad_feats = np.array(grad_feats)
    normal_stats['grad_feat_mean'] = np.mean(grad_feats, axis=0)
    normal_stats['grad_feat_std'] = np.std(grad_feats, axis=0) + 1e-6

    # Multi-resolution mean images
    for scale in [1, 2, 4]:
        scaled = []
        for g in grays:
            if scale > 1:
                s = cv2.resize(g, (g.shape[1]//scale, g.shape[0]//scale))
            else:
                s = g
            scaled.append(s)
        min_sh = min(s.shape[0] for s in scaled)
        min_sw = min(s.shape[1] for s in scaled)
        scaled = [s[:min_sh, :min_sw] for s in scaled]
        normal_stats[f'gray_mean_s{scale}'] = np.mean(scaled, axis=0)

    print("  Normal stats built!")

    # ========== Compute normal score distributions for z-score ==========
    print("\nComputing normal score distributions...")
    NEW_METHODS = {
        'FFT': method_fft,
        'DCT': method_dct,
        'LBP': method_lbp,
        'SSIM': method_ssim,
        'Gradient': method_gradient,
        'MultiReso': method_multireso,
    }

    normal_scores = {m: [] for m in NEW_METHODS}
    normal_scores['PatchCore'] = []

    # Load PatchCore
    mb_path = os.path.join(OUTPUT_DIR, 'group_1', 'memory_bank.npy')
    if not os.path.exists(mb_path):
        for g in range(1, 6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb): mb_path = mb; break
    print(f"  Memory bank: {mb_path}")

    ext_orig = FeatureExtractor('cuda')
    mdl_orig = PatchCoreModel()
    mdl_orig.load(mb_path)
    hm_ext = HeatmapExtractor('cuda')

    # Build spatial patch bank
    print("  Building spatial patch bank...")
    all_batch = torch.stack([T(cv2.cvtColor(t[0], cv2.COLOR_BGR2RGB)) for t in all_t])
    patch_bank_list = []
    for i in range(0, len(all_batch), 8):
        with torch.no_grad(): fm = hm_ext(all_batch[i:i+8])
        B, C, H, W = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        step = max(1, len(patches) // 2000)
        patch_bank_list.append(patches[::step])
    patch_bank = np.concatenate(patch_bank_list)
    patch_bank_t = torch.from_numpy(patch_bank).cuda()
    print(f"  Spatial bank: {patch_bank.shape}")

    normal_scores['HM_TopK5G'] = []

    # Score normal tiles
    for i in range(0, len(all_t), sample_step):
        tl = all_t[i][0]
        # New methods
        for mname, mfunc in NEW_METHODS.items():
            try:
                normal_scores[mname].append(mfunc(tl, normal_stats))
            except Exception as e:
                normal_scores[mname].append(0)

        # PatchCore + Heatmap
        inp = T(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            f = ext_orig(inp).cpu().numpy()
            fm = hm_ext(inp)
        normal_scores['PatchCore'].append(mdl_orig.score(f)[0])
        normal_scores['HM_TopK5G'].append(heatmap_topk_score(fm[0], patch_bank_t, topk=5, gaussian_weight=True))

    # Compute μ, σ for z-scores
    norm_mu = {m: np.mean(v) for m, v in normal_scores.items()}
    norm_std = {m: np.std(v) + 1e-8 for m, v in normal_scores.items()}
    print("\n  Normal distributions:")
    for m in sorted(norm_mu):
        print(f"    {m:>12}: μ={norm_mu[m]:.6f} σ={norm_std[m]:.6f}")

    # ========== Test all defects ==========
    print("\n" + "=" * 80)
    print("TESTING ALL DEFECTS...")
    print("=" * 80)

    ALL_METHODS = list(NEW_METHODS.keys()) + ['PatchCore', 'HM_TopK5G']
    # results[defect][level][tile_idx] = {method: raw_score}
    results = {}

    for dt in DEFECT_NAMES:
        results[dt] = {}
        for lv in LEVELS:
            results[dt][lv] = {}
            for ti_idx, pi in enumerate(picks):
                tl = all_t[pi][0]
                defected = apply_subtle_defect(tl, dt, lv)
                scores = {}

                # New methods
                for mname, mfunc in NEW_METHODS.items():
                    try:
                        scores[mname] = mfunc(defected, normal_stats)
                    except:
                        scores[mname] = norm_mu[mname]

                # PatchCore
                inp = T(cv2.cvtColor(defected, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                with torch.no_grad():
                    f = ext_orig(inp).cpu().numpy()
                    fm = hm_ext(inp)
                scores['PatchCore'] = mdl_orig.score(f)[0]
                scores['HM_TopK5G'] = heatmap_topk_score(fm[0], patch_bank_t, topk=5, gaussian_weight=True)

                results[dt][lv][ti_idx] = scores

        # Print progress
        print(f"  {dt} done")

    # ========== Z-score computation & detection ==========
    Z_THRESH = 2.0  # z > 2 = detected

    def z_score(method, raw):
        return (raw - norm_mu[method]) / norm_std[method]

    def is_detected(method, dt, lv, ti):
        return z_score(method, results[dt][lv][ti][method]) > Z_THRESH

    # Ensemble methods
    ENSEMBLE_METHODS = {
        'Ens_MAX': lambda dt, lv, ti: max(z_score(m, results[dt][lv][ti][m]) for m in ALL_METHODS),
        'Ens_TOP3': lambda dt, lv, ti: np.mean(sorted([z_score(m, results[dt][lv][ti][m]) for m in ALL_METHODS])[-3:]),
        'Ens_WeightedLF': lambda dt, lv, ti: (  # weighted toward low-freq methods
            0.25 * z_score('FFT', results[dt][lv][ti]['FFT']) +
            0.20 * z_score('DCT', results[dt][lv][ti]['DCT']) +
            0.15 * z_score('SSIM', results[dt][lv][ti]['SSIM']) +
            0.15 * z_score('MultiReso', results[dt][lv][ti]['MultiReso']) +
            0.10 * z_score('LBP', results[dt][lv][ti]['LBP']) +
            0.05 * z_score('Gradient', results[dt][lv][ti]['Gradient']) +
            0.05 * z_score('PatchCore', results[dt][lv][ti]['PatchCore']) +
            0.05 * z_score('HM_TopK5G', results[dt][lv][ti]['HM_TopK5G'])
        ),
    }

    ALL_EVAL = ALL_METHODS + list(ENSEMBLE_METHODS.keys())

    def get_z(method, dt, lv, ti):
        if method in ENSEMBLE_METHODS:
            return ENSEMBLE_METHODS[method](dt, lv, ti)
        return z_score(method, results[dt][lv][ti][method])

    def det(method, dt, lv, ti):
        return get_z(method, dt, lv, ti) > Z_THRESH

    # ========== Big results table ==========
    print("\n" + "=" * 120)
    print("DETECTION RESULTS (z > 2.0 = detected, shown as det_count/3)")
    print("=" * 120)

    header = f"{'Defect':<16} {'Lv':>2}"
    for m in ALL_EVAL:
        header += f" | {m:>10}"
    print(header)
    print("-" * len(header))

    total_det = {m: 0 for m in ALL_EVAL}
    total_cases = 0
    det_by_defect = {dt: {m: 0 for m in ALL_EVAL} for dt in DEFECT_NAMES}
    det_by_level = {lv: {m: 0 for m in ALL_EVAL} for lv in LEVELS}

    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            row = f"{dt:<16} L{lv}"
            total_cases += 3
            for m in ALL_EVAL:
                dc = sum(1 for ti in range(3) if det(m, dt, lv, ti))
                avg_z = np.mean([get_z(m, dt, lv, ti) for ti in range(3)])
                total_det[m] += dc
                det_by_defect[dt][m] += dc
                det_by_level[lv][m] += dc
                mark = "✓" if dc == 3 else ("△" if dc > 0 else "✗")
                row += f" | {dc}/3{mark}{avg_z:+.1f}"
            print(row)
        print()

    # ========== Summary ==========
    print("=" * 120)
    print("OVERALL DETECTION RATES:")
    print("=" * 120)
    for m in ALL_EVAL:
        rate = total_det[m] / total_cases * 100
        print(f"  {m:<20}: {total_det[m]}/{total_cases} ({rate:.1f}%)")

    print("\nBY LEVEL:")
    for lv in LEVELS:
        print(f"\n  Level {lv}:")
        for m in ALL_EVAL:
            r = det_by_level[lv][m] / (len(DEFECT_NAMES) * 3) * 100
            print(f"    {m:<20}: {det_by_level[lv][m]}/{len(DEFECT_NAMES)*3} ({r:.1f}%)")

    print("\nBY DEFECT (best method):")
    for dt in DEFECT_NAMES:
        best_m = max(ALL_EVAL, key=lambda m: det_by_defect[dt][m])
        best_r = det_by_defect[dt][best_m]
        print(f"  {dt:<18} ({DEFECT_LABELS_KR[dt]:<10}): {best_r}/9 by {best_m}")

    # Focus defects analysis
    print("\n" + "=" * 80)
    print("FOCUS: Previously undetectable defects")
    print("=" * 80)
    focus = ['gradient_stain', 'dent_shadow', 'oxide_edge', 'roughness']
    for dt in focus:
        print(f"\n  {dt} ({DEFECT_LABELS_KR[dt]}):")
        for lv in LEVELS:
            print(f"    L{lv}: ", end="")
            for m in ALL_EVAL:
                dc = sum(1 for ti in range(3) if det(m, dt, lv, ti))
                if dc > 0:
                    avg_z = np.mean([get_z(m, dt, lv, ti) for ti in range(3)])
                    print(f"{m}={dc}/3(z={avg_z:.1f}) ", end="")
            if not any(det(m, dt, lv, ti) for m in ALL_EVAL for ti in range(3)):
                print("ALL FAILED", end="")
            print()

    # ========== Save comparison images ==========
    print("\nGenerating comparison images...")
    for dt in DEFECT_NAMES:
        tl = all_t[picks[0]][0]
        h, w = tl.shape[:2]
        cols = 4; pad = 2
        cw = cols * (w + pad) + pad; ch = h + 60
        canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 240
        labels = ['Original', 'Level 1', 'Level 2', 'Level 3']
        images = [tl] + [apply_subtle_defect(tl, dt, lv) for lv in LEVELS]
        for j, (lb, im) in enumerate(zip(labels, images)):
            xo = pad + j * (w + pad)
            canvas[0:h, xo:xo+w] = im
            cv2.putText(canvas, lb, (xo, h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            if j > 0:
                lv = j
                best_m = max(ALL_EVAL, key=lambda m: np.mean([get_z(m, dt, lv, ti) for ti in range(3)]))
                best_z = np.mean([get_z(best_m, dt, lv, ti) for ti in range(3)])
                detected_any = any(det(m, dt, lv, ti) for m in ALL_EVAL for ti in range(3))
                color = (0, 150, 0) if detected_any else (0, 0, 200)
                cv2.putText(canvas, f"best:{best_m}(z={best_z:.1f})", (xo, h+35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                detected_methods = [m for m in ALL_EVAL if any(det(m, dt, lv, ti) for ti in range(3))]
                if detected_methods:
                    cv2.putText(canvas, ','.join(detected_methods[:3]), (xo, h+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,120,0), 1)
        cv2.putText(canvas, f"{dt} ({DEFECT_LABELS_KR[dt]})", (pad, h+58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,200), 1)
        cv2.imwrite(os.path.join(RESULT_DIR, f'phase2_{dt}.jpg'), canvas)

    # Combined big image
    tile_h, tile_w = all_t[picks[0]][0].shape[:2]
    n_def = len(DEFECT_NAMES); row_h = tile_h + 65
    total_w = 4 * (tile_w + 2) + 2
    total_h = n_def * row_h + 30
    big = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    for di, dt in enumerate(DEFECT_NAMES):
        tl = all_t[picks[0]][0]; yo = di * row_h + 20
        images = [tl] + [apply_subtle_defect(tl, dt, lv) for lv in LEVELS]
        for j, im in enumerate(images):
            xo = 2 + j * (tile_w + 2)
            big[yo:yo+tile_h, xo:xo+tile_w] = im
            if j > 0:
                lv = j
                detected_any = any(det(m, dt, lv, ti) for m in ALL_EVAL for ti in range(3))
                color = (0, 150, 0) if detected_any else (0, 0, 200)
                best_m = max(ALL_EVAL, key=lambda m: np.mean([get_z(m, dt, lv, ti) for ti in range(3)]))
                best_z = np.mean([get_z(best_m, dt, lv, ti) for ti in range(3)])
                cv2.putText(big, f"z={best_z:.1f}", (xo, yo+tile_h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        cv2.putText(big, DEFECT_LABELS_KR[dt], (2, yo-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,150), 1)

    combined = os.path.join(RESULT_DIR, 'phase2_all_combined.jpg')
    cv2.imwrite(combined, big)

    # ========== Save JSON results ==========
    json_results = {}
    for dt in DEFECT_NAMES:
        json_results[dt] = {}
        for lv in LEVELS:
            json_results[dt][str(lv)] = {}
            for m in ALL_EVAL:
                z_vals = [get_z(m, dt, lv, ti) for ti in range(3)]
                dc = sum(1 for z in z_vals if z > Z_THRESH)
                json_results[dt][str(lv)][m] = {
                    'z_scores': [round(float(z), 3) for z in z_vals],
                    'avg_z': round(float(np.mean(z_vals)), 3),
                    'detected': dc
                }

    with open(os.path.join(RESULT_DIR, 'phase2_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    # ========== Optimal ensemble recommendation ==========
    print("\n" + "=" * 80)
    print("OPTIMAL ENSEMBLE RECOMMENDATION")
    print("=" * 80)

    # Find best single methods for each defect category
    # Try many weight combos
    best_combo = None; best_rate = 0
    import itertools
    weight_options = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
    # Too many combos, use greedy approach
    # Start with individual method rankings
    method_scores_by_case = {}
    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            for ti in range(3):
                key = (dt, lv, ti)
                method_scores_by_case[key] = {m: get_z(m, dt, lv, ti) for m in ALL_METHODS}

    # Greedy: find best combo of methods with max
    print("\n  Testing ensemble strategies...")
    ens_results = {}

    # Strategy 1: MAX of all
    ens_name = "MAX_all"
    det_count = 0
    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            for ti in range(3):
                z = max(get_z(m, dt, lv, ti) for m in ALL_METHODS)
                if z > Z_THRESH: det_count += 1
    ens_results[ens_name] = det_count

    # Strategy 2: MAX of top methods
    for subset_name, subset in [
        ("MAX_freq", ['FFT', 'DCT', 'MultiReso']),
        ("MAX_texture", ['LBP', 'Gradient', 'SSIM']),
        ("MAX_freq+texture", ['FFT', 'DCT', 'MultiReso', 'LBP', 'SSIM']),
        ("MAX_new6", list(NEW_METHODS.keys())),
        ("MAX_all8", ALL_METHODS),
        ("MAX_new6+PC", list(NEW_METHODS.keys()) + ['HM_TopK5G']),
    ]:
        det_count = 0
        for dt in DEFECT_NAMES:
            for lv in LEVELS:
                for ti in range(3):
                    z = max(get_z(m, dt, lv, ti) for m in subset if m in ALL_METHODS)
                    if z > Z_THRESH: det_count += 1
        ens_results[subset_name] = det_count

    print("\n  Ensemble results:")
    for en, dc in sorted(ens_results.items(), key=lambda x: -x[1]):
        print(f"    {en:<25}: {dc}/{total_cases} ({dc/total_cases*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nAll results saved to: {RESULT_DIR}")
    print("DONE!")


if __name__ == '__main__':
    main()
