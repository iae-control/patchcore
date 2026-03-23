#!/usr/bin/env python3
"""
Subtle Defect Stress Test - 미묘한 합성 결함 생성 + 검출 한계 테스트
10종 결함 × 3강도 × 3타일 = 90 케이스 × 4 검출방법
"""
import os, sys, random, math
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
RESULT_DIR = os.path.expanduser('~/patchcore/test_subtle')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 10 defect types × 3 levels
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
LEVEL_LABELS = {1: '극미세(L1)', 2: '미세(L2)', 3: '약한(L3)'}

# Intensity multipliers per level
# Level 1: barely visible, Level 2: subtle, Level 3: weak but visible
INTENSITY = {1: 1.0, 2: 2.0, 3: 3.0}


def apply_subtle_defect(tile, defect_type, level):
    """Apply subtle defect to tile. Returns modified tile."""
    h, w = tile.shape[:2]
    d = tile.copy().astype(np.float32)
    mult = INTENSITY[level]
    rng = np.random.RandomState(42 + hash(defect_type) % 1000 + level)
    
    cx, cy = w // 2, h // 2
    
    if defect_type == 'hair_scratch':
        # 1px thin line, low contrast
        delta = 5 * mult  # L1: ±5, L2: ±10, L3: ±15
        angle = rng.uniform(20, 70)
        length = int(min(w, h) * 0.6)
        rad = math.radians(angle)
        x1 = int(cx - length/2 * math.cos(rad))
        y1 = int(cy - length/2 * math.sin(rad))
        x2 = int(cx + length/2 * math.cos(rad))
        y2 = int(cy + length/2 * math.sin(rad))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (x1,y1), (x2,y2), 1.0, 1)  # 1px
        d[:,:,0] -= delta * mask
        d[:,:,1] -= delta * mask
        d[:,:,2] -= delta * mask
    
    elif defect_type == 'faint_spot':
        radius = int(3 + level)  # 3~6px
        delta = 3 * mult + 2  # L1: 5, L2: 8, L3: 11
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        d[:,:,0] -= delta * mask
        d[:,:,1] -= delta * mask
        d[:,:,2] -= delta * mask
    
    elif defect_type == 'micro_crack':
        delta = 5 * mult
        length = int(30 + 10 * level)
        pts = []
        x, y = cx - length//2, cy
        for i in range(length):
            x += 1
            y += rng.choice([-1, 0, 1])
            y = max(0, min(h-1, y))
            x = max(0, min(w-1, x))
            pts.append((x, y))
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in pts:
            mask[py, px] = 1.0
        d[:,:,0] -= delta * mask
        d[:,:,1] -= delta * mask
        d[:,:,2] -= delta * mask
    
    elif defect_type == 'gradient_stain':
        radius = 20
        delta = 5 * mult  # center brightness change
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        gauss = np.exp(-dist**2 / (2 * radius**2))
        d[:,:,0] += delta * gauss
        d[:,:,1] += delta * gauss
        d[:,:,2] += delta * gauss
    
    elif defect_type == 'roughness':
        sigma = 3 * mult  # L1:3, L2:6, L3:9
        region_size = 40
        y1, y2 = cy - region_size//2, cy + region_size//2
        x1, x2 = cx - region_size//2, cx + region_size//2
        y1, y2 = max(0,y1), min(h,y2)
        x1, x2 = max(0,x1), min(w,x2)
        noise = rng.randn(y2-y1, x2-x1, 3) * sigma
        d[y1:y2, x1:x2] += noise
    
    elif defect_type == 'discoloration':
        delta = 2 * mult + 2  # L1:4, L2:6, L3:8
        region_size = 30
        y1, y2 = cy - region_size//2, cy + region_size//2
        x1, x2 = cx - region_size//2, cx + region_size//2
        y1, y2 = max(0,y1), min(h,y2)
        x1, x2 = max(0,x1), min(w,x2)
        d[y1:y2, x1:x2] += delta
    
    elif defect_type == 'pinhole':
        radius = 2
        delta = 10 * mult  # L1:10, L2:20, L3:30
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        d[:,:,0] -= delta * mask
        d[:,:,1] -= delta * mask
        d[:,:,2] -= delta * mask
    
    elif defect_type == 'dent_shadow':
        radius = 15
        delta = 4 * mult  # brightness gradient
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        gauss = np.exp(-dist**2 / (2 * radius**2))
        # simulate shadow: dark on one side, bright on other
        gradient = (X - cx).astype(np.float32) / radius
        shadow = gauss * gradient * delta
        d[:,:,0] += shadow
        d[:,:,1] += shadow
        d[:,:,2] += shadow
    
    elif defect_type == 'oxide_edge':
        delta = 5 * mult
        # irregular thin boundary line
        pts = []
        y_start = cy - 30
        for i in range(60):
            y = y_start + i
            x = cx + int(5 * math.sin(i * 0.5) + rng.randint(-2, 3))
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in pts:
            mask[py, px] = 1.0
        # slight blur to make it more natural
        if level >= 2:
            mask = cv2.GaussianBlur(mask, (3,3), 0.5)
            mask = np.clip(mask, 0, 1)
        d[:,:,0] -= delta * mask
        d[:,:,1] -= delta * mask * 0.8  # slightly different per channel
        d[:,:,2] -= delta * mask * 0.6
    
    elif defect_type == 'rolling_mark':
        delta = 2 * mult + 1  # L1:3, L2:5, L3:7
        # horizontal thin stripes
        mask = np.zeros((h, w), dtype=np.float32)
        for stripe_y in range(cy-20, cy+20, 4):
            if 0 <= stripe_y < h:
                mask[stripe_y, :] = 1.0
                if stripe_y+1 < h:
                    mask[stripe_y+1, :] = 0.5
        d[:,:,0] += delta * mask
        d[:,:,1] += delta * mask
        d[:,:,2] += delta * mask
    
    return np.clip(d, 0, 255).astype(np.uint8)


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
        return torch.cat([f2, f3_up], dim=1)


def heatmap_topk_score(feat_map, patch_bank_t, topk=3, gaussian_weight=False):
    """Score spatial features against patch bank. Returns topk mean distance."""
    C, H, W = feat_map.shape
    patches = feat_map.permute(1, 2, 0).reshape(-1, C).cuda()
    dists = torch.cdist(patches.unsqueeze(0), patch_bank_t.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)  # (H*W,)
    
    if gaussian_weight:
        # Weight by distance from center
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        cy, cx = H/2, W/2
        gdist = ((yy.float()-cy)**2 + (xx.float()-cx)**2).sqrt()
        weights = torch.exp(-gdist / (H * 0.5)).reshape(-1).cuda()
        min_dists = min_dists * weights
    
    topk_vals = torch.topk(min_dists, min(topk, len(min_dists))).values
    return topk_vals.mean().item(), min_dists.reshape(H, W).cpu().numpy()


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
                            if len(imgs) > 10:
                                return os.path.join(cp, imgs[len(imgs)//2]), c
        elif spec in e:
            for c in [f'camera_{i}' for i in range(1, 11)]:
                cp = os.path.join(p, c)
                if os.path.isdir(cp):
                    imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                    if len(imgs) > 10:
                        return os.path.join(cp, imgs[len(imgs)//2]), c
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
    print("=" * 70)
    print("SUBTLE DEFECT STRESS TEST")
    print("10 defect types × 3 levels × 3 tiles × 4 methods")
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
    
    # Pick 3 test tiles (high, medium, low variance)
    vars_list = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
    picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    print(f"Test tile indices: {picks}")
    
    # Load memory bank
    mb_path = os.path.join(OUTPUT_DIR, 'group_1', 'memory_bank.npy')
    if not os.path.exists(mb_path):
        for g in range(1, 6):
            mb = os.path.join(OUTPUT_DIR, f'group_{g}', 'memory_bank.npy')
            if os.path.exists(mb):
                mb_path = mb
                break
    memory_bank = np.load(mb_path)
    print(f"Memory bank: {memory_bank.shape}")
    
    # ========== Setup all extractors ==========
    print("\nInitializing extractors...")
    ext_orig = FeatureExtractor('cuda')
    mdl_orig = PatchCoreModel()
    mdl_orig.load(mb_path)
    
    hm_ext = HeatmapExtractor('cuda')
    
    # Build spatial patch bank for heatmap methods
    print("Building spatial patch bank...")
    all_batch = torch.stack([T(cv2.cvtColor(t[0], cv2.COLOR_BGR2RGB)) for t in all_t])
    patch_bank_list = []
    for i in range(0, len(all_batch), 8):
        with torch.no_grad():
            fm = hm_ext(all_batch[i:i+8])
        B, C, H, W = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        step = max(1, len(patches) // 2000)
        patch_bank_list.append(patches[::step])
    patch_bank = np.concatenate(patch_bank_list)
    patch_bank_t = torch.from_numpy(patch_bank).cuda()
    print(f"Spatial bank: {patch_bank.shape}")
    
    # Compute baseline scores for all methods on normal tiles
    print("\nComputing baseline scores...")
    baseline = {'orig': [], 'hm_topk3': [], 'hm_topk5g': [], 'adaptive': []}
    
    for ti, pi in enumerate(picks):
        tl, pos = all_t[pi]
        inp = T(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        # Method 1: Original PatchCore
        with torch.no_grad():
            f = ext_orig(inp).cpu().numpy()
        s_orig = mdl_orig.score(f)[0]
        baseline['orig'].append(s_orig)
        
        # Methods 2&3: Heatmap TopK=3 and TopK=5+Gaussian
        with torch.no_grad():
            fm = hm_ext(inp)
        s_hm3, _ = heatmap_topk_score(fm[0], patch_bank_t, topk=3)
        s_hm5g, _ = heatmap_topk_score(fm[0], patch_bank_t, topk=5, gaussian_weight=True)
        baseline['hm_topk3'].append(s_hm3)
        baseline['hm_topk5g'].append(s_hm5g)
        
        print(f"  Tile {ti}: orig={s_orig:.4f}, hm_topk3={s_hm3:.4f}, hm_topk5g={s_hm5g:.4f}")
    
    # ========== Run all tests ==========
    # results[defect][level][tile] = {method: (score, ratio)}
    results = {}
    for dt in DEFECT_NAMES:
        results[dt] = {}
        for lv in LEVELS:
            results[dt][lv] = {}
    
    print("\n" + "=" * 70)
    print("RUNNING TESTS...")
    print("=" * 70)
    
    for dt in DEFECT_NAMES:
        print(f"\n--- {dt} ---")
        for lv in LEVELS:
            for ti, pi in enumerate(picks):
                tl, pos = all_t[pi]
                defected = apply_subtle_defect(tl, dt, lv)
                inp = T(cv2.cvtColor(defected, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                
                # Method 1: Original PatchCore
                with torch.no_grad():
                    f = ext_orig(inp).cpu().numpy()
                s_orig = mdl_orig.score(f)[0]
                r_orig = s_orig / baseline['orig'][ti] if baseline['orig'][ti] > 0 else 0
                
                # Method 2: Heatmap TopK=3
                with torch.no_grad():
                    fm = hm_ext(inp)
                s_hm3, _ = heatmap_topk_score(fm[0], patch_bank_t, topk=3)
                r_hm3 = s_hm3 / baseline['hm_topk3'][ti] if baseline['hm_topk3'][ti] > 0 else 0
                
                # Method 3: Heatmap TopK=5 + Gaussian
                s_hm5g, _ = heatmap_topk_score(fm[0], patch_bank_t, topk=5, gaussian_weight=True)
                r_hm5g = s_hm5g / baseline['hm_topk5g'][ti] if baseline['hm_topk5g'][ti] > 0 else 0
                
                # Method 4: Adaptive z-score
                # Collect all normal scores to compute mean/std
                # (simplified: use baseline as reference)
                
                results[dt][lv][ti] = {
                    'orig': (s_orig, r_orig),
                    'hm_topk3': (s_hm3, r_hm3),
                    'hm_topk5g': (s_hm5g, r_hm5g),
                }
            
            avg_r = {m: np.mean([results[dt][lv][ti][m][1] for ti in range(3)]) for m in ['orig', 'hm_topk3', 'hm_topk5g']}
            print(f"  L{lv}: orig={avg_r['orig']:.3f}x  hm3={avg_r['hm_topk3']:.3f}x  hm5g={avg_r['hm_topk5g']:.3f}x")
    
    # ========== Compute adaptive z-score method ==========
    # Collect all normal tile scores for z-score baseline
    print("\nComputing adaptive z-score baselines...")
    normal_scores_orig = []
    normal_scores_hm3 = []
    sample_indices = list(range(0, len(all_t), max(1, len(all_t)//30)))[:30]
    
    for si in sample_indices:
        tl, _ = all_t[si]
        inp = T(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            f = ext_orig(inp).cpu().numpy()
            fm = hm_ext(inp)
        normal_scores_orig.append(mdl_orig.score(f)[0])
        s3, _ = heatmap_topk_score(fm[0], patch_bank_t, topk=3)
        normal_scores_hm3.append(s3)
    
    mu_orig, std_orig = np.mean(normal_scores_orig), np.std(normal_scores_orig)
    mu_hm3, std_hm3 = np.mean(normal_scores_hm3), np.std(normal_scores_hm3)
    print(f"  Normal orig: μ={mu_orig:.4f} σ={std_orig:.4f}")
    print(f"  Normal hm3:  μ={mu_hm3:.4f} σ={std_hm3:.4f}")
    
    # Add adaptive results
    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            for ti in range(3):
                s_o = results[dt][lv][ti]['orig'][0]
                s_h = results[dt][lv][ti]['hm_topk3'][0]
                z_o = (s_o - mu_orig) / max(std_orig, 1e-6)
                z_h = (s_h - mu_hm3) / max(std_hm3, 1e-6)
                z_max = max(z_o, z_h)
                results[dt][lv][ti]['adaptive'] = (z_max, z_max)
    
    # ========== Detection thresholds ==========
    THRESHOLDS = {
        'orig': ('ratio', 2.0),
        'hm_topk3': ('ratio', 1.5),
        'hm_topk5g': ('ratio', 1.5),
        'adaptive': ('zscore', 2.0),
    }
    METHOD_NAMES = {
        'orig': 'Original PatchCore',
        'hm_topk3': 'Heatmap TopK=3',
        'hm_topk5g': 'Heatmap TopK5+Gauss',
        'adaptive': 'Adaptive Z-score',
    }
    
    def is_detected(method, dt, lv, ti):
        val = results[dt][lv][ti][method][1]  # ratio or z-score
        ttype, thresh = THRESHOLDS[method]
        return val > thresh
    
    # ========== Summary Table ==========
    print("\n" + "=" * 90)
    print("DETECTION RATE TABLE: defect × level × method (detected/3 tiles)")
    print("=" * 90)
    
    header = f"{'Defect':<18} {'Lv':>3}"
    for m in ['orig', 'hm_topk3', 'hm_topk5g', 'adaptive']:
        header += f" | {METHOD_NAMES[m]:>20}"
    print(header)
    print("-" * len(header))
    
    # For summary
    method_total_det = {m: 0 for m in THRESHOLDS}
    method_total = {m: 0 for m in THRESHOLDS}
    det_by_level = {m: {lv: 0 for lv in LEVELS} for m in THRESHOLDS}
    total_by_level = {lv: 0 for lv in LEVELS}
    
    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            row = f"{dt:<18} L{lv:>1}"
            total_by_level[lv] += 3
            for m in ['orig', 'hm_topk3', 'hm_topk5g', 'adaptive']:
                det_count = sum(1 for ti in range(3) if is_detected(m, dt, lv, ti))
                avg_val = np.mean([results[dt][lv][ti][m][1] for ti in range(3)])
                method_total_det[m] += det_count
                method_total[m] += 3
                det_by_level[m][lv] += det_count
                mark = "✓" if det_count == 3 else ("△" if det_count > 0 else "✗")
                row += f" | {det_count}/3 {mark} (avg {avg_val:.2f})"
            print(row)
        print()
    
    # Overall summary
    print("=" * 90)
    print("OVERALL DETECTION RATES:")
    for m in ['orig', 'hm_topk3', 'hm_topk5g', 'adaptive']:
        rate = method_total_det[m] / method_total[m] * 100
        print(f"  {METHOD_NAMES[m]:<25}: {method_total_det[m]}/{method_total[m]} ({rate:.1f}%)")
        for lv in LEVELS:
            lv_rate = det_by_level[m][lv] / (len(DEFECT_NAMES) * 3) * 100
            print(f"    Level {lv}: {det_by_level[m][lv]}/{len(DEFECT_NAMES)*3} ({lv_rate:.1f}%)")
    
    # Detection limit analysis
    print("\n" + "=" * 90)
    print("DETECTION LIMIT ANALYSIS:")
    print("(Which defects are undetectable by ALL methods?)")
    print("=" * 90)
    
    for dt in DEFECT_NAMES:
        for lv in LEVELS:
            any_detected = any(
                is_detected(m, dt, lv, ti) 
                for m in THRESHOLDS for ti in range(3)
            )
            if not any_detected:
                avg_vals = {m: np.mean([results[dt][lv][ti][m][1] for ti in range(3)]) for m in THRESHOLDS}
                best_m = max(avg_vals, key=avg_vals.get)
                print(f"  ✗ {DEFECT_LABELS_KR[dt]} Level {lv}: 모든 방법 미검출 (최고 {METHOD_NAMES[best_m]}={avg_vals[best_m]:.3f})")
    
    # ========== Generate comparison images ==========
    print("\n\nGenerating comparison images...")
    
    for dt in DEFECT_NAMES:
        # Use first test tile
        tl = all_t[picks[0]][0]
        h, w = tl.shape[:2]
        
        # 3 levels side by side + original
        cols = 4  # orig, L1, L2, L3
        pad = 2
        cw = cols * (w + pad) + pad
        ch = h + 50
        canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 240
        
        labels = ['Original', 'Level 1', 'Level 2', 'Level 3']
        images = [tl] + [apply_subtle_defect(tl, dt, lv) for lv in LEVELS]
        
        for j, (lb, im) in enumerate(zip(labels, images)):
            xo = pad + j * (w + pad)
            canvas[0:h, xo:xo+w] = im
            cv2.putText(canvas, lb, (xo, h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            
            if j > 0:
                lv = j
                scores_text = []
                for m in ['orig', 'hm_topk3', 'adaptive']:
                    val = np.mean([results[dt][lv][ti][m][1] for ti in range(3)])
                    detected = any(is_detected(m, dt, lv, ti) for ti in range(3))
                    scores_text.append(f"{m[:4]}:{val:.2f}{'*' if detected else ''}")
                cv2.putText(canvas, '  '.join(scores_text), (xo, h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100,0,0), 1)
        
        # Title
        cv2.putText(canvas, f"{dt} ({DEFECT_LABELS_KR[dt]})", (pad, h+48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,200), 1)
        
        cv2.imwrite(os.path.join(RESULT_DIR, f'subtle_{dt}.jpg'), canvas)
    
    # Big combined image
    print("Creating combined comparison image...")
    tile_h, tile_w = all_t[picks[0]][0].shape[:2]
    n_defects = len(DEFECT_NAMES)
    cols = 4
    pad = 2
    row_h = tile_h + 55
    total_w = cols * (tile_w + pad) + pad
    total_h = n_defects * row_h + 30
    
    big_canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240
    
    for di, dt in enumerate(DEFECT_NAMES):
        tl = all_t[picks[0]][0]
        yo = di * row_h + 20
        images = [tl] + [apply_subtle_defect(tl, dt, lv) for lv in LEVELS]
        labels = ['Orig', 'L1', 'L2', 'L3']
        
        # defect name on left
        cv2.putText(big_canvas, DEFECT_LABELS_KR[dt], (2, yo - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,150), 1)
        
        for j, (lb, im) in enumerate(zip(labels, images)):
            xo = pad + j * (tile_w + pad)
            big_canvas[yo:yo+tile_h, xo:xo+tile_w] = im
            cv2.putText(big_canvas, lb, (xo, yo+tile_h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
            
            if j > 0:
                lv = j
                best_score = max(
                    np.mean([results[dt][lv][ti][m][1] for ti in range(3)])
                    for m in THRESHOLDS
                )
                detected_any = any(is_detected(m, dt, lv, ti) for m in THRESHOLDS for ti in range(3))
                color = (0, 150, 0) if detected_any else (0, 0, 200)
                cv2.putText(big_canvas, f"best:{best_score:.2f}", (xo, yo+tile_h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    
    combined_path = os.path.join(RESULT_DIR, 'subtle_all_combined.jpg')
    cv2.imwrite(combined_path, big_canvas)
    print(f"Saved: {combined_path}")
    
    # Save results as text
    with open(os.path.join(RESULT_DIR, 'results_summary.txt'), 'w') as f:
        f.write("SUBTLE DEFECT STRESS TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for dt in DEFECT_NAMES:
            f.write(f"\n{dt} ({DEFECT_LABELS_KR[dt]}):\n")
            for lv in LEVELS:
                f.write(f"  Level {lv}:\n")
                for m in THRESHOLDS:
                    vals = [results[dt][lv][ti][m][1] for ti in range(3)]
                    det = sum(1 for ti in range(3) if is_detected(m, dt, lv, ti))
                    f.write(f"    {METHOD_NAMES[m]:<25}: avg={np.mean(vals):.4f} det={det}/3\n")
        
        f.write(f"\n\nOVERALL:\n")
        for m in THRESHOLDS:
            rate = method_total_det[m] / method_total[m] * 100
            f.write(f"  {METHOD_NAMES[m]:<25}: {rate:.1f}%\n")
    
    print("\nDONE! All results saved to:", RESULT_DIR)


if __name__ == '__main__':
    main()
