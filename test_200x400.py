#!/usr/bin/env python3
"""Synthetic defect test for 200x400 PatchCore models — same procedure as 150x75 test."""
import os, sys, random, json, time
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from src.patchcore import FeatureExtractor, greedy_coreset_selection
from src.config import TILE_SIZE, CAMERA_GROUPS, IMAGE_WIDTH, IMAGE_HEIGHT

SPEC = '200x400'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser(f'~/patchcore/test_results_{SPEC}')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def find_test_folders(spec, n=5):
    """Find folders for testing (different from training if possible)."""
    folders = []
    for entry in sorted(os.listdir(NAS_DIR)):
        p = os.path.join(NAS_DIR, entry)
        if not os.path.isdir(p):
            continue
        if len(entry) == 8 and entry.isdigit():
            for sub in sorted(os.listdir(p)):
                if spec in sub:
                    sp = os.path.join(p, sub)
                    cam1 = os.path.join(sp, 'camera_1')
                    if os.path.isdir(cam1):
                        imgs = [f for f in os.listdir(cam1) if f.endswith('.jpg')]
                        if len(imgs) > 200:
                            folders.append(sp)
        elif spec in entry:
            cam1 = os.path.join(p, 'camera_1')
            if os.path.isdir(cam1):
                imgs = [f for f in os.listdir(cam1) if f.endswith('.jpg')]
                if len(imgs) > 200:
                    folders.append(p)
    random.seed(42)
    random.shuffle(folders)
    return folders[:n]

def get_test_image(folder, cam_id):
    """Get middle image from a folder/camera."""
    cam_dir = os.path.join(folder, f'camera_{cam_id}')
    if not os.path.isdir(cam_dir):
        return None
    imgs = sorted([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
    if len(imgs) < 200:
        return None
    # Use middle image (trimming head/tail 100)
    mid = len(imgs) // 2
    return os.path.join(cam_dir, imgs[mid])

def tiles_from_image(img):
    """Split image into tiles."""
    h, w = img.shape[:2]
    ts = TILE_SIZE
    result = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            result.append((img[y:y+ts, x:x+ts], (x, y)))
    return result

def make_defects(tile, kind):
    """Create synthetic defect on tile."""
    h, w = tile.shape[:2]
    d = tile.copy()
    if kind == 'scratch':
        cv2.line(d, (int(w*.1), int(h*.1)), (int(w*.9), int(h*.9)), (30,30,30), 2)
    elif kind == 'thick_scratch':
        cv2.line(d, (int(w*.2), int(h*.1)), (int(w*.8), int(h*.9)), (20,20,20), 4)
    elif kind == 'bright_line':
        cv2.line(d, (int(w*.15), int(h*.5)), (int(w*.85), int(h*.5)), (220,220,220), 3)
    elif kind == 'spot':
        cv2.circle(d, (w//2, h//2), 8, (25,25,25), -1)
    elif kind == 'big_spot':
        cv2.circle(d, (w//2, h//2), 15, (30,30,30), -1)
    elif kind == 'crack':
        pts = [(int(w*(.1+.16*i)), int(h*(.4+.1*((-1)**i)))) for i in range(6)]
        for i in range(len(pts)-1):
            cv2.line(d, pts[i], pts[i+1], (25,25,25), 2)
    elif kind == 'stain':
        cv2.ellipse(d, (w//2, h//2), (18, 10), 30, 0, 360, (40,40,40), -1)
    elif kind == 'multi_scratch':
        for i in range(3):
            x = int(w*.15*(i+1))
            cv2.line(d, (x, 0), (x+5, h), (30,30,30), 2)
    return d

def extract_feats(tile_list, ext):
    """Extract features from tiles."""
    batch = torch.stack([T(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tile_list])
    with torch.no_grad():
        return ext(batch.cuda()).cpu().numpy()

def score_knn(features, bank, k=3):
    """Score features against memory bank with k-NN."""
    bank_t = torch.from_numpy(bank).cuda()
    feat_t = torch.from_numpy(features).cuda()
    dists = torch.cdist(feat_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    if k == 1:
        scores = dists.min(dim=1)[0]
    else:
        topk = dists.topk(k, dim=1, largest=False)[0]
        scores = topk.mean(dim=1)
    return scores.cpu().numpy()

def main():
    print("=" * 60)
    print(f"PatchCore Synthetic Defect Test — {SPEC}")
    print("=" * 60)
    
    # Find test folders
    folders = find_test_folders(SPEC, n=5)
    print(f"\nTest folders found: {len(folders)}")
    for f in folders:
        print(f"  {os.path.basename(f)}")
    
    if not folders:
        print("ERROR: No test folders found!")
        return
    
    # Load models
    ext = FeatureExtractor('cuda')
    print("\nFeature extractor loaded")
    
    models = {}
    for gid in range(1, 6):
        bank_path = os.path.join(OUTPUT_DIR, f'group_{gid}', 'memory_bank.npy')
        if os.path.exists(bank_path):
            bank = np.load(bank_path, allow_pickle=True)
            if not isinstance(bank, np.ndarray) or bank.ndim != 2 or bank.shape[0] < 10:
                print(f"  Group {gid}: INVALID bank, skipping")
                continue
            models[gid] = bank
            print(f"  Group {gid}: bank {bank.shape}")
        else:
            print(f"  Group {gid}: NOT FOUND")
    
    if not models:
        print("ERROR: No models found!")
        return
    
    defect_kinds = ['scratch', 'thick_scratch', 'bright_line', 'spot', 'big_spot', 'crack', 'stain', 'multi_scratch']
    
    # Results accumulator
    all_results = {}
    
    for gid, bank in models.items():
        cams = CAMERA_GROUPS[gid]['cams']
        desc = CAMERA_GROUPS[gid]['desc']
        print(f"\n{'='*60}")
        print(f"GROUP {gid}: {desc} (cams {cams})")
        print(f"{'='*60}")
        
        # Compute normal baseline from bank
        n_base = min(500, bank.shape[0])
        base_idx = np.random.RandomState(42).choice(bank.shape[0], n_base, replace=False)
        base_scores = score_knn(bank[base_idx], bank, k=3)
        # Leave-one-out: skip self (score≈0), take mean of non-zero
        base_nz = base_scores[base_scores > 0.01]
        baseline = float(np.mean(base_nz)) if len(base_nz) > 0 else float(np.mean(base_scores))
        p95 = float(np.percentile(base_nz, 95)) if len(base_nz) > 0 else baseline * 1.5
        print(f"  Baseline: mean={baseline:.4f}, p95={p95:.4f}")
        
        group_results = {'detected': 0, 'total': 0, 'details': []}
        
        for fi, folder in enumerate(folders[:3]):  # Test 3 folders
            cam_id = cams[0]  # Use first camera in group
            img_path = get_test_image(folder, cam_id)
            if img_path is None:
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            all_tiles = tiles_from_image(img)
            if not all_tiles:
                continue
            
            # Pick 3 tiles with varying variance
            variances = [np.var(t[0]) for t in all_tiles]
            sorted_idx = sorted(range(len(variances)), key=lambda i: variances[i], reverse=True)
            picks = [sorted_idx[0], sorted_idx[len(sorted_idx)//4], sorted_idx[len(sorted_idx)//2]]
            
            print(f"\n  Folder: {os.path.basename(folder)} cam_{cam_id}")
            print(f"  Image: {os.path.basename(img_path)}")
            print(f"  Tiles: {len(all_tiles)}, test tiles: {picks}")
            
            for ti, pi in enumerate(picks):
                tile, pos = all_tiles[pi]
                
                # Original + defects
                defect_tiles = [tile] + [make_defects(tile, k) for k in defect_kinds]
                feats = extract_feats(defect_tiles, ext)
                scores = score_knn(feats, bank, k=3)
                
                s0 = scores[0]
                print(f"\n    Tile {ti} pos={pos} var={variances[pi]:.0f}")
                print(f"    Original score: {s0:.4f} (baseline: {baseline:.4f})")
                print(f"    {'Defect':<16} {'Score':>8} {'Ratio':>8} {'vsP95':>8} {'Det':>5}")
                print(f"    {'-'*48}")
                
                for ki, dk in enumerate(defect_kinds):
                    ds = scores[ki + 1]
                    ratio = ds / baseline if baseline > 0 else 0
                    vs_p95 = ds / p95 if p95 > 0 else 0
                    det = ratio > 1.5 or ds > p95 * 1.2
                    group_results['total'] += 1
                    if det:
                        group_results['detected'] += 1
                    marker = 'O' if det else 'X'
                    print(f"    {dk:<16} {ds:>8.4f} {ratio:>7.2f}x {vs_p95:>7.2f}x   {marker}")
                    group_results['details'].append({
                        'folder': os.path.basename(folder),
                        'tile': ti, 'defect': dk,
                        'score': float(ds), 'ratio': float(ratio),
                        'detected': det
                    })
        
        det_rate = group_results['detected'] / group_results['total'] * 100 if group_results['total'] > 0 else 0
        print(f"\n  >>> Group {gid} detection rate: {group_results['detected']}/{group_results['total']} ({det_rate:.1f}%)")
        all_results[f'group_{gid}'] = group_results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_det = sum(r['detected'] for r in all_results.values())
    total_all = sum(r['total'] for r in all_results.values())
    for gk, r in all_results.items():
        rate = r['detected'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"  {gk}: {r['detected']}/{r['total']} ({rate:.1f}%)")
    
    overall = total_det / total_all * 100 if total_all > 0 else 0
    print(f"\n  OVERALL: {total_det}/{total_all} ({overall:.1f}%)")
    
    # Per-defect breakdown
    print(f"\n  Per-defect breakdown:")
    for dk in defect_kinds:
        det_dk = sum(1 for r in all_results.values() for d in r['details'] if d['defect'] == dk and d['detected'])
        tot_dk = sum(1 for r in all_results.values() for d in r['details'] if d['defect'] == dk)
        rate_dk = det_dk / tot_dk * 100 if tot_dk > 0 else 0
        print(f"    {dk:<16}: {det_dk}/{tot_dk} ({rate_dk:.1f}%)")
    
    # Save JSON
    with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULT_DIR}/results.json")

if __name__ == '__main__':
    main()
