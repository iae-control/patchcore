#!/usr/bin/env python3
"""Test synthetic defects on ALL 5 camera groups with matching camera images."""
import os, sys, random, json
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from src.patchcore import PatchCoreModel, FeatureExtractor
from src.config import TILE_SIZE, CAMERA_GROUPS

SPEC = '150x75'
OUTPUT_DIR = os.path.expanduser(f'~/patchcore/output/{SPEC}')
NAS_DIR = os.path.expanduser('~/nas_storage')
RESULT_DIR = os.path.expanduser('~/patchcore/test_all_groups')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE,TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

def find_img_for_cam(spec, cam_id):
    """Find an image for specific camera from NAS."""
    cam_name = f'camera_{cam_id}'
    for e in os.listdir(NAS_DIR):
        p = os.path.join(NAS_DIR, e)
        if not os.path.isdir(p): continue
        if len(e)==8 and e.isdigit():
            for s in os.listdir(p):
                if spec in s:
                    cp = os.path.join(p, s, cam_name)
                    if os.path.isdir(cp):
                        imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                        if len(imgs) > 200:
                            return os.path.join(cp, imgs[len(imgs)//2])
        elif spec in e:
            cp = os.path.join(p, cam_name)
            if os.path.isdir(cp):
                imgs = sorted([f for f in os.listdir(cp) if f.endswith('.jpg')])
                if len(imgs) > 200:
                    return os.path.join(cp, imgs[len(imgs)//2])
    return None

def tiles(img):
    h,w = img.shape[:2]; ts=TILE_SIZE; r=[]
    for y in range(0,h-ts+1,ts):
        for x in range(0,w-ts+1,ts): r.append((img[y:y+ts,x:x+ts],(x,y)))
    return r

def defect(t,k):
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

def feats(tile_list, ext):
    batch = torch.stack([T(cv2.cvtColor(t,cv2.COLOR_BGR2RGB)) for t in tile_list])
    with torch.no_grad(): return ext(batch).cpu().numpy()

def main():
    print("="*60)
    print("PatchCore Synthetic Defect Test - ALL 5 GROUPS")
    print("="*60)

    ext = FeatureExtractor('cuda')
    print("Extractor ready\n")
    
    dkinds = ['scratch','thick_scr','bright','spot','big_spot','crack','stain','multi_scr']
    all_results = {}

    for gid, gi in sorted(CAMERA_GROUPS.items()):
        gn = gi['desc']
        cams = gi['cams']
        cam_id = cams[0]  # Use first camera of the pair
        
        mb_path = os.path.join(OUTPUT_DIR, f'group_{gid}', 'memory_bank.npy')
        if not os.path.exists(mb_path):
            print(f"\n❌ Group {gid}: memory_bank not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"GROUP {gid}: {gn} (cam {cams[0]}+{cams[1]})")
        print(f"{'='*60}")
        
        # Find image for this camera
        img_path = find_img_for_cam(SPEC, cam_id)
        if not img_path:
            print(f"  ❌ No image found for camera_{cam_id}")
            continue
        print(f"  Image: {img_path}")
        
        img = cv2.imread(img_path)
        print(f"  Size: {img.shape}")
        
        # Save source image
        cv2.imwrite(os.path.join(RESULT_DIR, f'source_g{gid}.jpg'), img)
        
        all_t = tiles(img)
        print(f"  Tiles: {len(all_t)}")
        
        # Pick 3 test tiles (high/mid/low variance)
        vars_list = [np.var(t[0]) for t in all_t]
        idx = sorted(range(len(vars_list)), key=lambda i: vars_list[i], reverse=True)
        picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
        
        # Load model
        mdl = PatchCoreModel()
        mdl.load(mb_path)
        if mdl.memory_bank is None or (hasattr(mdl.memory_bank, "shape") and mdl.memory_bank.shape == ()):
            print(f"  ⚠️ Group {gid}: memory_bank is None/empty, skipping")
            continue
        print(f"  Memory bank: {mdl.memory_bank.shape}")
        
        # Load threshold
        th = None
        for tf in ['threshold.txt', 'self_val/threshold.txt']:
            tp = os.path.join(OUTPUT_DIR, f'group_{gid}', tf)
            if os.path.exists(tp):
                th = float(open(tp).read().strip())
                break
        if th:
            print(f"  Threshold: {th:.6f}")
        
        group_results = {'cam': cam_id, 'desc': gn, 'threshold': th, 'tiles': []}
        
        for ti, pi in enumerate(picks):
            tl, pos = all_t[pi]
            print(f"\n  Tile {ti+1} pos={pos} var={vars_list[pi]:.0f}")
            
            dts = [defect(tl, k) for k in dkinds]
            all_pieces = [tl] + dts
            f = feats(all_pieces, ext)
            sc = mdl.score(f)
            s0 = sc[0]
            
            print(f"  Original: {s0:.6f}")
            print(f"  {'Defect':<14} {'Score':>10} {'Ratio':>8} {'Det':>6}")
            print(f"  {'-'*40}")
            
            tile_result = {'pos': pos, 'var': round(float(vars_list[pi])), 
                          'original_score': round(float(s0), 6), 'defects': {}}
            
            detected = 0
            total = len(dkinds)
            for ki, dk in enumerate(dkinds):
                ds = sc[ki+1]
                rat = ds/s0 if s0 > 0 else 0
                det = (th and ds > th) or rat > 2.0
                dt = '✓' if det else ' '
                if det: detected += 1
                print(f"  {dk:<14} {ds:>10.4f} {rat:>7.2f}x {dt:>6}")
                tile_result['defects'][dk] = {
                    'score': round(float(ds), 4),
                    'ratio': round(float(rat), 2),
                    'detected': bool(det)
                }
            
            tile_result['detected'] = detected
            tile_result['total'] = total
            group_results['tiles'].append(tile_result)
            
            # Draw comparison image
            h,w = tl.shape[:2]; n=len(dts)+1; pad=5
            cw = n*(w+pad)+pad; ch = h+70
            canvas = np.ones((ch,cw,3), dtype=np.uint8)*240
            canvas[35:35+h, pad:pad+w] = tl
            cv2.putText(canvas, 'Original', (pad,20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            cv2.putText(canvas, f'{s0:.4f}', (pad,ch-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,100,0), 1)
            for j, dt in enumerate(dts):
                xo = pad+(j+1)*(w+pad)
                canvas[35:35+h, xo:xo+w] = dt
                cv2.putText(canvas, dkinds[j], (xo,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                ds = sc[j+1]
                col = (0,0,200) if (th and ds > th) else (0,100,0)
                cv2.putText(canvas, f'{ds:.4f}', (xo,ch-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            cv2.imwrite(os.path.join(RESULT_DIR, f'comp_g{gid}_t{ti}.jpg'), canvas)
        
        # Summary for this group
        total_det = sum(t['detected'] for t in group_results['tiles'])
        total_all = sum(t['total'] for t in group_results['tiles'])
        pct = 100*total_det/total_all if total_all else 0
        group_results['summary'] = {'detected': total_det, 'total': total_all, 'pct': round(pct,1)}
        print(f"\n  ▶ Group {gid} Summary: {total_det}/{total_all} ({pct:.1f}%)")
        all_results[f'group_{gid}'] = group_results
        
        # Baseline (normal tile distribution)
        nb = min(50, len(all_t))
        samp = [random.choice(all_t)[0] for _ in range(nb)]
        f = feats(samp, ext)
        bs = mdl.score(f)
        p90, p95, p99 = np.percentile(bs, [90,95,99])
        print(f"\n  Baseline N={nb}: Mean={np.mean(bs):.6f} Std={np.std(bs):.6f}")
        print(f"  Min={np.min(bs):.6f} Max={np.max(bs):.6f}")
        print(f"  P90={p90:.6f} P95={p95:.6f} P99={p99:.6f}")
        group_results['baseline'] = {
            'n': nb, 'mean': round(float(np.mean(bs)),6),
            'std': round(float(np.std(bs)),6),
            'min': round(float(np.min(bs)),6),
            'max': round(float(np.max(bs)),6),
            'p90': round(float(p90),6), 'p95': round(float(p95),6), 'p99': round(float(p99),6)
        }

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Group':<8} {'Description':<20} {'Det/Total':>12} {'Rate':>8}")
    print(f"{'-'*50}")
    grand_det = grand_tot = 0
    for gk in sorted(all_results):
        r = all_results[gk]
        s = r['summary']
        grand_det += s['detected']; grand_tot += s['total']
        print(f"{gk:<8} {r['desc']:<20} {s['detected']:>3}/{s['total']:<3}     {s['pct']:>5.1f}%")
    if grand_tot:
        print(f"{'TOTAL':<8} {'':20} {grand_det:>3}/{grand_tot:<3}     {100*grand_det/grand_tot:>5.1f}%")
    
    with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {RESULT_DIR}")

if __name__ == '__main__':
    main()
