#!/usr/bin/env python3
"""
Synthetic defect test for trained PatchCore models.
"""
import os, sys, random
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
RESULT_DIR = os.path.expanduser('~/patchcore/test_synthetic')
os.makedirs(RESULT_DIR, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE,TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

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
    print("="*60+"\nPatchCore Synthetic Defect Test\n"+"="*60)
    path,cam = find_img(SPEC)
    if not path: print("No image found!"); return
    print(f"Image: {path}\nCamera: {cam}")
    img = cv2.imread(path)
    print(f"Size: {img.shape}")
    cv2.imwrite(os.path.join(RESULT_DIR,'source.jpg'),img)
    all_t=tiles(img)
    print(f"Tiles: {len(all_t)}")
    vars = [np.var(t[0]) for t in all_t]
    idx = sorted(range(len(vars)), key=lambda i:vars[i], reverse=True)
    picks = [idx[0], idx[len(idx)//4], idx[len(idx)//2]]
    print(f"Test tiles: {picks}, vars: {[round(vars[i]) for i in picks]}")
    ext = FeatureExtractor('cuda'); print("Extractor ready")
    dkinds = ['scratch','thick_scr','bright','spot','big_spot','crack','stain','multi_scr']
    cn = int(cam.replace('camera_',''))
    avail = []
    for gid,gi in CAMERA_GROUPS.items():
        if cn in gi["cams"]:
            mb=os.path.join(OUTPUT_DIR,f'group_{gid}','memory_bank.npy')
            if os.path.exists(mb): avail.append((gid,gi,mb))
    if not avail:
        for gid in range(1,6):
            mb=os.path.join(OUTPUT_DIR,f'group_{gid}','memory_bank.npy')
            if os.path.exists(mb): avail.append((gid,CAMERA_GROUPS.get(gid,{}),mb))
    print(f"Models: {len(avail)}")
    for gid,gi,mb in avail:
        gn=gi.get('desc',f'Group {gid}')
        print(f"\n=== GEOUP {gid}: {gn} ===")
        mdl = PatchCoreModel(); mdl.load(mb)
        print(f"Memory bank: {mdl.memory_bank.shape}")
        th=None
        for tf in ['threshold.txt','self_val/threshold.txt']:
            tp=os.path.join(OUTPUT_DIR,f'group_{gid}',tf)
            if os.path.exists(tp): th=float(open(tp).read().strip()); break
        if th: print(f"Threshold: {th:.6f}")
        for ti,pi in enumerate(picks):
            tl,pos = all_t[pi]
            print(f"\n  Tile {ti+1} pos={pos} var={vars[pi]:.0f}")
            dts = [defect(tl,k) for k in dkinds]
            all_pieces = [tl] + dts
            f = feats(all_pieces, ext)
            sc = mdl.score(f)
            s0 = sc[0]
            print(f"  Original: {s0:.6f}")
            if th: print(f"  Threshold: {th:.6f}")
            print(f"  {'Defect':<14} {'Score':>10} {'Ratio':>8} {'Det':>6}")
            print(f"  {'-'*40}")
            for ki,dk in enumerate(dkinds):
                ds=sc[ki+1]; rat=ds/s0 if s0>0 else 0
                dt='✓' if (th and ds>th) or rat>2 else ' '
                print(f"  {dk:<14} {ds:>10.4f} {rat:>7.2f}x {dt:>6}")
            # Save comparison
            h,w=tl.shape[:2]; n=len(dts)+1; pad=5
            cw=n*(w+pad)+pad; ch=h+70
            canvas=np.ones((ch,cw,3),dtype=np.uint8)*240
            canvas[35:35+h,pad:pad+w]=tl
            cv2.putText(canvas,'Original',(pad,20),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)
            cv2.putText(canvas,f'{s0:.4f}',(pad,ch-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,100,0),1)
            for j,dt in enumerate(dts):
                xo=pad+(j+1)*(w+pad)
                canvas[35:35+h,xo:xo+w]=dt
                cv2.putText(canvas,dkinds[j],(xo,20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
                ds=sc[j+1]
                col=(0,0,200) if (th and ds>th) else (0,100,0)
                cv2.putText(canvas,f'{ds:.4f}',(xo,ch-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)
            cv2.imwrite(os.path.join(RESULT_DIR,f'comp_g{gid}_t{ti}.jpg'),canvas)
    # Baseline
    print(f"\n=== BASELINE: Normal tile distribution ===")
    nb=min(50,len(all_t))
    samp=[random.choice(all_t)[0] for _ in range(nb)]
    for gid,gi,mb in avail:
        mdl=PatchCoreModel(); mdl.load(mb)
        f=feats(samp,ext); bs=mdl.score(f)
        print(f"\n  Group {gid} ({gi.get('desc','')}):")
        print(f"  N={nb} Mean={np.mean(bs):.6f} Std={np.std(bs):.6f}")
        print(f"  Min={np.min(bs):.6f} Max={np.max(bs):.6f} Med={np.median(bs):.6f}")
        p=np.percentile(bs,[90,95,99])
        print(f"  P90={p[0]:.6f} P95={p[1]:.6f} P99={p[2]:.6f}")
    print(f"\nResults in: {RESULT_DIR}")

if __name__=='__main__': main()
