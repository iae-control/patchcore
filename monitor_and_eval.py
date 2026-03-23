#!/usr/bin/env python3
"""Monitor 210x205/group_5 completion, kill training, run evaluation."""
import json, os, sys, time, signal, subprocess, glob, random
import numpy as np
from pathlib import Path

SPEC = "210x205"
PATCHCORE_DIR = os.path.expanduser("~/patchcore")
OUTPUT_DIR = os.path.join(PATCHCORE_DIR, "output", SPEC)
PROGRESS_FILE = os.path.join(PATCHCORE_DIR, "output", "training_progress.json")
NAS_DIR = os.path.expanduser("~/nas_storage/20251102")
EVAL_DIR = os.path.join(PATCHCORE_DIR, "eval_results", SPEC)
TRAIN_DATE = "20251102"
TILE_SIZE = 256
K = 3
GROUPS = {"group_1": [1, 10],"group_2": [2, 9],"group_3": [3, 8],"group_4": [4, 7],"group_5": [5, 6]}
GROUP_NAMES = {"group_1": "flange_top_inner","group_2": "fillet_top","group_3": "flange_outer","group_4": "fillet_bottom","group_5": "flange_bottom_inner"}

def wait_for_completion():
    target = f"{SPEC}/group_5"
    print(f"Waiting for {target}...")
    while True:
        try:
            with open(PROGRESS_FILE) as f: data = json.load(f)
            if target in data.get("completed", []):
                print(f"COMPLETE: {target}"); return True
        except Exception as e: print(f"Error: {e}")
        time.sleep(10)

def kill_training():
    print("Killing training...")
    subprocess.run(["pkill", "-f", "train.py"], capture_output=True)
    time.sleep(3)
    r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True)
    if r.returncode != 0: print("Training killed OK")
    else: subprocess.run(["pkill", "-9", "-f", "train.py"]); time.sleep(2)

def load_memory_bank(group):
    p = os.path.join(OUTPUT_DIR, group, "memory_bank.npy")
    if not os.path.exists(p): return None
    b = np.load(p); sz = os.path.getsize(p)
    if sz < 1000: print(f"  {group} bank too small"); return None
    print(f"  Loaded {group}: {b.shape}, {sz//1048576}MB"); return b

def get_test_images(group, cameras):
    all_f = sorted(glob.glob(os.path.join(NAS_DIR, f"*_{SPEC}")))
    if not all_f: return []
    test_f = all_f[15:] if len(all_f) > 15 else all_f[-3:]
    print(f"  Using {len(test_f)} test folders")
    imgs = []
    for fd in test_f:
        for c in cameras:
            cd = os.path.join(fd, f"camera_{c}")
            if os.path.exists(cd):
                imgs += sorted(glob.glob(os.path.join(cd, "*.bmp")))
                imgs += sorted(glob.glob(os.path.join(cd, "*.png")))
    return imgs

def extract_tiles(img_path):
    from PIL import Image
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape
    tiles = []
    for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            tiles.append(arr[y:y+TILE_SIZE, x:x+TILE_SIZE].flatten())
    return np.array(tiles) if tiles else None

def score_image(tiles, bank):
    if tiles is None or len(tiles) == 0: return 0.0
    if len(tiles) > 50: tiles = tiles[np.random.choice(len(tiles), 50, replace=False)]
    scores = []
    for t in tiles:
        d = np.linalg.norm(bank - t, axis=1)
        scores.append(float(np.mean(np.sort(d)[:K])))
    return max(scores)

def create_synthetic_defect(img_path, dtype):
    from PIL import Image, ImageDraw
    img = Image.open(img_path).convert("L")
    draw = ImageDraw.Draw(img); w, h = img.size
    if dtype == "scratch":
        draw.line([(random.randint(0,w//4),random.randint(0,h//4)),(random.randint(w*3//4,w-1),random.randint(h*3//4,h-1))], fill=0, width=3)
    elif dtype == "spots":
        for _ in range(10): cx,cy=random.randint(w//4,w*3//4),random.randint(h//4,h*3//4);r=random.randint(3,8);draw.ellipse([cx-r,cy-r,cx+r,cy+r],fill=0)
    elif dtype == "crack":
        pts = [(random.randint(0,w//3),h//2)]
        for _ in range(8): px,py=pts[-1];pts.append((px+random.randint(20,w//8),py+random.randint(-30,30)))
        draw.line(pts, fill=0, width=2)
    elif dtype == "stain": r=min(w,h)//6;draw.ellipse([w//2-r,h//2-r,w//2+r,h//2+r],fill=60)
    elif dtype == "multi":
        for _ in range(5): x1,y1=random.randint(0,w-50),random.randint(0,h-50);draw.line([(x1,y1),(x1+random.randint(30,80),y1+random.randint(-20,20))],fill=0,width=2)
    return img

def evaluate_group(group, cams, bank):
    print(f"\n=== {group}: {GROUP_NAMES[group]} (cams {cams}) ===")
    images = get_test_images(group, cams)
    if not images: print("  No images!"); return None
    print(f"  Found {len(images)} images")
    bl = []
    for p in random.sample(images, min(10, len(images))): bl.append(score_image(extract_tiles(p), bank))
    baseline = np.mean(bl); thresh = baseline * 1.3
    print(f"  Baseline: {baseline:.4f}, Thresh: {thresh:.4f}")
    res = {"group":group,"desc":GROUP_NAMES[group],"baseline":float(baseline),"threshold":float(thresh),"images":[]}
    # Normal 3
    nfp=0
    for i,p in enumerate(random.sample(images, min(3,len(images)))):
        s=score_image(extract_tiles(p),bank);d=s>thresh;nfp+=int(d)
        print(f"  Normal {i+1}: {s:.4f} {s/baseline:.2f}x {'FP' if d else 'OK'}")
        res["images"].append({"type":"normal","score":float(s),"ratio":float(s/baseline),"defect":bool(d)})
    # Edge 2
    efp=0; eidx=[40,50] if len(images)>60 else [len(images)//3,len(images)//2]
    for ix in eidx:
        if ix<len(images):
            s=score_image(extract_tiles(images[ix]),bank);d=s>thresh;efp+=int(d)
            print(f"  Edge idx={ix}: {s:.4f} {s/baseline:.2f}x {'FP' if d else 'OK'}")
            res["images"].append({"type":"edge","score":float(s),"ratio":float(s/baseline),"defect":bool(d),"img_idx":ix,"desc":f"index {ix}"})
    # Synth 5
    sd=0; base_img=random.choice(images)
    for dt in ["scratch","spots","crack","stain","multi"]:
        simg=create_synthetic_defect(base_img,dt);tp=f//tmp/s{group}_{dt}.bmp";simg.save(tp)
        s=score_image(extract_tiles(tp),bank);d=s>thresh;sd+=int(d)
        print(f"  Synth {dt}: {s:.4f} {s/baseline:.2f}x {'DET' if d else 'MISS'}")
        os.makedirs(os.path.join(EVAL_DIR,group),exist_ok=True);simg.save(os.path.join(EVAL_DIR,group,f"synth_{dt}.bmp"))
        res["images"].append({"type":"synthetic","kind":dt,"score":float(s),"ratio":float(s/baseline),"defect":bool(d)});os.remove(tp)
    res["summary"]={"normal_fp":nfp,"edge_fp":efp,"synth_det":sd,"synth_total":5}
    print(f"  Summary: NFP={nfp}/3 EFP={efp}/2 Synth={sd}/5"); return res

def main():
    print(f"=== Monitor & Eval: {SPEC} ===")
    wait_for_completion()
    kill_training()
    os.makedirs(EVAL_DIR, exist_ok=True)
    all_r = {"spec":SPEC,"train_date":TRAIN_DATE,"groups":{}}
    for g,cs in GROUPS.items():
        b=load_memory_bank(g)
        if b is None: print(f"Skip {g}"); all_r["groups"][g]={"error":"no bank"}; continue
        r=evaluate_group(g,cs,b)
        if r: all_r["groups"][g]=r
    rp=os.path.join(EVAL_DIR,"results.json")
    with open(rp,"w") as f: json.dump(all_r,f,indent=2,ensure_ascii=False)
    print(f"Saved: {rp}")
    print("DONE")

if __name__ == "__main__": main()
