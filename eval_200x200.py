#!/usr/bin/env python3
import os, sys, json, random, numpy as np, torch, cv2, time
sys.path.insert(0, os.path.expanduser("~/patchcore"))
from src.patchcore import FeatureExtractor
from src.config import CAMERA_GROUPS, TILE_SIZE
from torchvision import transforms

SPEC = "200x200"
TEST_DATE = "20251118"
NAS = os.path.expanduser("~/nas_storage")
OUT = os.path.expanduser("~/patchcore/output")
RESULT = os.path.expanduser(f"~/patchcore/eval_results/200x200")
os.makedirs(RESULT, exist_ok=True)

T = transforms.Compose([transforms.ToPILImage(), transforms.Resize((TILE_SIZE,TILE_SIZE)),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def get_test_folders():
    dp = os.path.join(NAS, TEST_DATE)
    folders = []
    for s in sorted(os.listdir(dp)):
        if SPEC in s:
            sp = os.path.join(dp, s)
            cam1 = os.path.join(sp, "camera_1")
            if os.path.isdir(cam1):
                imgs = [f for f in os.listdir(cam1) if f.endswith(".jpg")]
                if len(imgs) > 100:
                    folders.append(sp)
    return folders

def tiles(img):
    h,w = img.shape[:2]; ts=TILE_SIZE; r=[]
    for y in range(0,h-ts+1,ts):
        for x in range(0,w-ts+1,ts):
            r.append((img[y:y+ts,x:x+ts],(x,y)))
    return r

def feats(tile_list, ext):
    tensors = []
    for t,_ in tile_list:
        t3 = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) if len(t.shape)==2 else t
        tensors.append(T(cv2.cvtColor(t3, cv2.COLOR_BGR2RGB)))
    batch = torch.stack(tensors)
    all_f = []
    for i in range(0, len(batch), 64):
        with torch.no_grad():
            all_f.append(ext(batch[i:i+64].cuda()).cpu().numpy())
    return np.concatenate(all_f)

def score_knn(features, bank, k=3):
    bt = torch.from_numpy(bank).cuda()
    ft = torch.from_numpy(features).cuda()
    d = torch.cdist(ft.unsqueeze(0), bt.unsqueeze(0)).squeeze(0)
    return d.topk(k, dim=1, largest=False)[0].mean(dim=1).cpu().numpy()

def make_defect(img, kind):
    d = img.copy(); h,w = d.shape[:2]
    if kind=="scratch": cv2.line(d,(int(w*.1),int(h*.1)),(int(w*.9),int(h*.9)),30,2); return d,"긴 스크래치"
    if kind=="spots":
        for _ in range(5): cv2.circle(d,(w//2+random.randint(-50,50),h//2+random.randint(-50,50)),random.randint(5,12),25,-1)
        return d,"다크 스팟 클러스터"
    if kind=="crack":
        pts=[(int(w*(.1+.16*i)),int(h*(.3+.15*((-1)**i)))) for i in range(6)]
        for i in range(len(pts)-1): cv2.line(d,pts[i],pts[i+1],25,2)
        return d,"지그재그 크랙"
    if kind=="stain": cv2.ellipse(d,(w//3,h//2),(80,40),15,0,360,40,-1); return d,"큰 오염"
    if kind=="multi":
        for i in range(4): x=int(w*(.2+.15*i)); cv2.line(d,(x,int(h*.05)),(x+10,int(h*.95)),30,2)
        return d,"다중 스크래치"
    return d,"unknown"

def baseline(bank, k=3):
    n=min(500,bank.shape[0]); idx=np.random.RandomState(42).choice(bank.shape[0],n,replace=False)
    bt=torch.from_numpy(bank).cuda(); st=torch.from_numpy(bank[idx]).cuda()
    d=torch.cdist(st.unsqueeze(0),bt.unsqueeze(0)).squeeze(0)
    sc=d.topk(k+1,dim=1,largest=False)[0][:,1:k+1].mean(dim=1).cpu().numpy()
    return float(np.mean(sc)), float(np.percentile(sc,95))

print("Loading extractor...")
ext = FeatureExtractor("cuda")
folders = get_test_folders()
print(f"Test folders: {len(folders)}")

all_results = {"spec": SPEC, "test_date": TEST_DATE, "groups": {}}
defect_kinds = ["scratch","spots","crack","stain","multi"]

for gid in range(1,6):
    gi = CAMERA_GROUPS[gid]
    cam = gi["cams"][0]
    bp = os.path.join(OUT, SPEC, f"group_{gid}", "memory_bank.npy")
    if not os.path.exists(bp): continue
    bank = np.load(bp, allow_pickle=True)
    if not isinstance(bank, np.ndarray) or bank.ndim!=2 or bank.shape[0]<10:
        print(f"Group {gid}: invalid bank"); continue
    bm, bp95 = baseline(bank)
    thresh = bp95 * 1.2
    print(f"\nGroup {gid} ({gi['desc']}): bank={bank.shape}, baseline={bm:.4f}, thresh={thresh:.4f}")
    
    gr = {"images":[], "cam": cam, "desc": gi["desc"]}
    gdir = os.path.join(RESULT, f"group_{gid}")
    os.makedirs(gdir, exist_ok=True)
    
    # Normal images (3)
    for ni, folder in enumerate(folders[:3]):
        cam_dir = os.path.join(folder, f"camera_{cam}")
        imgs = sorted([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
        if len(imgs)<200: continue
        mid = len(imgs)//2
        path = os.path.join(cam_dir, imgs[mid])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        ts = tiles(img); ft = feats(ts, ext); sc = score_knn(ft, bank)
        ms = float(np.max(sc)); ratio = ms/bm
        det = ms > thresh
        print(f"  Normal {ni+1}: {imgs[mid]} score={ms:.4f} ratio={ratio:.2f}x {'FP!' if det else 'OK'}")
        cv2.imwrite(os.path.join(gdir, f"normal_{ni+1}.jpg"), img)
        gr["images"].append({"type":"normal","file":f"normal_{ni+1}.jpg","score":ms,"ratio":ratio,"defect":det,"source":os.path.basename(folder)})
    
    # Edge images (2) - index 40-60
    for ei, folder in enumerate(folders[:2]):
        cam_dir = os.path.join(folder, f"camera_{cam}")
        imgs = sorted([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
        if len(imgs)<100: continue
        idx = 40 + ei*10
        path = os.path.join(cam_dir, imgs[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        ts = tiles(img); ft = feats(ts, ext); sc = score_knn(ft, bank)
        ms = float(np.max(sc)); ratio = ms/bm
        det = ms > thresh
        print(f"  Edge {ei+1} (idx={idx}): score={ms:.4f} ratio={ratio:.2f}x {'FP!' if det else 'OK'}")
        cv2.imwrite(os.path.join(gdir, f"edge_{ei+1}.jpg"), img)
        gr["images"].append({"type":"edge","file":f"edge_{ei+1}.jpg","score":ms,"ratio":ratio,"defect":det,"img_idx":idx})
    
    # Synthetic defects (5)
    base_folder = folders[0] if folders else None
    if base_folder:
        cam_dir = os.path.join(base_folder, f"camera_{cam}")
        imgs = sorted([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
        base_path = os.path.join(cam_dir, imgs[len(imgs)//2+5])
        base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
        if base_img is not None:
            for di, dk in enumerate(defect_kinds):
                dimg, ddesc = make_defect(base_img, dk)
                ts = tiles(dimg); ft = feats(ts, ext); sc = score_knn(ft, bank)
                ms = float(np.max(sc)); ratio = ms/bm
                det = ms > thresh
                mark = "DETECTED" if det else "MISSED"
                print(f"  Defect {di+1} ({ddesc}): score={ms:.4f} ratio={ratio:.2f}x {mark}")
                cv2.imwrite(os.path.join(gdir, f"defect_{di+1}_{dk}.jpg"), dimg)
                gr["images"].append({"type":"synthetic","file":f"defect_{di+1}_{dk}.jpg","kind":dk,"desc":ddesc,"score":ms,"ratio":ratio,"defect":det})
    
    nfp = sum(1 for x in gr["images"] if x["type"]=="normal" and x["defect"])
    efp = sum(1 for x in gr["images"] if x["type"]=="edge" and x["defect"])
    sdet = sum(1 for x in gr["images"] if x["type"]=="synthetic" and x["defect"])
    stot = sum(1 for x in gr["images"] if x["type"]=="synthetic")
    gr["summary"] = {"normal_fp":nfp,"edge_fp":efp,"synth_det":sdet,"synth_total":stot}
    print(f"  >>> Normal FP={nfp}/3, Edge FP={efp}/2, Synth={sdet}/{stot}")
    all_results["groups"][f"group_{gid}"] = gr

with open(os.path.join(RESULT, "results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nDONE")
