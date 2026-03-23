import time, sys, os, numpy as np, threading
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
import torch, torch.nn.functional as F
from torchvision import models
from concurrent.futures import ThreadPoolExecutor
import cv2

spec = "222x209"
SCALE = 0.5
BATCH_IMG = 20

bank_np = np.load("output/%s/group_1/memory_bank.npy" % spec)

def make_engine(dev_id):
    dev = torch.device("cuda:%d" % dev_id)
    bank = torch.from_numpy(bank_np).half().to(dev)
    mdl = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    mdl = mdl.half().to(dev).eval()
    lyrs = {}
    def hk(name):
        def fn(m, i, o): lyrs[name] = o
        return fn
    mdl.layer2.register_forward_hook(hk("layer2"))
    mdl.layer3.register_forward_hook(hk("layer3"))
    mn = torch.tensor([0.485,0.456,0.406], device=dev, dtype=torch.half).view(1,3,1,1)
    sd = torch.tensor([0.229,0.224,0.225], device=dev, dtype=torch.half).view(1,3,1,1)
    return {"dev": dev, "bank": bank, "model": mdl, "layers": lyrs, "mean": mn, "std": sd}

def gpu_tile(img_np, eng):
    t = torch.from_numpy(img_np).half().to(eng["dev"]).permute(2,0,1).unsqueeze(0) / 255.0
    H, W = t.shape[2], t.shape[3]
    tile = 224
    t = t[:,:,:(H//tile)*tile,:(W//tile)*tile]
    p = t.unfold(2,tile,tile).unfold(3,tile,tile).contiguous().view(-1,3,tile,tile)
    return (p - eng["mean"]) / eng["std"]

def infer_batch(paths, eng):
    loaded = list(io_pool.map(load_resize, paths))
    all_tiles = [gpu_tile(img, eng) for img in loaded if img is not None]
    if not all_tiles:
        return []
    mega = torch.cat(all_tiles, dim=0)
    with torch.no_grad():
        eng["model"](mega)
        l2 = F.adaptive_avg_pool2d(eng["layers"]["layer2"], 1).flatten(1)
        l3 = F.adaptive_avg_pool2d(eng["layers"]["layer3"], 1).flatten(1)
        feats = torch.cat([l2, l3], dim=1)
        dists = torch.cdist(feats, eng["bank"])
        scores = dists.min(dim=1).values
    torch.cuda.synchronize(eng["dev"])
    return scores.cpu().numpy()

def load_resize(path):
    img = cv2.imread(path)
    if img is None: return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*SCALE), int(h*SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Loading 2 GPU engines...")
eng0 = make_engine(0)
eng1 = make_engine(1)
print("Engines ready")

folder = "/home/dk-sdd/nas_storage/20251229/20251229141938_222x209/camera_1"
all_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
imgs_paths = all_files[:100]
print("Test images: %d" % len(imgs_paths))

io_pool = ThreadPoolExecutor(max_workers=8)

# warmup both GPUs
img0 = load_resize(imgs_paths[0])
t0a = gpu_tile(img0, eng0)
t0b = gpu_tile(img0, eng1)
with torch.no_grad():
    eng0["model"](t0a)
    eng1["model"](t0b)
torch.cuda.synchronize()
print("Warmup done")

# === 2GPU parallel: split images 50/50, run on separate threads ===
def run_half(paths, eng):
    results = []
    for start in range(0, len(paths), BATCH_IMG):
        bp = paths[start:start+BATCH_IMG]
        r = infer_batch(bp, eng)
        results.append(r)
    return results

half = len(imgs_paths) // 2
paths_gpu0 = imgs_paths[:half]
paths_gpu1 = imgs_paths[half:]

torch.cuda.synchronize()
t_start = time.perf_counter()

th0 = threading.Thread(target=run_half, args=(paths_gpu0, eng0))
th1 = threading.Thread(target=run_half, args=(paths_gpu1, eng1))
th0.start()
th1.start()
th0.join()
th1.join()

torch.cuda.synchronize()
t_total = time.perf_counter() - t_start

ms_per_img = (t_total / len(imgs_paths)) * 1000
print("")
print("=== 2GPU Parallel (50/50 split, B%d, 50%% resize) ===" % BATCH_IMG)
print("Total: %.2fs for %d images" % (t_total, len(imgs_paths)))
print("Mean: %.1f ms/img" % ms_per_img)
print("10000 images: %.1fs (%.1fmin)" % (ms_per_img*10, ms_per_img*10/60))
print("BENCH_DONE")

io_pool.shutdown()
