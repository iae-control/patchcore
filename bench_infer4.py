import time, sys, os, numpy as np
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from concurrent.futures import ThreadPoolExecutor
import cv2

spec = "222x209"
SCALE = 0.5

# === Setup: 2-GPU DataParallel ===
bank_np = np.load("output/%s/group_1/memory_bank.npy" % spec)

# GPU 1 only (no training contention)
dev1 = torch.device("cuda:1")
bank1 = torch.from_numpy(bank_np).half().to(dev1)

model_single = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model_single = model_single.half().to(dev1).eval()

# DataParallel across both GPUs
model_dp = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model_dp = model_dp.half().cuda().eval()
model_dp = nn.DataParallel(model_dp, device_ids=[0,1])

# hooks for single GPU
layers1 = {}
def hook1(name):
    def fn(m, i, o):
        layers1[name] = o
    return fn
model_single.layer2.register_forward_hook(hook1("layer2"))
model_single.layer3.register_forward_hook(hook1("layer3"))

# hooks for DP (module is wrapped)
layers_dp = {}
def hookdp(name):
    def fn(m, i, o):
        layers_dp[name] = o
    return fn
model_dp.module.layer2.register_forward_hook(hookdp("layer2"))
model_dp.module.layer3.register_forward_hook(hookdp("layer3"))

mean1 = torch.tensor([0.485,0.456,0.406], device=dev1, dtype=torch.half).view(1,3,1,1)
std1 = torch.tensor([0.229,0.224,0.225], device=dev1, dtype=torch.half).view(1,3,1,1)
mean0 = torch.tensor([0.485,0.456,0.406], device="cuda:0", dtype=torch.half).view(1,3,1,1)
std0 = torch.tensor([0.229,0.224,0.225], device="cuda:0", dtype=torch.half).view(1,3,1,1)

folder = "/home/dk-sdd/nas_storage/20251229/20251229141938_222x209/camera_1"
all_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
imgs_paths = all_files[:100]
print("Test images: %d" % len(imgs_paths))

def load_resize(path):
    img = cv2.imread(path)
    if img is None: return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*SCALE), int(h*SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def gpu_tile(img_np, dev, mn, sd):
    t = torch.from_numpy(img_np).half().to(dev).permute(2,0,1).unsqueeze(0) / 255.0
    H, W = t.shape[2], t.shape[3]
    tile = 224
    t = t[:,:,:(H//tile)*tile,:(W//tile)*tile]
    p = t.unfold(2,tile,tile).unfold(3,tile,tile).contiguous().view(-1,3,tile,tile)
    return (p - mn) / sd

executor = ThreadPoolExecutor(max_workers=4)

for batch_n, label in [(10, "B10"), (20, "B20"), (50, "B50")]:
    # warmup
    img0 = load_resize(imgs_paths[0])
    t0 = gpu_tile(img0, dev1, mean1, std1)
    with torch.no_grad(): model_single(t0)
    torch.cuda.synchronize()

    total_time = 0
    total_count = 0
    for start in range(0, len(imgs_paths), batch_n):
        bp = imgs_paths[start:start+batch_n]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loaded = list(executor.map(load_resize, bp))
        all_tiles = [gpu_tile(img, dev1, mean1, std1) for img in loaded if img is not None]
        mega = torch.cat(all_tiles, dim=0)
        with torch.no_grad():
            model_single(mega)
            l2 = F.adaptive_avg_pool2d(layers1["layer2"], 1).flatten(1)
            l3 = F.adaptive_avg_pool2d(layers1["layer3"], 1).flatten(1)
            feats = torch.cat([l2, l3], dim=1)
            dists = torch.cdist(feats, bank1)
            scores = dists.min(dim=1).values
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t1
        total_count += len(bp)
    ms = (total_time / total_count) * 1000
    print("1GPU %s 50%%: %.1f ms/img | 10k: %.1fs (%.1fmin)" % (label, ms, ms*10, ms*10/60))

print("")
bank0 = torch.from_numpy(bank_np).half().to("cuda:0")
for batch_n, label in [(10, "B10"), (20, "B20"), (50, "B50")]:
    img0 = load_resize(imgs_paths[0])
    t0 = gpu_tile(img0, "cuda:0", mean0, std0)
    with torch.no_grad(): model_dp(t0)
    torch.cuda.synchronize()

    total_time = 0
    total_count = 0
    for start in range(0, len(imgs_paths), batch_n):
        bp = imgs_paths[start:start+batch_n]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loaded = list(executor.map(load_resize, bp))
        all_tiles = [gpu_tile(img, "cuda:0", mean0, std0) for img in loaded if img is not None]
        mega = torch.cat(all_tiles, dim=0)
        with torch.no_grad():
            model_dp(mega)
            l2 = F.adaptive_avg_pool2d(layers_dp["layer2"], 1).flatten(1)
            l3 = F.adaptive_avg_pool2d(layers_dp["layer3"], 1).flatten(1)
            feats = torch.cat([l2, l3], dim=1)
            dists = torch.cdist(feats, bank0)
            scores = dists.min(dim=1).values
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t1
        total_count += len(bp)
    ms = (total_time / total_count) * 1000
    print("2GPU-DP %s 50%%: %.1f ms/img | 10k: %.1fs (%.1fmin)" % (label, ms, ms*10, ms*10/60))

executor.shutdown()
print("BENCH_DONE")
