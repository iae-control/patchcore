import time, sys, os, numpy as np
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
import torch, torch.nn.functional as F
from torchvision import models
from concurrent.futures import ThreadPoolExecutor
import cv2, faiss

spec = "222x209"
SCALE = 0.5
BATCH_IMG = 20
device = torch.device("cuda:1")

bank_np = np.load("output/%s/group_1/memory_bank.npy" % spec).astype(np.float32)
print("Bank shape: %s" % str(bank_np.shape))

# FAISS GPU index
res = faiss.StandardGpuResources()
res.setTempMemory(64 * 1024 * 1024)
dim = bank_np.shape[1]
index_flat = faiss.IndexFlatL2(dim)
index_gpu = faiss.index_cpu_to_gpu(res, 1, index_flat)
index_gpu.add(bank_np)
print("FAISS GPU index built: %d vectors" % index_gpu.ntotal)

# torch.cdist bank for comparison
bank_gpu = torch.from_numpy(bank_np).half().to(device)

model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model = model.half().to(device).eval()
layers = {}
def hook(name):
    def fn(m, i, o): layers[name] = o
    return fn
model.layer2.register_forward_hook(hook("layer2"))
model.layer3.register_forward_hook(hook("layer3"))
mn = torch.tensor([0.485,0.456,0.406], device=device, dtype=torch.half).view(1,3,1,1)
sd = torch.tensor([0.229,0.224,0.225], device=device, dtype=torch.half).view(1,3,1,1)

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

def gpu_tile(img_np):
    t = torch.from_numpy(img_np).half().to(device).permute(2,0,1).unsqueeze(0) / 255.0
    H, W = t.shape[2], t.shape[3]
    tile = 224
    t = t[:,:,:(H//tile)*tile,:(W//tile)*tile]
    p = t.unfold(2,tile,tile).unfold(3,tile,tile).contiguous().view(-1,3,tile,tile)
    return (p - mn) / sd

executor = ThreadPoolExecutor(max_workers=8)

# warmup
img0 = load_resize(imgs_paths[0])
t0 = gpu_tile(img0)
with torch.no_grad(): model(t0)
torch.cuda.synchronize()
print("Warmup done")

for knn_method, knn_label in [("cdist", "torch.cdist"), ("faiss", "FAISS GPU")]:
    total_time = 0
    total_count = 0
    for start in range(0, len(imgs_paths), BATCH_IMG):
        bp = imgs_paths[start:start+BATCH_IMG]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loaded = list(executor.map(load_resize, bp))
        all_tiles = [gpu_tile(img) for img in loaded if img is not None]
        mega = torch.cat(all_tiles, dim=0)
        with torch.no_grad():
            model(mega)
            l2 = F.adaptive_avg_pool2d(layers["layer2"], 1).flatten(1)
            l3 = F.adaptive_avg_pool2d(layers["layer3"], 1).flatten(1)
            feats = torch.cat([l2, l3], dim=1)
            if knn_method == "cdist":
                dists = torch.cdist(feats, bank_gpu)
                scores = dists.min(dim=1).values
            else:
                feats_np = feats.float().cpu().numpy()
                D, I = index_gpu.search(feats_np, 1)
                scores = D[:, 0]
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t1
        total_count += len(bp)
    ms = (total_time / total_count) * 1000
    print("%s B%d 50%%: %.1f ms/img | 10k: %.1fs (%.1fmin)" % (knn_label, BATCH_IMG, ms, ms*10, ms*10/60))

executor.shutdown()
print("BENCH_DONE")
