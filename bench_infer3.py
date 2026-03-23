import time, sys, os, numpy as np
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
import torch, torch.nn.functional as F
from torchvision import transforms, models
from concurrent.futures import ThreadPoolExecutor
import cv2

spec = "222x209"
device = torch.device("cuda:1")
bank_np = np.load("output/%s/group_1/memory_bank.npy" % spec)
bank_gpu = torch.from_numpy(bank_np).half().to(device)
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model = model.half().to(device).eval()
layers = {}
def hook(name):
    def fn(m, i, o):
        layers[name] = o
    return fn
model.layer2.register_forward_hook(hook("layer2"))
model.layer3.register_forward_hook(hook("layer3"))
mean = torch.tensor([0.485,0.456,0.406], device=device, dtype=torch.half).view(1,3,1,1)
std = torch.tensor([0.229,0.224,0.225], device=device, dtype=torch.half).view(1,3,1,1)

folder = "/home/dk-sdd/nas_storage/20251229/20251229141938_222x209/camera_1"
all_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])
imgs_paths = all_files[:100]
print("Test images: %d" % len(imgs_paths))

def load_and_resize(path, scale):
    img = cv2.imread(path)
    if img is None:
        return None
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def gpu_tile(img_np):
    t = torch.from_numpy(img_np).half().to(device).permute(2,0,1).unsqueeze(0) / 255.0
    H, W = t.shape[2], t.shape[3]
    tile = 224
    H2 = (H // tile) * tile
    W2 = (W // tile) * tile
    t = t[:,:,:H2,:W2]
    patches = t.unfold(2, tile, tile).unfold(3, tile, tile)
    patches = patches.contiguous().view(-1, 3, tile, tile)
    patches = (patches - mean) / std
    return patches

executor = ThreadPoolExecutor(max_workers=4)
BATCH_IMG = 10

for scale, label in [(1.0, "100%"), (0.75, "75%"), (0.5, "50%"), (0.375, "37.5%")]:
    # warmup
    img0 = load_and_resize(imgs_paths[0], scale)
    t0 = gpu_tile(img0)
    print("%s: image %dx%d -> %d tiles" % (label, img0.shape[1], img0.shape[0], t0.shape[0]))
    with torch.no_grad():
        model(t0)
    torch.cuda.synchronize(device)

    total_time = 0
    total_count = 0
    for start in range(0, len(imgs_paths), BATCH_IMG):
        bp = imgs_paths[start:start+BATCH_IMG]
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        loaded = list(executor.map(lambda p: load_and_resize(p, scale), bp))
        all_tiles = []
        for img in loaded:
            if img is not None:
                all_tiles.append(gpu_tile(img))
        mega = torch.cat(all_tiles, dim=0)
        with torch.no_grad():
            model(mega)
            l2 = F.adaptive_avg_pool2d(layers["layer2"], 1).flatten(1)
            l3 = F.adaptive_avg_pool2d(layers["layer3"], 1).flatten(1)
            feats = torch.cat([l2, l3], dim=1)
            dists = torch.cdist(feats, bank_gpu)
            scores = dists.min(dim=1).values
        torch.cuda.synchronize(device)
        total_time += time.perf_counter() - t1
        total_count += len(bp)
    ms = (total_time / total_count) * 1000
    print("  -> %.1f ms/img | 10k: %.1fs (%.1fmin)" % (ms, ms*10, ms*10/60))
    print("")

executor.shutdown()
print("BENCH_DONE")
