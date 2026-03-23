import time, sys, os, numpy as np
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
import torch, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

spec = "222x209"
device = torch.device("cuda:1")
t0 = time.perf_counter()
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
tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
t_load = time.perf_counter() - t0
print("Model load: %.2fs" % t_load)
print("Bank shape: %s" % str(bank_gpu.shape))

# use known folder directly
folder = "/home/dk-sdd/nas_storage/20251229/20251229141938_222x209/camera_1"
imgs = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])[:100]
print("Test images: %d" % len(imgs))

tile_h, tile_w = 224, 224
img0 = Image.open(imgs[0]).convert("RGB")
w0, h0 = img0.size
tile_coords = []
for y in range(0, h0 - tile_h + 1, tile_h):
    for x in range(0, w0 - tile_w + 1, tile_w):
        tile_coords.append((x, y, x+tile_w, y+tile_h))
print("Tiles per image: %d" % len(tile_coords))

with torch.no_grad():
    dummy = torch.randn(len(tile_coords), 3, 224, 224, device=device, dtype=torch.half)
    model(dummy)
    del dummy
torch.cuda.synchronize(device)
print("Warmup done")

times = []
for p in imgs:
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    img = Image.open(p).convert("RGB")
    tiles = [tfm(img.crop(c)) for c in tile_coords]
    batch = torch.stack(tiles).half().to(device)
    with torch.no_grad():
        model(batch)
        l2 = F.adaptive_avg_pool2d(layers["layer2"], 1).flatten(1)
        l3 = F.adaptive_avg_pool2d(layers["layer3"], 1).flatten(1)
        feats = torch.cat([l2, l3], dim=1)
        dists = torch.cdist(feats, bank_gpu)
        min_dists = dists.min(dim=1).values
        max_score = min_dists.max().item()
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    times.append(t2 - t1)

times = np.array(times)
print("")
print("=== OPTIMIZED Benchmark (%d imgs, GPU:1 FP16) ===" % len(imgs))
print("Mean: %.1f ms/img" % (times.mean()*1000))
print("Median: %.1f ms/img" % (np.median(times)*1000))
print("Min: %.1f ms/img" % (times.min()*1000))
print("Max: %.1f ms/img" % (times.max()*1000))
print("Std: %.1f ms" % (times.std()*1000))
print("")
print("10000 images: %.1fs = %.1fmin" % (times.mean()*10000, times.mean()*10000/60))
print("Speedup vs old(1044ms): %.1fx" % (1.044/times.mean()))
print("BENCH_DONE")
