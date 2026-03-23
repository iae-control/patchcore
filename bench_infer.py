import time, sys, os, glob, numpy as np
sys.path.insert(0, "/home/dk-sdd/patchcore")
os.chdir("/home/dk-sdd/patchcore")
from src.config import CAMERA_GROUPS
import torch
from torchvision import transforms, models
from PIL import Image
from scipy.spatial.distance import cdist
spec = "222x209"
group = 1
cam = CAMERA_GROUPS[group]["cams"][0]
t0 = time.perf_counter()
bank = np.load("output/%s/group_%d/memory_bank.npy" % (spec, group))
device = torch.device("cuda:0")
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model = model.to(device).eval()
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
print("Bank shape: %s" % str(bank.shape))
nas = "/home/dk-sdd/nas_storage"
folders = sorted(glob.glob("%s/*/*_%s/camera_%d" % (nas, spec, cam)))
imgs = []
for f in folders[:5]:
    for p in sorted(glob.glob(f + "/*.jpg"))[:50]:
        imgs.append(p)
    if len(imgs) >= 100:
        break
imgs = imgs[:100]
print("Test images: %d" % len(imgs))
img0 = Image.open(imgs[0]).convert("RGB")
with torch.no_grad():
    x = tfm(img0).unsqueeze(0).to(device)
    model(x)
print("Warmup done")
times = []
ntiles = 0
for p in imgs:
    t1 = time.perf_counter()
    img = Image.open(p).convert("RGB")
    w, h = img.size
    tile_h, tile_w = 224, 224
    tiles = []
    for y in range(0, h - tile_h + 1, tile_h):
        for x in range(0, w - tile_w + 1, tile_w):
            tiles.append(img.crop((x, y, x+tile_w, y+tile_h)))
    ntiles = len(tiles)
    batch = torch.stack([tfm(t) for t in tiles]).to(device)
    with torch.no_grad():
        model(batch)
        feats = torch.cat([torch.nn.functional.adaptive_avg_pool2d(layers["layer2"], 1).flatten(1), torch.nn.functional.adaptive_avg_pool2d(layers["layer3"], 1).flatten(1)], dim=1).cpu().numpy()
    dists = cdist(feats, bank, metric="euclidean")
    scores = np.min(dists, axis=1)
    max_score = float(np.max(scores))
    t2 = time.perf_counter()
    times.append(t2 - t1)
times = np.array(times)
print("")
print("=== Benchmark (%d images) ===" % len(imgs))
print("Tiles per image: %d" % ntiles)
print("Mean: %.1f ms/img" % (times.mean()*1000))
print("Median: %.1f ms/img" % (np.median(times)*1000))
print("Min: %.1f ms/img" % (times.min()*1000))
print("Max: %.1f ms/img" % (times.max()*1000))
print("Std: %.1f ms" % (times.std()*1000))
print("")
print("10000 images: %.0fs = %.1fmin" % (times.mean()*10000, times.mean()*10000/60))
print("BENCH_DONE")
