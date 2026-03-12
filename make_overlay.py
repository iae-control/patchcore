#!/usr/bin/env python3
import os, sys, random, json
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/patchcore"))
from src.patchcore import FeatureExtractor
from src.config import CAMERA_GROUPS, TILE_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NAS_ROOT, OUTPUT_DIR, TRIM_HEAD
from torchvision import transforms

SPEC = "222x209"
GROUP = 1
OUT_DIR = Path(os.path.expanduser("~/patchcore/eval_results/overlay"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_test_image(spec, cam_id):
    for entry in sorted(os.listdir(str(NAS_ROOT)), reverse=True):
        p = NAS_ROOT / entry
        if not p.is_dir(): continue
        if len(entry) == 8 and entry.isdigit():
            for sub in sorted(os.listdir(str(p))):
                if spec in sub:
                    cam_dir = p / sub / f"camera_{cam_id}"
                    if cam_dir.is_dir():
                        imgs = sorted([f for f in os.listdir(str(cam_dir)) if f.endswith(".jpg")])
                        if len(imgs) > TRIM_HEAD + 100:
                            mid = len(imgs) // 2
                            return str(cam_dir / imgs[mid])
        elif spec in entry:
            cam_dir = p / f"camera_{cam_id}"
            if cam_dir.is_dir():
                imgs = sorted([f for f in os.listdir(str(cam_dir)) if f.endswith(".jpg")])
                if len(imgs) > TRIM_HEAD + 100:
                    mid = len(imgs) // 2
                    return str(cam_dir / imgs[mid])
    return None

def inject_defects(img):
    d = img.copy()
    h, w = d.shape[:2]
    # Big scratch
    cv2.line(d, (int(w*0.3), int(h*0.15)), (int(w*0.7), int(h*0.85)), 30, 3)
    # Spot cluster
    cx, cy = int(w*0.6), int(h*0.3)
    for _ in range(6):
        ox, oy = random.randint(-30, 30), random.randint(-30, 30)
        cv2.circle(d, (cx+ox, cy+oy), random.randint(5, 12), 25, -1)
    # Stain
    cv2.ellipse(d, (int(w*0.25), int(h*0.7)), (60, 30), 20, 0, 360, 40, -1)
    return d

def tiles_from_image(img):
    h, w = img.shape[:2]
    ts = TILE_SIZE
    result = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            tile = img[y:y+ts, x:x+ts]
            result.append((tile, (x, y)))
    return result

def main():
    cam_id = CAMERA_GROUPS[GROUP]["cams"][0]
    print("Finding test image for %s cam %d..." % (SPEC, cam_id))
    img_path = find_test_image(SPEC, cam_id)
    if not img_path:
        print("ERROR: No test image found"); return
    print("Found: %s" % img_path)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERROR: Cannot read image"); return
    print("Image shape: %s" % str(img.shape))
    
    # Inject defects
    defected = inject_defects(img)
    
    # Load model
    bank_path = OUTPUT_DIR / SPEC / ("group_%d" % GROUP) / "memory_bank.npy"
    if not bank_path.exists():
        print("ERROR: No model at %s" % bank_path); return
    bank = np.load(str(bank_path))
    print("Bank shape: %s" % str(bank.shape))
    
    # Extract features
    print("Loading extractor...")
    extractor = FeatureExtractor("cuda")
    
    tiles = tiles_from_image(defected)
    print("Tiles: %d" % len(tiles))
    
    tensors = []
    positions = []
    for tile, pos in tiles:
        tile_3ch = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tensors.append(T(cv2.cvtColor(tile_3ch, cv2.COLOR_BGR2RGB)))
        positions.append(pos)
    
    batch = torch.stack(tensors)
    all_feats = []
    bs = 64
    for i in range(0, len(batch), bs):
        with torch.no_grad():
            feats = extractor(batch[i:i+bs].cuda())
        all_feats.append(feats.cpu().numpy())
    features = np.concatenate(all_feats, axis=0)
    
    # Score
    bank_t = torch.from_numpy(bank).cuda()
    feat_t = torch.from_numpy(features).cuda()
    dists = torch.cdist(feat_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    topk = dists.topk(3, dim=1, largest=False)[0]
    scores = topk.mean(dim=1).cpu().numpy()
    
    # Baseline
    n_sample = min(500, bank.shape[0])
    idx = np.random.RandomState(42).choice(bank.shape[0], n_sample, replace=False)
    samples = bank[idx]
    samp_t = torch.from_numpy(samples).cuda()
    d2 = torch.cdist(samp_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    topk2 = d2.topk(4, dim=1, largest=False)[0]
    baseline_scores = topk2[:, 1:4].mean(dim=1).cpu().numpy()
    threshold = float(np.percentile(baseline_scores, 95)) * 1.2
    print("Threshold: %.4f" % threshold)
    
    # Create heatmap overlay
    h, w = defected.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    ts = TILE_SIZE
    
    for i, (x, y) in enumerate(positions):
        heatmap[y:y+ts, x:x+ts] += scores[i]
        count_map[y:y+ts, x:x+ts] += 1
    
    count_map[count_map == 0] = 1
    heatmap /= count_map
    
    # Normalize heatmap
    vmin = float(np.percentile(baseline_scores, 50))
    vmax = max(float(scores.max()), threshold * 1.5)
    heatmap_norm = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original
    defected_bgr = cv2.cvtColor(defected, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(defected_bgr, 0.5, heatmap_color, 0.5, 0)
    
    # Add threshold line and info
    n_defect_tiles = int((scores > threshold).sum())
    info = "%s/group_%d | Threshold: %.4f | Defect tiles: %d/%d" % (SPEC, GROUP, threshold, n_defect_tiles, len(scores))
    cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, "RED=high anomaly, BLUE=normal", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save side-by-side: original | defected | heatmap overlay
    original_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Resize for display
    scale = 0.5
    orig_sm = cv2.resize(original_bgr, None, fx=scale, fy=scale)
    def_sm = cv2.resize(defected_bgr, None, fx=scale, fy=scale)
    over_sm = cv2.resize(overlay, None, fx=scale, fy=scale)
    
    # Labels
    cv2.putText(orig_sm, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(def_sm, "+ Synthetic Defects", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(over_sm, "Anomaly Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    combined = np.hstack([orig_sm, def_sm, over_sm])
    
    out_path = str(OUT_DIR / "defect_overlay_222x209.jpg")
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("Saved: %s" % out_path)
    print("Image score (max tile): %.4f" % float(scores.max()))
    print("OVERLAY_DONE")

if __name__ == "__main__":
    main()
