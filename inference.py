"""H-beam PatchCore Inference with k=3 + CLAHE enhancement.

Usage:
    # Single image
    python inference.py --spec 150x75 --group 1 --image /path/to/image.jpg
    
    # Folder scan  
    python inference.py --spec 150x75 --group 1 --folder /path/to/camera_1/ [--limit 100]
    
    # All groups for a spec
    python inference.py --spec 150x75 --all-groups --folder /path/to/folder/

Options:
    --k          kNN k value (default: 3)
    --clahe      Enable CLAHE preprocessing (default: on)
    --no-clahe   Disable CLAHE
    --threshold  Anomaly ratio threshold (default: 1.135, ~p95 of normal)
    --json       Output JSON format
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms

# ── Constants ──
IMAGE_W, IMAGE_H = 1920, 1200
TILE_SIZE = 256
TILE_STRIDE = 256
RESIZE = 224
OUTPUT_DIR = Path("~/patchcore/output").expanduser()
NAS_ROOT = Path("~/nas_storage").expanduser()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CAMERA_GROUPS = {
    1: {"cams": [1, 10], "desc": "상부 플랜지 내면"},
    2: {"cams": [2, 9],  "desc": "상부 필릿"},
    3: {"cams": [3, 8],  "desc": "플랜지 외면"},
    4: {"cams": [4, 7],  "desc": "하부 필릿"},
    5: {"cams": [5, 6],  "desc": "하부 플랜지 내면"},
}


class FeatureExtractor(nn.Module):
    """WideResNet50 layer2+layer3 feature extractor."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)
        
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        h = self.layer1(x)
        f2 = self.layer2(h)
        f3 = self.layer3(f2)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)  # (B, 1536, H, W)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (B, 1536)
        return features


def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def image_to_tiles(img_gray: np.ndarray) -> list:
    """Split 1920x1200 grayscale image into 256x256 tiles.
    Returns list of (tile_gray, row, col) tuples.
    """
    tiles = []
    rows = IMAGE_H // TILE_STRIDE
    cols = IMAGE_W // TILE_STRIDE
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * TILE_STRIDE, c * TILE_STRIDE
            y1, x1 = y0 + TILE_SIZE, x0 + TILE_SIZE
            if y1 <= IMAGE_H and x1 <= IMAGE_W:
                tile = img_gray[y0:y1, x0:x1]
                tiles.append((tile, r, c))
    return tiles


def tile_to_tensor(tile_gray: np.ndarray, use_clahe: bool = False) -> torch.Tensor:
    """Convert grayscale tile to 3ch normalized tensor."""
    if use_clahe:
        tile_gray = apply_clahe(tile_gray)
    # Resize to 224x224
    tile_resized = cv2.resize(tile_gray, (RESIZE, RESIZE))
    # Replicate to 3 channels
    tile_3ch = np.stack([tile_resized] * 3, axis=-1).astype(np.float32) / 255.0
    # ImageNet normalize
    tensor = torch.from_numpy(tile_3ch).permute(2, 0, 1)  # (3, 224, 224)
    for i in range(3):
        tensor[i] = (tensor[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    return tensor


def score_knn(features: np.ndarray, bank: np.ndarray, k: int = 3, 
              batch_size: int = 2048) -> np.ndarray:
    """Compute anomaly scores using k-NN distance to memory bank.
    
    k=1: min distance (original, FP-prone)
    k=3: mean of top-3 nearest distances (recommended, lower FP)
    """
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    bank_t = torch.from_numpy(bank).cuda()
    
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(features[i:i + batch_size]).cuda()
        dists = torch.cdist(batch.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
        
        if k == 1:
            min_dists, _ = dists.min(dim=1)
            scores[i:i + batch_size] = min_dists.cpu().numpy()
        else:
            # top-k smallest distances, mean
            topk_dists, _ = dists.topk(k, dim=1, largest=False)
            scores[i:i + batch_size] = topk_dists.mean(dim=1).cpu().numpy()
    
    return scores


def load_model(spec: str, group_id: int) -> np.ndarray:
    """Load memory bank for spec/group."""
    bank_path = OUTPUT_DIR / spec / f"group_{group_id}" / "memory_bank.npy"
    if not bank_path.exists():
        raise FileNotFoundError(f"Memory bank not found: {bank_path}")
    bank = np.load(str(bank_path))
    print(f"Loaded memory bank: {bank_path} ({bank.shape[0]} features, {bank.shape[1]}D)")
    return bank


def compute_normal_baseline(bank: np.ndarray, k: int = 3, n_sample: int = 500) -> float:
    """Estimate normal score baseline by scoring random bank samples against the bank.
    Uses leave-one-out: score each sampled bank member against rest.
    """
    n = min(n_sample, bank.shape[0])
    indices = np.random.RandomState(42).choice(bank.shape[0], n, replace=False)
    samples = bank[indices]
    
    bank_t = torch.from_numpy(bank).cuda()
    samples_t = torch.from_numpy(samples).cuda()
    
    dists = torch.cdist(samples_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    # For leave-one-out: the nearest is itself (dist=0), so skip it
    topk_dists, _ = dists.topk(k + 1, dim=1, largest=False)
    # Skip the first (self, dist≈0), take next k
    scores = topk_dists[:, 1:k+1].mean(dim=1).cpu().numpy()
    
    return float(np.mean(scores))


def inspect_image(img_path: str, bank: np.ndarray, extractor: FeatureExtractor,
                  k: int = 3, use_clahe: bool = True, threshold: float = 1.135):
    """Inspect a single image. Returns dict with per-tile scores."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"Cannot read image: {img_path}"}
    if img.shape != (IMAGE_H, IMAGE_W):
        img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    
    tiles = image_to_tiles(img)
    
    # Extract features
    tensors = torch.stack([tile_to_tensor(t, use_clahe) for t, _, _ in tiles])
    
    # Batch through extractor
    all_feats = []
    bs = 16
    for i in range(0, len(tensors), bs):
        feats = extractor(tensors[i:i+bs])
        all_feats.append(feats.cpu().numpy())
    features = np.concatenate(all_feats, axis=0)
    
    # Score
    scores = score_knn(features, bank, k=k)
    
    # Baseline
    baseline = compute_normal_baseline(bank, k=k)
    
    # Results
    results = []
    defect_count = 0
    for idx, ((tile, r, c), score) in enumerate(zip(tiles, scores)):
        ratio = float(score / baseline) if baseline > 0 else 0
        is_defect = ratio > threshold
        if is_defect:
            defect_count += 1
        results.append({
            "tile_idx": idx,
            "row": r, "col": c,
            "score": float(score),
            "ratio": ratio,
            "defect": is_defect,
        })
    
    return {
        "image": str(img_path),
        "baseline": baseline,
        "threshold": threshold,
        "k": k,
        "clahe": use_clahe,
        "total_tiles": len(tiles),
        "defect_tiles": defect_count,
        "verdict": "NG" if defect_count > 0 else "OK",
        "tiles": results,
    }


def main():
    parser = argparse.ArgumentParser(description="H-beam PatchCore Inference (k=3 + CLAHE)")
    parser.add_argument("--spec", required=True, help="Spec name (e.g., 150x75)")
    parser.add_argument("--group", type=int, default=1, help="Camera group (1-5)")
    parser.add_argument("--all-groups", action="store_true", help="Run all 5 groups")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--folder", help="Folder path to scan")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--k", type=int, default=3, help="kNN k value (default: 3)")
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    parser.add_argument("--no-clahe", dest="clahe", action="store_false")
    parser.add_argument("--threshold", type=float, default=1.135, help="Anomaly ratio threshold")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Either --image or --folder is required")
    
    groups = list(range(1, 6)) if args.all_groups else [args.group]
    
    print(f"Config: spec={args.spec}, k={args.k}, clahe={args.clahe}, threshold={args.threshold}")
    print(f"Groups: {groups}")
    
    extractor = FeatureExtractor()
    
    for gid in groups:
        print(f"\n{'='*60}")
        print(f"Group {gid}: {CAMERA_GROUPS[gid]['desc']} (cam {CAMERA_GROUPS[gid]['cams']})")
        print(f"{'='*60}")
        
        try:
            bank = load_model(args.spec, gid)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        
        if args.image:
            result = inspect_image(args.image, bank, extractor, 
                                   k=args.k, use_clahe=args.clahe, threshold=args.threshold)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                v = result["verdict"]
                print(f"  {result['image']}: {v} ({result['defect_tiles']}/{result['total_tiles']} defect tiles)")
                if v == "NG":
                    for t in result["tiles"]:
                        if t["defect"]:
                            print(f"    NG tile [{t['row']},{t['col']}] score={t['score']:.4f} ratio={t['ratio']:.3f}")
        
        elif args.folder:
            folder = Path(args.folder)
            images = sorted([f for f in folder.iterdir() 
                           if f.suffix.lower() in (".jpg", ".png", ".bmp")])
            if args.limit > 0:
                images = images[:args.limit]
            
            ok_count = ng_count = 0
            all_results = []
            
            for img_path in images:
                result = inspect_image(str(img_path), bank, extractor,
                                       k=args.k, use_clahe=args.clahe, threshold=args.threshold)
                if result.get("verdict") == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
                all_results.append(result)
                
                if not args.json:
                    v = result["verdict"]
                    sym = "✅" if v == "OK" else "❌"
                    print(f"  {sym} {img_path.name}: {v} ({result.get('defect_tiles',0)}/{result.get('total_tiles',0)})")
            
            print(f"\nSummary: {ok_count} OK, {ng_count} NG out of {len(images)} images")
            
            if args.json:
                print(json.dumps({"results": all_results, "ok": ok_count, "ng": ng_count}, 
                               indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
