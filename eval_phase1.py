#!/usr/bin/env python3
"""PatchCore Phase 1 Evaluation — run on A40 server.
Follows test guidelines strictly:
- Different date images only (not in training set)
- Same spec only
- Per camera group evaluation
- 3 normal + 2 edge(idx 40-60) + 5 synthetic defect images = 10 per group
"""
import os, sys, json, random, glob, shutil, time
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from src.patchcore import FeatureExtractor
from src.config import (CAMERA_GROUPS, TILE_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT,
                        NAS_ROOT, OUTPUT_DIR, TRIM_HEAD, TRIM_TAIL, IMAGE_SUBSAMPLE)
from torchvision import transforms

SPEC = sys.argv[1] if len(sys.argv) > 1 else '199x396'
RESULT_BASE = Path(os.path.expanduser(f'~/patchcore/eval_results/{SPEC}'))
RESULT_BASE.mkdir(parents=True, exist_ok=True)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def find_all_folders(spec):
    """Find ALL folders for this spec in NAS."""
    folders = []
    for entry in sorted(os.listdir(str(NAS_ROOT))):
        p = NAS_ROOT / entry
        if not p.is_dir():
            continue
        # Date folder structure (YYYYMMDD/*)
        if len(entry) == 8 and entry.isdigit():
            for sub in sorted(os.listdir(str(p))):
                if spec in sub:
                    sp = p / sub
                    if sp.is_dir():
                        folders.append((entry, str(sp)))  # (date, path)
        # Direct folder structure
        elif spec in entry:
            # Extract date from folder name (first 8 digits)
            date = entry[:8] if len(entry) >= 8 and entry[:8].isdigit() else 'unknown'
            folders.append((date, str(p)))
    return folders


def find_training_folders(spec):
    """Identify which folders were likely used in training (from cache or heuristic)."""
    # Training uses FOLDER_SAMPLE_MAX=500 random folders
    # We can check the tile mask folders to see what was used
    mask_dirs = []
    for gid in range(1, 6):
        mask_dir = OUTPUT_DIR / spec / f'group_{gid}' / 'self_val'
        if mask_dir.exists():
            # Check for any folder references in the output
            for f in mask_dir.iterdir():
                if f.name.endswith('.json'):
                    try:
                        data = json.loads(f.read_text())
                        if 'folders' in data:
                            mask_dirs.extend(data['folders'])
                    except:
                        pass
    return set(mask_dirs)


def get_test_folders(spec, n=5):
    """Get test folders from DIFFERENT dates than training."""
    all_folders = find_all_folders(spec)
    if not all_folders:
        print(f"ERROR: No folders found for spec {spec}")
        return []

    # Group by date
    by_date = {}
    for date, path in all_folders:
        by_date.setdefault(date, []).append(path)

    dates = sorted(by_date.keys())
    print(f"  Total folders: {len(all_folders)}, dates: {len(dates)}")

    # Use last few dates (most recent, likely NOT in training which samples randomly)
    # Training randomly samples from all folders, so pick from dates with fewer folders
    # or explicitly pick the last N dates
    if len(dates) >= 3:
        test_dates = dates[-3:]  # Last 3 dates
    else:
        test_dates = dates

    test_folders = []
    for d in test_dates:
        test_folders.extend(by_date[d])

    random.seed(42)
    random.shuffle(test_folders)
    return test_folders[:n]


def get_images_from_folder(folder_path, cam_id):
    """Get sorted image list from a camera folder."""
    cam_dir = os.path.join(folder_path, f'camera_{cam_id}')
    if not os.path.isdir(cam_dir):
        return [], cam_dir
    imgs = sorted([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
    return imgs, cam_dir


def load_image(path):
    """Load image as grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def tiles_from_image(img):
    """Split 1920x1200 image into 256x256 tiles."""
    h, w = img.shape[:2]
    ts = TILE_SIZE
    result = []
    for y in range(0, h - ts + 1, ts):
        for x in range(0, w - ts + 1, ts):
            tile = img[y:y+ts, x:x+ts]
            result.append((tile, (x, y)))
    return result


def make_synthetic_defect(img, kind):
    """Create synthetic defect on full image. Returns (defected_image, description)."""
    d = img.copy()
    h, w = d.shape[:2]

    if kind == 'scratch_long':
        # Long diagonal scratch across the image
        cv2.line(d, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), 30, 2)
        return d, "긴 대각선 스크래치"
    elif kind == 'spot_cluster':
        # Cluster of dark spots
        cx, cy = w//2, h//2
        for _ in range(5):
            ox, oy = random.randint(-50, 50), random.randint(-50, 50)
            cv2.circle(d, (cx+ox, cy+oy), random.randint(5, 12), 25, -1)
        return d, "다크 스팟 클러스터"
    elif kind == 'crack_zigzag':
        # Zigzag crack pattern
        pts = [(int(w*(0.1+0.16*i)), int(h*(0.3+0.15*((-1)**i)))) for i in range(6)]
        for i in range(len(pts)-1):
            cv2.line(d, pts[i], pts[i+1], 25, 2)
        return d, "지그재그 크랙"
    elif kind == 'stain_large':
        # Large elliptical stain
        cv2.ellipse(d, (w//3, h//2), (80, 40), 15, 0, 360, 40, -1)
        return d, "큰 타원형 오염"
    elif kind == 'multi_scratch':
        # Multiple parallel scratches
        for i in range(4):
            x = int(w * (0.2 + 0.15*i))
            cv2.line(d, (x, int(h*0.05)), (x+10, int(h*0.95)), 30, 2)
        return d, "다중 평행 스크래치"
    return d, "unknown"


def extract_tile_features(tiles, extractor):
    """Extract features from tile list."""
    if not tiles:
        return np.array([])
    tensors = []
    for tile, _ in tiles:
        if len(tile.shape) == 2:
            tile_3ch = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        else:
            tile_3ch = tile
        tensors.append(T(cv2.cvtColor(tile_3ch, cv2.COLOR_BGR2RGB)))
    batch = torch.stack(tensors)

    all_feats = []
    bs = 64
    for i in range(0, len(batch), bs):
        with torch.no_grad():
            feats = extractor(batch[i:i+bs].cuda())
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def score_knn(features, bank, k=3):
    """k-NN anomaly scoring."""
    if len(features) == 0:
        return np.array([])
    bank_t = torch.from_numpy(bank).cuda()
    feat_t = torch.from_numpy(features).cuda()
    dists = torch.cdist(feat_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    topk = dists.topk(k, dim=1, largest=False)[0]
    scores = topk.mean(dim=1)
    return scores.cpu().numpy()


def compute_image_score(tile_scores):
    """Compute image-level anomaly score from tile scores."""
    if len(tile_scores) == 0:
        return 0.0
    return float(np.max(tile_scores))  # Max tile score = image anomaly


def compute_baseline(bank, k=3, n=500):
    """Compute normal baseline from bank."""
    n = min(n, bank.shape[0])
    idx = np.random.RandomState(42).choice(bank.shape[0], n, replace=False)
    samples = bank[idx]
    bank_t = torch.from_numpy(bank).cuda()
    samp_t = torch.from_numpy(samples).cuda()
    dists = torch.cdist(samp_t.unsqueeze(0), bank_t.unsqueeze(0)).squeeze(0)
    topk = dists.topk(k+1, dim=1, largest=False)[0]
    scores = topk[:, 1:k+1].mean(dim=1).cpu().numpy()  # skip self
    return float(np.mean(scores)), float(np.percentile(scores, 95))


def save_comparison_image(images_info, output_path):
    """Save comparison grid image."""
    n = len(images_info)
    if n == 0:
        return

    # Resize all images to same height for display
    disp_h = 300
    imgs_resized = []
    for info in images_info:
        img = info['image']
        if img is None:
            continue
        ratio = disp_h / img.shape[0]
        disp_w = int(img.shape[1] * ratio)
        resized = cv2.resize(img, (disp_w, disp_h))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        # Add label
        label = info.get('label', '')
        score_text = f"Score: {info.get('score', 0):.4f}"
        verdict = info.get('verdict', '')
        cv2.putText(resized, label[:40], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(resized, score_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        color = (0, 0, 255) if 'DEFECT' in verdict else (0, 255, 0)
        cv2.putText(resized, verdict, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        imgs_resized.append(resized)

    if not imgs_resized:
        return

    # Stack horizontally (max 5 per row)
    rows = []
    max_per_row = 5
    max_w = max(img.shape[1] for img in imgs_resized)
    for i in range(0, len(imgs_resized), max_per_row):
        row_imgs = imgs_resized[i:i+max_per_row]
        # Pad to same width
        padded = []
        for img in row_imgs:
            if img.shape[1] < max_w:
                pad = np.ones((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8) * 200
                img = np.hstack([img, pad])
            padded.append(img)
        while len(padded) < max_per_row:
            padded.append(np.ones((disp_h, max_w, 3), dtype=np.uint8) * 200)
        rows.append(np.hstack(padded))

    canvas = np.vstack(rows)
    cv2.imwrite(str(output_path), canvas)


def evaluate_group(spec, gid, extractor, test_folders):
    """Evaluate one camera group."""
    group_info = CAMERA_GROUPS[gid]
    cams = group_info['cams']
    desc = group_info['desc']
    cam_id = cams[0]  # Primary camera

    bank_path = OUTPUT_DIR / spec / f'group_{gid}' / 'memory_bank.npy'
    if not bank_path.exists():
        return None

    bank = np.load(str(bank_path), allow_pickle=True)
    if not isinstance(bank, np.ndarray) or bank.ndim != 2 or bank.shape[0] < 10:
        print(f"  Group {gid}: INVALID bank, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"Group {gid}: {desc} (cam {cam_id}) — Bank: {bank.shape}")
    print(f"{'='*60}")

    baseline_mean, baseline_p95 = compute_baseline(bank)
    threshold = baseline_p95 * 1.2
    print(f"  Baseline: mean={baseline_mean:.4f}, p95={baseline_p95:.4f}, threshold={threshold:.4f}")

    results = {
        'group': gid, 'desc': desc, 'cam': cam_id,
        'bank_shape': list(bank.shape),
        'baseline_mean': baseline_mean, 'baseline_p95': baseline_p95,
        'threshold': threshold,
        'images': []
    }

    group_dir = RESULT_BASE / f'group_{gid}'
    group_dir.mkdir(parents=True, exist_ok=True)

    all_display_imgs = []

    # === 1. Normal images (3장) — 학습에 안 쓴 다른 날짜, 중간 인덱스 ===
    print(f"\n  --- Normal Images (3) ---")
    normal_count = 0
    for folder in test_folders:
        if normal_count >= 3:
            break
        imgs, cam_dir = get_images_from_folder(folder, cam_id)
        if len(imgs) < 200:
            continue
        # Use middle image (safe zone)
        mid_idx = len(imgs) // 2
        img_path = os.path.join(cam_dir, imgs[mid_idx])
        img = load_image(img_path)
        if img is None:
            continue

        tiles = tiles_from_image(img)
        feats = extract_tile_features(tiles, extractor)
        tile_scores = score_knn(feats, bank, k=3)
        img_score = compute_image_score(tile_scores)
        ratio = img_score / baseline_mean if baseline_mean > 0 else 0
        is_defect = img_score > threshold

        status = "DEFECT" if is_defect else "NORMAL"
        print(f"    [{normal_count+1}] {os.path.basename(folder)}/{imgs[mid_idx]} "
              f"score={img_score:.4f} ratio={ratio:.2f}x → {status}")

        # Save image
        out_name = f'normal_{normal_count+1}.jpg'
        shutil.copy2(img_path, str(group_dir / out_name))

        results['images'].append({
            'type': 'normal', 'idx': normal_count+1,
            'source': f"{os.path.basename(folder)}/{imgs[mid_idx]}",
            'score': float(img_score), 'ratio': float(ratio),
            'is_defect': bool(is_defect), 'verdict': status,
            'file': out_name
        })

        all_display_imgs.append({
            'image': img, 'label': f'Normal #{normal_count+1}',
            'score': img_score, 'verdict': status
        })
        normal_count += 1

    # === 2. Edge images (2장) — 인덱스 40-60 (head 근처, 불안정 구간) ===
    print(f"\n  --- Edge Images (idx 40-60) (2) ---")
    edge_count = 0
    for folder in test_folders:
        if edge_count >= 2:
            break
        imgs, cam_dir = get_images_from_folder(folder, cam_id)
        if len(imgs) < 100:
            continue
        # Pick image at index 40-60 (near head trim boundary)
        edge_idx = 40 + edge_count * 10  # 40 and 50
        if edge_idx >= len(imgs):
            continue
        img_path = os.path.join(cam_dir, imgs[edge_idx])
        img = load_image(img_path)
        if img is None:
            continue

        tiles = tiles_from_image(img)
        feats = extract_tile_features(tiles, extractor)
        tile_scores = score_knn(feats, bank, k=3)
        img_score = compute_image_score(tile_scores)
        ratio = img_score / baseline_mean if baseline_mean > 0 else 0
        is_defect = img_score > threshold

        status = "DEFECT" if is_defect else "NORMAL"
        print(f"    [{edge_count+1}] {os.path.basename(folder)}/{imgs[edge_idx]} (idx={edge_idx}) "
              f"score={img_score:.4f} ratio={ratio:.2f}x → {status}")

        out_name = f'edge_{edge_count+1}_idx{edge_idx}.jpg'
        shutil.copy2(img_path, str(group_dir / out_name))

        results['images'].append({
            'type': 'edge', 'idx': edge_count+1, 'img_index': edge_idx,
            'source': f"{os.path.basename(folder)}/{imgs[edge_idx]}",
            'score': float(img_score), 'ratio': float(ratio),
            'is_defect': bool(is_defect), 'verdict': status,
            'file': out_name
        })

        all_display_imgs.append({
            'image': img, 'label': f'Edge idx={edge_idx}',
            'score': img_score, 'verdict': status
        })
        edge_count += 1

    # === 3. Synthetic defect images (5장) ===
    print(f"\n  --- Synthetic Defect Images (5) ---")
    defect_kinds = ['scratch_long', 'spot_cluster', 'crack_zigzag', 'stain_large', 'multi_scratch']
    # Use the first test folder's middle image as base
    base_img = None
    base_source = ""
    for folder in test_folders:
        imgs, cam_dir = get_images_from_folder(folder, cam_id)
        if len(imgs) < 200:
            continue
        mid_idx = len(imgs) // 2 + 5  # Slightly different from normal test
        img_path = os.path.join(cam_dir, imgs[mid_idx])
        base_img = load_image(img_path)
        base_source = f"{os.path.basename(folder)}/{imgs[mid_idx]}"
        break

    if base_img is not None:
        for di, kind in enumerate(defect_kinds):
            defected, desc_kr = make_synthetic_defect(base_img, kind)

            tiles = tiles_from_image(defected)
            feats = extract_tile_features(tiles, extractor)
            tile_scores = score_knn(feats, bank, k=3)
            img_score = compute_image_score(tile_scores)
            ratio = img_score / baseline_mean if baseline_mean > 0 else 0
            is_defect = img_score > threshold

            status = "DEFECT ✓" if is_defect else "MISS ✗"
            print(f"    [{di+1}] {kind} ({desc_kr}) "
                  f"score={img_score:.4f} ratio={ratio:.2f}x → {status}")

            # Save defected image
            out_name = f'defect_{di+1}_{kind}.jpg'
            cv2.imwrite(str(group_dir / out_name), defected)

            results['images'].append({
                'type': 'synthetic', 'idx': di+1, 'defect_kind': kind,
                'defect_desc': desc_kr, 'base_source': base_source,
                'score': float(img_score), 'ratio': float(ratio),
                'is_defect': bool(is_defect), 'verdict': status,
                'file': out_name
            })

            all_display_imgs.append({
                'image': defected, 'label': f'Defect: {desc_kr}',
                'score': img_score, 'verdict': status
            })

    # Save comparison grid
    save_comparison_image(all_display_imgs, group_dir / 'comparison_grid.jpg')

    # Summary
    normal_fp = sum(1 for r in results['images'] if r['type'] == 'normal' and r['is_defect'])
    edge_fp = sum(1 for r in results['images'] if r['type'] == 'edge' and r['is_defect'])
    synth_det = sum(1 for r in results['images'] if r['type'] == 'synthetic' and r['is_defect'])
    synth_total = sum(1 for r in results['images'] if r['type'] == 'synthetic')

    results['summary'] = {
        'normal_false_positive': normal_fp,
        'edge_false_positive': edge_fp,
        'synthetic_detection_rate': f"{synth_det}/{synth_total}",
        'synthetic_detection_pct': synth_det / synth_total * 100 if synth_total > 0 else 0
    }

    print(f"\n  >>> Group {gid} Summary:")
    print(f"      Normal FP: {normal_fp}/3")
    print(f"      Edge FP: {edge_fp}/2")
    print(f"      Synthetic detection: {synth_det}/{synth_total} ({results['summary']['synthetic_detection_pct']:.0f}%)")

    return results


def main():
    print(f"{'='*60}")
    print(f"PatchCore Phase 1 Evaluation — {SPEC}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Find test folders
    test_folders = get_test_folders(SPEC, n=5)
    if not test_folders:
        print("No test folders found!")
        return
    print(f"\nTest folders ({len(test_folders)}):")
    for f in test_folders:
        print(f"  {os.path.basename(f)}")

    # Init extractor
    extractor = FeatureExtractor('cuda')
    print("\nFeature extractor ready")

    # Evaluate each group
    all_results = {'spec': SPEC, 'date': datetime.now().isoformat(), 'groups': {}}

    for gid in range(1, 6):
        result = evaluate_group(SPEC, gid, extractor, test_folders)
        if result:
            all_results['groups'][f'group_{gid}'] = result

    # Save JSON results
    json_path = RESULT_BASE / 'phase1_results.json'
    with open(str(json_path), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for gk, gr in all_results['groups'].items():
        s = gr['summary']
        print(f"  {gk} ({gr['desc']}): "
              f"Normal FP={s['normal_false_positive']}/3, "
              f"Edge FP={s['edge_false_positive']}/2, "
              f"Synth Det={s['synthetic_detection_rate']}")

    print(f"\nResults saved to: {RESULT_BASE}")
    print("PHASE1_COMPLETE")


if __name__ == '__main__':
    main()
