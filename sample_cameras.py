#!/usr/bin/env python3
"""Sample one image per camera from 596x199 and 700x300 for visual comparison."""
import os, shutil
from pathlib import Path
from PIL import Image
import numpy as np

NAS = Path("/home/dk-sdd/nas_storage")
OUT = Path("/home/dk-sdd/patchcore/camera_samples")
OUT.mkdir(exist_ok=True)

# 596x199 from 0630
spec_596 = NAS / "20250630" / "20250630150401_596x199"
# 700x300 — find one with images
spec_700_root = NAS / "20250319181040_700X300"
# Also try a common spec: 300x300 from a date dir
spec_300 = None
for date_dir in sorted(NAS.iterdir()):
    if date_dir.name == "20250609":
        for sub in date_dir.iterdir():
            if "300x300" in sub.name:
                spec_300 = sub
                break
        break

specs = {
    "596x199": spec_596,
    "700X300": spec_700_root,
}
if spec_300:
    specs["300x300"] = spec_300

for spec_name, spec_dir in specs.items():
    print(f"\n=== {spec_name}: {spec_dir} ===")
    if not spec_dir.exists():
        print(f"  NOT FOUND")
        continue

    spec_out = OUT / spec_name
    spec_out.mkdir(exist_ok=True)

    for cam_id in range(1, 11):
        cam_dir = spec_dir / f"camera_{cam_id}"
        if not cam_dir.exists():
            print(f"  camera_{cam_id}: NOT FOUND")
            continue

        imgs = sorted([f for f in cam_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
        if len(imgs) < 200:
            print(f"  camera_{cam_id}: only {len(imgs)} images")
            if len(imgs) == 0:
                continue
            img_path = imgs[len(imgs)//2]
        else:
            # Pick middle image (skip trim)
            img_path = imgs[len(imgs)//2]

        img = Image.open(img_path)
        print(f"  camera_{cam_id}: {img.size} {img.mode} — {img_path.name}")

        # Save as jpg for easy transfer
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        img_rgb.save(spec_out / f"cam{cam_id}.jpg", quality=90)

        # Also compute basic stats per 192-wide strip
        arr = np.array(img.convert('L')).astype(float)
        h, w = arr.shape
        strip_w = 192
        n_strips = w // strip_w
        strip_stats = []
        for s in range(n_strips):
            strip = arr[:, s*strip_w:(s+1)*strip_w]
            strip_stats.append(f"{strip.mean():.0f}")
        print(f"    Strip means ({n_strips}): [{', '.join(strip_stats)}]")

# Camera pairing analysis: compare cam1 vs cam10 (Group 1 pair)
print(f"\n{'='*60}")
print(f"CAMERA PAIRING ANALYSIS (596x199)")
print(f"{'='*60}")

cam_pairs = [(1,10), (2,9), (3,8), (4,7), (5,6)]
for ca, cb in cam_pairs:
    fa = spec_596 / f"camera_{ca}"
    fb = spec_596 / f"camera_{cb}"
    if not fa.exists() or not fb.exists():
        continue

    imgs_a = sorted([f for f in fa.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')])
    imgs_b = sorted([f for f in fb.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')])

    if len(imgs_a) < 200 or len(imgs_b) < 200:
        continue

    # Compare middle images
    img_a = np.array(Image.open(imgs_a[len(imgs_a)//2]).convert('L')).astype(float)
    img_b = np.array(Image.open(imgs_b[len(imgs_b)//2]).convert('L')).astype(float)

    # Flip camera_b (mirror pair)
    img_b_flip = np.fliplr(img_b)

    print(f"\n  Group cam[{ca},{cb}]:")
    print(f"    cam{ca} mean={img_a.mean():.1f}, std={img_a.std():.1f}")
    print(f"    cam{cb} mean={img_b.mean():.1f}, std={img_b.std():.1f}")
    print(f"    cam{cb}_flip mean={img_b_flip.mean():.1f}")

    # Per-strip comparison (192px strips)
    diffs = []
    for s in range(10):
        sa = img_a[:, s*192:(s+1)*192].mean()
        sb = img_b_flip[:, s*192:(s+1)*192].mean()
        diffs.append(abs(sa - sb))
    print(f"    Strip mean diffs (|A - B_flip|): [{', '.join(f'{d:.1f}' for d in diffs)}]")
    print(f"    Avg strip diff: {np.mean(diffs):.1f}, Max: {np.max(diffs):.1f}")

print("\nDone. Samples saved to /home/dk-sdd/patchcore/camera_samples/")
