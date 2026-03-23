
import os, re
from pathlib import Path

NAS_ROOT = Path("/home/dk-sdd/nas_storage")
TARGET_SPEC = "596x199"
DEFECT_PREFIX = "20250630160852"
TRIM_HEAD = 100
TRIM_TAIL = 100

folders = []
for entry in sorted(NAS_ROOT.iterdir()):
    if not entry.is_dir():
        continue
    if re.match(r'^\d{8}$', entry.name):
        try:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and TARGET_SPEC in sub.name:
                    if (sub / "camera_1").is_dir():
                        folders.append(sub)
        except PermissionError:
            continue
    elif TARGET_SPEC in entry.name:
        if (entry / "camera_1").is_dir():
            folders.append(entry)

print(f"Total folders with {TARGET_SPEC}: {len(folders)}")
print()

defect = []
train = []
for f in folders:
    if DEFECT_PREFIX in f.name:
        defect.append(f)
    else:
        train.append(f)

print(f"Training folders: {len(train)}")
print(f"Defect folders: {len(defect)}")
print()

# Count images per folder
total_cam1 = 0
total_cam10 = 0
total_usable_cam1 = 0
total_usable_cam10 = 0

print(f"{'Date':>10} {'Folder':>40} {'Cam1':>6} {'Cam10':>6} {'Usable1':>8} {'Usable10':>8}")
print("-" * 90)

for f in train:
    cam1_dir = f / "camera_1"
    cam10_dir = f / "camera_10"

    cam1_imgs = sorted([p for p in cam1_dir.iterdir() if p.suffix.lower() in ('.jpg','.png')]) if cam1_dir.exists() else []
    cam10_imgs = sorted([p for p in cam10_dir.iterdir() if p.suffix.lower() in ('.jpg','.png')]) if cam10_dir.exists() else []

    n1 = len(cam1_imgs)
    n10 = len(cam10_imgs)

    usable1 = max(0, n1 - TRIM_HEAD - TRIM_TAIL)
    usable10 = max(0, n10 - TRIM_HEAD - TRIM_TAIL)

    total_cam1 += n1
    total_cam10 += n10
    total_usable_cam1 += usable1
    total_usable_cam10 += usable10

    date_part = f.parent.name if re.match(r'^\d{8}$', f.parent.name) else ""
    print(f"{date_part:>10} {f.name:>40} {n1:>6} {n10:>6} {usable1:>8} {usable10:>8}")

print("-" * 90)
print(f"{'TOTAL':>10} {'':>40} {total_cam1:>6} {total_cam10:>6} {total_usable_cam1:>8} {total_usable_cam10:>8}")
print()
print(f"Total usable images (both cams): {total_usable_cam1 + total_usable_cam10}")
print()

# With different subsample rates
for step in [1, 3, 5, 10, 15]:
    n = (total_usable_cam1 + total_usable_cam10) // step
    print(f"  subsample={step:>2}: ~{n:>6} training images")

# Defect folder info
print()
for f in defect:
    cam1_dir = f / "camera_1"
    cam1_imgs = sorted([p for p in cam1_dir.iterdir() if p.suffix.lower() in ('.jpg','.png')]) if cam1_dir.exists() else []
    usable = max(0, len(cam1_imgs) - TRIM_HEAD - TRIM_TAIL)
    print(f"Defect folder: {f.name} -> {len(cam1_imgs)} images (usable: {usable})")
