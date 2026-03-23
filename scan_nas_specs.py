#!/usr/bin/env python3
"""Scan NAS to catalog all H-beam specs, dates, camera configs."""
import os, sys, json
from collections import Counter, defaultdict
from pathlib import Path

NAS = Path("/home/dk-sdd/nas_storage")

spec_by_date = defaultdict(Counter)
all_specs = Counter()
date_folder_counts = {}
spec_dates = defaultdict(set)  # spec -> set of dates
spec_has_images = defaultdict(int)  # spec -> folders with actual images

# Scan date directories
dates = sorted([d for d in NAS.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8])
print(f"Date directories: {len(dates)}")

for date_dir in dates:
    subs = sorted([s for s in date_dir.iterdir() if s.is_dir()])
    date_folder_counts[date_dir.name] = len(subs)
    for sub in subs:
        parts = sub.name.split("_")
        if len(parts) >= 2:
            spec = "_".join(parts[1:])
        else:
            spec = "unknown"
        spec_by_date[date_dir.name][spec] += 1
        all_specs[spec] += 1
        spec_dates[spec].add(date_dir.name)
        # Check if has actual images (camera_1 with files)
        cam1 = sub / "camera_1"
        if cam1.is_dir():
            files = list(cam1.iterdir())
            if len(files) > 10:  # has actual images, not just metadata
                spec_has_images[spec] += 1

# Also scan root-level folders
root_folders = [d for d in NAS.iterdir() if d.is_dir() and not (d.name.isdigit() and len(d.name) == 8)]
for folder in root_folders:
    parts = folder.name.split("_")
    if len(parts) >= 2:
        spec = "_".join(parts[1:])
    else:
        spec = folder.name
    date_str = parts[0][:8] if parts[0][:8].isdigit() else "unknown"
    # Check for images
    cam1 = folder / "camera_1"
    if cam1.is_dir():
        files = list(cam1.iterdir())
        if len(files) > 10:
            spec_has_images[spec] += 1
            all_specs[spec] += 1
            spec_dates[spec].add(date_str)

total_sub = sum(date_folder_counts.values())
print(f"Total sub-folders in date dirs: {total_sub}")
print(f"Root-level folders: {len(root_folders)}")

print(f"\n{'='*80}")
print(f"ALL H-BEAM SPECS")
print(f"{'='*80}")
print(f"{'Spec':<25s} {'Folders':>8s} {'Dates':>6s} {'WithImg':>8s}")
print(f"{'-'*25} {'-'*8} {'-'*6} {'-'*8}")

for spec, cnt in all_specs.most_common(60):
    n_dates = len(spec_dates[spec])
    n_with_img = spec_has_images.get(spec, 0)
    print(f"  {spec:<23s} {cnt:>8d} {n_dates:>6d} {n_with_img:>8d}")

# Camera structure check for top specs
print(f"\n{'='*80}")
print(f"CAMERA STRUCTURE CHECK (top specs)")
print(f"{'='*80}")

top_specs = [s for s, _ in all_specs.most_common(10)]
for spec in top_specs:
    # Find a folder with images for this spec
    for date in sorted(spec_dates[spec]):
        date_path = NAS / date
        if date_path.is_dir():
            for sub in date_path.iterdir():
                if spec in sub.name and sub.is_dir():
                    cam_dirs = sorted([c for c in sub.iterdir() if c.is_dir() and c.name.startswith("camera_")])
                    if cam_dirs:
                        cam_names = [c.name for c in cam_dirs]
                        # Check file counts per camera
                        cam_counts = {}
                        for c in cam_dirs:
                            cam_counts[c.name] = len(list(c.iterdir()))
                        print(f"\n  {spec} (from {sub.name}):")
                        print(f"    Cameras: {cam_names}")
                        print(f"    Files/cam: {cam_counts}")
                        break
            else:
                continue
            break

# Summary of spec dimensions (parse WxH)
print(f"\n{'='*80}")
print(f"UNIQUE PHYSICAL DIMENSIONS (WxH)")
print(f"{'='*80}")

import re
dim_specs = defaultdict(list)
for spec in all_specs:
    m = re.match(r'^(\d+)[xX](\d+)$', spec)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        dim_specs[(w,h)].append(spec)

for (w,h), specs in sorted(dim_specs.items()):
    total = sum(all_specs[s] for s in specs)
    n_dates = len(set().union(*[spec_dates[s] for s in specs]))
    n_img = sum(spec_has_images.get(s, 0) for s in specs)
    print(f"  {w}x{h}: {total} folders, {n_dates} dates, {n_img} with images")

print("\nDone.")
