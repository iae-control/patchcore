#!/usr/bin/env python3
"""Fast NAS scan — folder names only, no file I/O."""
import os, re
from collections import Counter, defaultdict
from pathlib import Path

NAS = Path("/home/dk-sdd/nas_storage")

all_specs = Counter()
spec_dates = defaultdict(set)

# 1. Root-level folders
root_items = sorted(os.listdir(NAS))
date_dirs = [d for d in root_items if d.isdigit() and len(d) == 8]
direct_folders = [d for d in root_items if not (d.isdigit() and len(d) == 8)]

for f in direct_folders:
    parts = f.split("_", 1)
    if len(parts) >= 2:
        date8 = parts[0][:8]
        spec = parts[1]
        all_specs[spec] += 1
        spec_dates[spec].add(date8)

print(f"Root-level direct folders: {len(direct_folders)}")

# 2. Date directories — list subfolders only (no file stat)
print(f"Date directories: {len(date_dirs)}")
total_subs = 0

for dname in date_dirs:
    dpath = NAS / dname
    try:
        subs = os.listdir(dpath)
    except:
        continue
    for s in subs:
        if not os.path.isdir(dpath / s):
            continue
        parts = s.split("_", 1)
        if len(parts) >= 2:
            spec = parts[1]
        else:
            spec = "unknown"
        all_specs[spec] += 1
        spec_dates[spec].add(dname)
        total_subs += 1

print(f"Total sub-folders in date dirs: {total_subs}")

# 3. Parse dimensions and normalize
print(f"\n{'='*70}")
print(f"ALL SPECS (by folder count)")
print(f"{'='*70}")
print(f"  {'Spec':<25s} {'Folders':>8s} {'Dates':>6s}")
print(f"  {'-'*25} {'-'*8} {'-'*6}")

for spec, cnt in all_specs.most_common(50):
    n_dates = len(spec_dates[spec])
    print(f"  {spec:<25s} {cnt:>8d} {n_dates:>6d}")

# 4. Group by physical dimension (WxH)
print(f"\n{'='*70}")
print(f"UNIQUE PHYSICAL DIMENSIONS (normalized WxH, W>=H)")
print(f"{'='*70}")

dim_map = defaultdict(lambda: {"folders": 0, "dates": set(), "specs": []})
for spec in all_specs:
    m = re.match(r'^(\d+)[xX](\d+)$', spec)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        # Normalize: always W >= H
        key = (max(w,h), min(w,h))
        dim_map[key]["folders"] += all_specs[spec]
        dim_map[key]["dates"].update(spec_dates[spec])
        dim_map[key]["specs"].append(spec)

print(f"  {'WxH':<15s} {'Folders':>8s} {'Dates':>6s} {'Spec variants'}")
print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*30}")

for (w,h), info in sorted(dim_map.items(), key=lambda x: -x[1]["folders"]):
    specs_str = ", ".join(info["specs"])
    print(f"  {w}x{h:<10d} {info['folders']:>8d} {len(info['dates']):>6d}   {specs_str}")

# 5. Date range per major spec
print(f"\n{'='*70}")
print(f"DATE RANGES (top specs)")
print(f"{'='*70}")

for spec, cnt in all_specs.most_common(15):
    dates = sorted(spec_dates[spec])
    if len(dates) > 0:
        print(f"  {spec}: {dates[0]}~{dates[-1]} ({len(dates)} dates, {cnt} folders)")

print("\nDone.")
