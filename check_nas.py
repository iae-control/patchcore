#!/usr/bin/env python3
"""Quick check of NAS structure for v9 FP test."""
from pathlib import Path
import re

NAS_ROOT = Path("/home/dk-sdd/nas_storage")

# Check date dirs
dates = [d.name for d in sorted(NAS_ROOT.iterdir()) if d.is_dir() and re.match(r'^\d{8}$', d.name)]
print(f"Date dirs: {len(dates)}")
print(f"First 10: {dates[:10]}")
print(f"Last 10: {dates[-10:]}")

# Check which dates have 596x199
spec_dates = []
for d in dates:
    dp = NAS_ROOT / d
    subs = [s.name for s in dp.iterdir() if s.is_dir() and '596x199' in s.name]
    if subs:
        spec_dates.append((d, len(subs)))

print(f"\nDates with 596x199: {len(spec_dates)}")
for d, n in spec_dates:
    tag = " [TRAIN]" if d in ['20250831','20251027'] else " [DEFECT]" if d == '20250630' else ""
    print(f"  {d}: {n} folders{tag}")
