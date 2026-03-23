
import os, re, json
from pathlib import Path
from collections import defaultdict

NAS_ROOT = Path("/home/dk-sdd/nas_storage")
OUTPUT = Path("/home/dk-sdd/patchcore/nas_inventory.json")
OUTPUT_TXT = Path("/home/dk-sdd/patchcore/nas_inventory.txt")

results = []
date_summary = defaultdict(lambda: {"folders": 0, "total_images": 0, "specs": set()})
spec_summary = defaultdict(lambda: {"folders": 0, "total_images": 0, "dates": set()})

print("Scanning NAS root...", flush=True)
all_entries = sorted(NAS_ROOT.iterdir())
print(f"  Top-level entries: {len(all_entries)}", flush=True)

for entry in all_entries:
    if not entry.is_dir():
        continue

    # Date folders (YYYYMMDD)
    if re.match(r'^\d{8}$', entry.name):
        date_str = entry.name
        try:
            subs = sorted(entry.iterdir())
        except PermissionError:
            continue

        for sub in subs:
            if not sub.is_dir():
                continue

            # Check level2.txt
            has_level2 = (sub / "level2.txt").exists()

            # Count images per camera
            cam_counts = {}
            total_imgs = 0
            for cam_dir in sorted(sub.iterdir()):
                if not cam_dir.is_dir() or not cam_dir.name.startswith("camera_"):
                    continue
                n_imgs = len([f for f in cam_dir.iterdir()
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                if n_imgs > 0:
                    cam_counts[cam_dir.name] = n_imgs
                    total_imgs += n_imgs

            if total_imgs == 0:
                continue

            # Extract spec from folder name
            spec = sub.name

            record = {
                "date": date_str,
                "folder": str(sub),
                "spec": spec,
                "has_level2": has_level2,
                "cameras": cam_counts,
                "total_images": total_imgs,
            }
            results.append(record)

            date_summary[date_str]["folders"] += 1
            date_summary[date_str]["total_images"] += total_imgs
            date_summary[date_str]["specs"].add(spec)

            spec_summary[spec]["folders"] += 1
            spec_summary[spec]["total_images"] += total_imgs
            spec_summary[spec]["dates"].add(date_str)

        print(f"  Date {date_str}: {len([r for r in results if r['date']==date_str])} folders", flush=True)

# Also check non-date top-level folders (like defect folders)
for entry in all_entries:
    if not entry.is_dir():
        continue
    if re.match(r'^\d{8}$', entry.name):
        continue  # already processed

    has_level2 = (entry / "level2.txt").exists()
    cam_counts = {}
    total_imgs = 0
    for cam_dir in sorted(entry.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("camera_"):
            continue
        n_imgs = len([f for f in cam_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if n_imgs > 0:
            cam_counts[cam_dir.name] = n_imgs
            total_imgs += n_imgs

    if total_imgs > 0:
        record = {
            "date": "toplevel",
            "folder": str(entry),
            "spec": entry.name,
            "has_level2": has_level2,
            "cameras": cam_counts,
            "total_images": total_imgs,
        }
        results.append(record)

print(f"\nTotal records: {len(results)}", flush=True)

# Save JSON (convert sets to lists)
for k in date_summary:
    date_summary[k]["specs"] = sorted(date_summary[k]["specs"])
for k in spec_summary:
    spec_summary[k]["dates"] = sorted(spec_summary[k]["dates"])

inventory = {
    "total_folders": len(results),
    "total_images": sum(r["total_images"] for r in results),
    "folders_with_level2": sum(1 for r in results if r["has_level2"]),
    "unique_dates": len(date_summary),
    "unique_specs": len(spec_summary),
    "date_summary": dict(date_summary),
    "spec_summary": dict(spec_summary),
    "folders": results,
}

with open(OUTPUT, "w") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

# Also save human-readable summary
lines = []
lines.append("=" * 80)
lines.append("NAS INVENTORY SUMMARY")
lines.append("=" * 80)
lines.append(f"Total folders with images: {len(results)}")
lines.append(f"Total images: {sum(r['total_images'] for r in results):,}")
lines.append(f"Folders with level2.txt: {sum(1 for r in results if r['has_level2'])}")
lines.append(f"Unique dates: {len(date_summary)}")
lines.append(f"Unique specs: {len(spec_summary)}")
lines.append("")

# Date summary
lines.append("=" * 80)
lines.append("BY DATE")
lines.append("=" * 80)
lines.append(f"{'Date':>10} {'Folders':>8} {'Images':>10} {'Specs':>6} {'Has level2':>10}")
lines.append("-" * 50)
for date in sorted(date_summary.keys()):
    d = date_summary[date]
    n_l2 = sum(1 for r in results if r["date"] == date and r["has_level2"])
    lines.append(f"{date:>10} {d['folders']:>8} {d['total_images']:>10,} {len(d['specs']):>6} {n_l2:>10}")

lines.append("")

# Spec summary (top 20 by image count)
lines.append("=" * 80)
lines.append("BY SPEC (top 30 by image count)")
lines.append("=" * 80)
lines.append(f"{'Spec':>30} {'Folders':>8} {'Images':>10} {'Dates':>6}")
lines.append("-" * 60)
sorted_specs = sorted(spec_summary.items(), key=lambda x: x[1]["total_images"], reverse=True)
for spec, d in sorted_specs[:30]:
    lines.append(f"{spec:>30} {d['folders']:>8} {d['total_images']:>10,} {len(d['dates']):>6}")

lines.append("")

# 596x199 specific
lines.append("=" * 80)
lines.append("596x199 DETAIL")
lines.append("=" * 80)
target_records = [r for r in results if "596x199" in r["spec"]]
lines.append(f"Folders: {len(target_records)}")
lines.append(f"Total images: {sum(r['total_images'] for r in target_records):,}")
lines.append(f"With level2: {sum(1 for r in target_records if r['has_level2'])}")
lines.append("")
lines.append(f"{'Date':>10} {'Folder':>40} {'Cam1':>6} {'Cam10':>6} {'Total':>7} {'L2':>3}")
lines.append("-" * 80)
for r in sorted(target_records, key=lambda x: x["date"]):
    c1 = r["cameras"].get("camera_1", 0)
    c10 = r["cameras"].get("camera_10", 0)
    l2 = "Y" if r["has_level2"] else ""
    name = r["spec"] if len(r["spec"]) <= 40 else r["spec"][:37] + "..."
    lines.append(f"{r['date']:>10} {name:>40} {c1:>6} {c10:>6} {r['total_images']:>7} {l2:>3}")

txt = "\n".join(lines)
with open(OUTPUT_TXT, "w") as f:
    f.write(txt)
print(txt)
print("\nSaved to:", OUTPUT, OUTPUT_TXT, flush=True)
print("INVENTORY_DONE")
