#!/usr/bin/env python3
import os, sys, random, re, shutil
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "/home/dk-sdd/patchcore")
from src import config
from src.utils import discover_all_specs, get_trainable_specs, _natural_sort_key
OUTPUT_BASE = Path("/home/dk-sdd/patchcore/test_data")
IMAGES_PER_CAM = 5
SEED = 99
def get_all_images(folder, cam_id):
    cam_dir = folder["path"] / f"camera_{cam_id}"
    if not cam_dir.is_dir(): return []
    return sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")], key=_natural_sort_key)
def compute_step(n_folders, n_cams):
    est = n_folders * n_cams * 700
    if est <= config.TARGET_IMAGES_PER_MODEL: return 1
    return max(1, round(est / config.TARGET_IMAGES_PER_MODEL))
def select_test(spec_key, folders, gid):
    grp = config.CAMERA_GROUPS[gid]; cam_ids = grp["cams"]
    if len(folders) > config.FOLDER_SAMPLE_MAX:
        random.seed(42)
        sampled_names = set(f["name"] for f in random.sample(folders, config.FOLDER_SAMPLE_MAX))
        train_f = [f for f in folders if f["name"] in sampled_names]
        unused_f = [f for f in folders if f["name"] not in sampled_names]
    else:
        train_f = folders; unused_f = []
    step = compute_step(len(train_f), len(cam_ids))
    result = {}
    for cam_id in cam_ids:
        cands = []
        if unused_f:
            rng = random.Random(SEED + gid*100 + cam_id)
            uf = list(unused_f); rng.shuffle(uf)
            for folder in uf:
                imgs = get_all_images(folder, cam_id)
                if len(imgs) > config.TRIM_HEAD + config.TRIM_TAIL:
                    mid = imgs[config.TRIM_HEAD:len(imgs)-config.TRIM_TAIL]
                    if mid: cands.append(rng.choice(mid))
                if len(cands) >= IMAGES_PER_CAM: break
        if len(cands) < IMAGES_PER_CAM:
            rng = random.Random(SEED + gid*100 + cam_id + 50)
            tf = list(train_f); rng.shuffle(tf)
            for folder in tf:
                if len(cands) >= IMAGE_PER_CAM: break
                imgs = get_all_images(folder, cam_id)
                if len(imgs) <= config.TRIM_HEAD + config.TRIM_TAIL: continue
                pool = imgs[:config.TRIM_HEAD] + imgs[len(imgs)-config.TRIM_TAIL:]
                if pool: cands.append(rng.choice(pool))
        if len(cands) < IMAGES_PER_CAM and step > 1:
            rng = random.Random(SEED + gid*100 + cam_id + 99)
            for folder in train_f:
                if len(cands) >= IMAGES_PER_CAM: break
                imgs = get_all_images(folder, cam_id)
                if len(imgs) <= config.TRIM_HEAD + config.TRIM_TAIL: continue
                trimmed = imgs[config.TRIM_HEAD:len(imgs)-config.TRIM_TAIL]
                train_set = set(str(p) for p in trimmed[::step])
                between = [p for p in trimmed if str(p) not in train_set]
                if between: cands.append(rng.choice(between))
        result[cam_id] = cands[:IMAGES_PER_CAM]
    return result
def main():
    print("Scanning NAS...")
    all_specs = discover_all_specs()
    trainable, sparse = get_trainable_specs(all_specs)
    print(f"Found {len(trainable)} trainable specs")
    if OUTPUT_BASE.exists(): shutil.rmtree(OUTPUT_BASE)
    total_copied = 0; total_specs = 0
    for sk, nf, w, h in sorted(trainable, key=lambda x: x[1], reverse=True):
        folders = all_specs[sk]; total_specs += 1
        print(f"[{total_specs}/{len(trainable)}] {sk} ({nf} folders)")
        for gid in range(1, 6):
            ti = select_test(sk, folders, gid)
            for cam_id, images in ti.items():
                if not images: continue
                od = OUTPUT_BASE / sk / f"group_{gid}" / f"camera_{cam_id}"
                od.mkdir(parents=True, exist_ok=True)
                for ip in images:
                    try: shutil.copy2(str(ip), str(od / ip.name)); total_copied += 1
                    except Exception as e: print(f"  WARN: {e}")
            n = sum(len(v) for v in ti.values())
            if n > 0: print(f"  group_{gid}: {n} imgs")
    print()
    print(f"DONE: {total_copied} images, {total_specs} specs")
    print(f"Output: {OUTPUT_BASE}")
    with open(OUTPUT_BASE / "SUMMARY.txt", "w") as f:
        f.write("PatchCore Test Dataset\n")
        f.write(f"Specs: {total_specs}\n")
        f.write(f"Images: {total_copied}\n")
        f.write("Structure: <spec>/group_<N>/camera_<C>/<image>.jpg\n")
        f.write("Selection: Images NOT used in training\n")
if __name__ == "__main__": main()
