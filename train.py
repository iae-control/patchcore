#!/usr/bin/env python3
"""H-beam PatchCore Training — v3: individual spec×camera models with checkpoint/resume."""
import argparse
import sys
import time
import json
import torch
from pathlib import Path
from datetime import datetime

from src import config
from src.utils import (
    discover_all_specs, get_trainable_specs, build_fallback_map,
    adaptive_subsample, ensure_dir, load_progress, save_progress,
)
from src.tile_mask import load_or_compute_masks
from src.patchcore import FeatureExtractor
from src.self_validation import self_validation_loop


def train_spec_camera(spec_key: str, folders: list, group_id: int,
                      extractor: FeatureExtractor, subsample_step: int):
    """Train one model: spec × camera group."""
    grp = config.CAMERA_GROUPS[group_id]
    cam_ids = grp["cams"]
    mirror_cam_id = cam_ids[config.MIRROR_CAM_INDEX]
    group_name = grp["name"]

    print(f"\n{'='*60}")
    print(f"Spec: {spec_key} | Group {group_id}: {grp['desc']} ({group_name})")
    print(f"  Cameras: {cam_ids}, Folders: {len(folders)}, Step: {subsample_step}")
    print(f"{'='*60}")

    # Check if already trained
    model_path = config.OUTPUT_DIR / spec_key / f"group_{group_id}" / "memory_bank.npy"
    if model_path.exists():
        print(f"  Already trained: {model_path} — SKIP")
        return True

    # Step 1: Tile masks
    print("\n[1/2] Computing tile masks...")
    tile_masks = load_or_compute_masks(folders, cam_ids, group_name)
    for fname, mask in list(tile_masks.items())[:2]:
        n_valid = mask.sum()
        print(f"  {fname}: {n_valid}/{len(mask)} valid tiles")

    # Step 2: Self-validation training
    print("\n[2/2] Self-validation training...")
    try:
        model, excluded = self_validation_loop(
            folders=folders,
            cam_ids=cam_ids,
            mirror_cam_id=mirror_cam_id,
            tile_masks=tile_masks,
            spec_key=spec_key,
            group_id=group_id,
            group_name=group_name,
            extractor=extractor,
            subsample_step=subsample_step,
        )
        print(f"\n  ✓ {spec_key}/group_{group_id} complete. Excluded: {len(excluded)} tiles")
        return True
    except Exception as e:
        print(f"\n  ✗ {spec_key}/group_{group_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="H-beam PatchCore Training v3")
    parser.add_argument("--spec", type=str, help="Train single spec (e.g. '500x200')")
    parser.add_argument("--group", type=int, help="Train single camera group (1-5)")
    parser.add_argument("--all", action="store_true", help="Train ALL eligible specs × ALL camera groups")
    parser.add_argument("--list", action="store_true", help="List specs and eligibility")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed spec×camera pairs")
    parser.add_argument("--target-images", type=int, help="Override target images per model")
    parser.add_argument("--min-folders", type=int, help="Override min folders threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without training")
    args = parser.parse_args()

    if args.target_images:
        config.TARGET_IMAGES_PER_MODEL = args.target_images
    if args.min_folders:
        config.MIN_FOLDERS = args.min_folders

    # Discover all specs
    print("Scanning NAS...")
    t0 = time.time()
    all_specs = discover_all_specs()
    elapsed = time.time() - t0
    print(f"Scan complete: {sum(len(v) for v in all_specs.values()):,} folders, "
          f"{len(all_specs)} unique specs in {elapsed:.1f}s")

    trainable, sparse = get_trainable_specs(all_specs)
    fallback_map = build_fallback_map(trainable, sparse)

    # List mode
    if args.list:
        print(f"\n=== Trainable Specs (>= {config.MIN_FOLDERS} folders) ===")
        total_models = 0
        for sk, n, w, h in trainable:
            print(f"  {sk:20s}  {n:5d} folders  ({w}×{h}mm)")
            total_models += 5
        print(f"\nTotal: {len(trainable)} specs × 5 cameras = {total_models} models")
        
        print(f"\n=== Sparse Specs (< {config.MIN_FOLDERS} folders) → Fallback ===")
        for sk, n, w, h in sparse:
            fb = fallback_map.get(sk, "???")
            print(f"  {sk:20s}  {n:3d} folders  → fallback: {fb}")
        
        # Save fallback map
        fb_path = config.OUTPUT_DIR / "fallback_map.json"
        ensure_dir(fb_path.parent)
        with open(fb_path, "w") as f:
            json.dump(fallback_map, f, indent=2)
        print(f"\nFallback map saved: {fb_path}")
        return

    if not args.spec and not args.all:
        parser.error("Specify --spec NAME, --all, or --list")

    # Setup device
    device = config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        config.DEVICE = "cpu"
    else:
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Load backbone
    print("\nLoading WideResNet50 backbone...")
    extractor = FeatureExtractor(device=device)

    # Determine what to train
    if args.spec:
        specs_to_train = [(sk, n, w, h) for sk, n, w, h in trainable if sk == args.spec]
        if not specs_to_train:
            # Check if it's a sparse spec
            if args.spec in [sk for sk, _, _, _ in sparse]:
                print(f"Spec '{args.spec}' has < {config.MIN_FOLDERS} folders. "
                      f"Fallback: {fallback_map.get(args.spec, '???')}")
                sys.exit(1)
            print(f"Unknown spec: {args.spec}")
            print(f"Available: {[sk for sk, _, _, _ in trainable]}")
            sys.exit(1)
    else:
        specs_to_train = trainable

    cam_groups = [args.group] if args.group else list(range(1, 6))

    # Training plan
    total_jobs = len(specs_to_train) * len(cam_groups)
    print(f"\n{'#'*60}")
    print(f"# Training Plan: {len(specs_to_train)} specs × {len(cam_groups)} cameras = {total_jobs} models")
    print(f"# Resume mode: {'ON' if args.resume else 'OFF'}")
    print(f"{'#'*60}")

    if args.dry_run:
        for sk, n, w, h in specs_to_train:
            for gid in cam_groups:
                model_path = config.OUTPUT_DIR / sk / f"group_{gid}" / "memory_bank.npy"
                status = "EXISTS" if model_path.exists() else "TODO"
                print(f"  {sk}/group_{gid}: {n} folders — {status}")
        return

    # Progress tracking
    progress_path = config.OUTPUT_DIR / "training_progress.json"
    progress = load_progress(progress_path)

    completed = 0
    failed = 0
    skipped = 0

    for si, (sk, n_folders, w, h) in enumerate(specs_to_train):
        folders = all_specs[sk]
        
        for gid in cam_groups:
            job_key = f"{sk}/group_{gid}"
            
            # Resume: skip completed
            if args.resume and job_key in progress["completed"]:
                skipped += 1
                continue

            print(f"\n>>> [{si*len(cam_groups)+cam_groups.index(gid)+1}/{total_jobs}] {job_key}")
            
            # Adaptive subsample
            cam_ids = config.CAMERA_GROUPS[gid]["cams"]
            sampled, step = adaptive_subsample(folders, cam_ids)
            
            t_start = time.time()
            ok = train_spec_camera(sk, sampled, gid, extractor, step)
            elapsed = time.time() - t_start
            
            if ok:
                completed += 1
                progress["completed"].append(job_key)
            else:
                failed += 1
                progress["failed"].append(job_key)
            
            save_progress(progress_path, progress)
            print(f"  Time: {elapsed/60:.1f} min | Progress: {completed} done, "
                  f"{failed} failed, {skipped} skipped / {total_jobs} total")

    # Save fallback map
    fb_path = config.OUTPUT_DIR / "fallback_map.json"
    with open(fb_path, "w") as f:
        json.dump(fallback_map, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Fallback map: {fb_path}")
    print(f"  Output: {config.OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
