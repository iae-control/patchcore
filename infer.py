#!/usr/bin/env python3
"""Inference script: score new images against trained PatchCore models."""
import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src import config
from src.utils import discover_folders, tile_positions, ensure_dir
from src.tile_mask import load_or_compute_masks
from src.dataset import TileDataset
from src.patchcore import FeatureExtractor, extract_features, PatchCoreModel


def infer_group(group_id: int, folders: list, extractor: FeatureExtractor, threshold: float = None):
    """Run inference for a camera group."""
    grp = config.CAMERA_GROUPS[group_id]
    cam_ids = grp["cams"]
    mirror_cam_id = cam_ids[config.MIRROR_CAM_INDEX]
    group_name = grp["name"]

    # Load model
    model_path = config.OUTPUT_DIR / f"group_{group_id}" / "memory_bank.npy"
    if not model_path.exists():
        print(f"Model not found for group {group_id}: {model_path}")
        return

    model = PatchCoreModel()
    model.load(model_path)
    print(f"\nGroup {group_id} ({grp['desc']}): loaded {model.memory_bank.shape[0]} coreset features")

    # Load masks
    tile_masks = load_or_compute_masks(folders, cam_ids, group_name)

    # Build dataset
    dataset = TileDataset(
        folders=folders,
        cam_ids=cam_ids,
        mirror_cam_id=mirror_cam_id,
        tile_mask=tile_masks,
    )
    
    if len(dataset) == 0:
        print("  No tiles to infer.")
        return

    # Extract features
    features = extract_features(dataset, extractor, desc=f"Group {group_id} inference")

    # Score
    scores = model.score(features)

    # Aggregate results per image
    results = {}
    for idx in range(len(dataset)):
        key = dataset.get_key(idx)
        folder, cam_id, img_idx, tile_idx = key
        img_key = f"{folder}/cam{cam_id}/img{img_idx}"
        if img_key not in results:
            results[img_key] = {"scores": [], "max_score": 0.0}
        s = float(scores[idx])
        results[img_key]["scores"].append(s)
        results[img_key]["max_score"] = max(results[img_key]["max_score"], s)

    # Save results
    out_path = config.OUTPUT_DIR / f"group_{group_id}" / "inference_results.json"
    ensure_dir(out_path.parent)
    
    # Simplify for JSON
    summary = {k: {"max_score": v["max_score"], "mean_score": np.mean(v["scores"]),
                    "n_tiles": len(v["scores"])} for k, v in results.items()}
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Stats
    all_max = [v["max_score"] for v in results.values()]
    print(f"  Images: {len(results)}, Score range: {min(all_max):.4f} - {max(all_max):.4f}")
    if threshold:
        anomalous = sum(1 for s in all_max if s >= threshold)
        print(f"  Anomalous (>{threshold}): {anomalous}/{len(results)}")
    print(f"  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="H-beam PatchCore Inference")
    parser.add_argument("--group", type=int, help="Infer specific group (1-5)")
    parser.add_argument("--all", action="store_true", help="Infer all groups")
    parser.add_argument("--spec", type=str, help="Filter by spec")
    parser.add_argument("--threshold", type=float, help="Anomaly threshold")
    args = parser.parse_args()

    if not args.group and not args.all:
        parser.error("Specify --group N or --all")

    device = config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        config.DEVICE = "cpu"

    folders = discover_folders(spec_filter=args.spec)
    if not folders:
        print("No valid folders found!")
        sys.exit(1)

    extractor = FeatureExtractor(device=device)
    groups = list(range(1, 6)) if args.all else [args.group]
    
    for gid in groups:
        infer_group(gid, folders, extractor, threshold=args.threshold)


if __name__ == "__main__":
    main()
