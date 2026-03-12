"""Self-validation — v3: checkpoint/resume after feature extraction."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Tuple
from src import config
from src.utils import ensure_dir, save_checkpoint, load_checkpoint, remove_checkpoint
from src.dataset import TileDataset
from src.patchcore import FeatureExtractor, extract_features, PatchCoreModel


def self_validation_loop(
    folders: list,
    cam_ids: list,
    mirror_cam_id: int,
    tile_masks: dict,
    spec_key: str,
    group_id: int,
    group_name: str,
    extractor: FeatureExtractor,
    rounds: int = None,
    reject_pct: float = None,
    subsample_step: int = None,
) -> Tuple[PatchCoreModel, Set]:
    """Run self-validation loop with checkpoint/resume support."""
    rounds = rounds or config.SELF_VAL_ROUNDS
    reject_pct = reject_pct if reject_pct is not None else config.SELF_VAL_REJECT_PERCENT
    
    out_dir = config.OUTPUT_DIR / spec_key / f"group_{group_id}" / "self_val"
    ckpt_dir = config.CHECKPOINT_DIR / spec_key / f"group_{group_id}"
    ensure_dir(out_dir)

    excluded_keys: Set[tuple] = set()
    model = PatchCoreModel()

    for rnd in range(rounds):
        print(f"\n  ── Self-validation round {rnd}/{rounds - 1} "
              f"(excluded: {len(excluded_keys)} tiles) ──")

        # Check for feature checkpoint from previous crashed run
        ckpt_path = ckpt_dir / f"round_{rnd}_features.npz"
        ckpt = load_checkpoint(ckpt_path)
        
        if ckpt is not None:
            # Resume: skip feature extraction, go straight to coreset
            features = ckpt["features"]
            tile_keys_flat = ckpt["tile_keys"]  # saved as string array
            print(f"  Resumed {len(features):,} features from checkpoint")
        else:
            # Normal: build dataset and extract features
            dataset = TileDataset(
                folders=folders,
                cam_ids=cam_ids,
                mirror_cam_id=mirror_cam_id,
                tile_mask=tile_masks,
                exclude_keys=excluded_keys,
                subsample_step=subsample_step,
            )
            
            if len(dataset) == 0:
                print("  WARNING: No tiles left!")
                break

            print(f"  Dataset size: {len(dataset):,} tiles")

            features = extract_features(dataset, extractor, desc=f"Round {rnd} features")
            
            # Save checkpoint BEFORE coreset (the expensive/crash-prone step)
            tile_keys_flat = np.array([str(dataset.get_key(i)) for i in range(len(dataset))])
            save_checkpoint(ckpt_path, {
                "features": features,
                "tile_keys": tile_keys_flat,
            })

        # Coreset + scoring (this is where Group 4 crashed)
        print(f"  Fitting coreset on {len(features):,} features...")
        model.fit(features)
        scores = model.score(features)

        # Remove checkpoint after successful coreset
        remove_checkpoint(ckpt_path)

        _save_histogram(scores, out_dir / f"round_{rnd}_hist.png", rnd)

        threshold = np.percentile(scores, 100 - reject_pct)
        reject_mask = scores >= threshold
        n_reject = reject_mask.sum()
        print(f"  Threshold: {threshold:.4f}, rejecting {n_reject:,} tiles ({reject_pct}%)")

        if n_reject == 0:
            print("  Converged (no rejections).")
            break

        # Parse tile keys back from strings and add to excluded set
        new_excluded = set()
        for idx in np.where(reject_mask)[0]:
            key_str = tile_keys_flat[idx]
            # Parse "(folder, cam, img, tile)" string back to tuple
            new_excluded.add(key_str)
        excluded_keys |= new_excluded

        log_path = out_dir / f"round_{rnd}_rejected.txt"
        with open(log_path, "w") as f:
            for k in sorted(new_excluded):
                f.write(f"{k}\n")

    # Save final model
    model_path = config.OUTPUT_DIR / spec_key / f"group_{group_id}" / "memory_bank.npy"
    ensure_dir(model_path.parent)
    model.save(model_path)
    print(f"  Model saved: {model_path}")

    return model, excluded_keys


def _save_histogram(scores: np.ndarray, path: Path, round_num: int):
    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=100, edgecolor="black", alpha=0.7)
    plt.title(f"Anomaly Score Distribution - Round {round_num}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.axvline(np.percentile(scores, 95), color='r', linestyle='--',
                label=f'95th pct: {np.percentile(scores, 95):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
