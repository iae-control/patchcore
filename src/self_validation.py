"""Self-validation v4: adaptive threshold (MAD-based) instead of fixed percentile.

Changes from v3:
- Replaced fixed `reject_pct` (top 5%) with MAD-based adaptive threshold
- Threshold = median + k * MAD  (k defaults to config.SELF_VAL_MAD_K, typically 3.5~4.0)
- If no scores exceed threshold, round is skipped (natural convergence)
- Histogram now shows both old 95th-pct line (reference) and new MAD threshold
- Added max_reject_pct safety cap to prevent runaway removal
"""
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


def _compute_mad_threshold(scores: np.ndarray, k: float) -> float:
    """Compute adaptive threshold: median + k * MAD.
    
    MAD (Median Absolute Deviation) is robust to outliers, unlike std.
    k=3.5 is conservative (~0.02% FPR for normal dist), k=3.0 is moderate.
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    # Scale MAD to be consistent with std for normal distributions
    mad_std = 1.4826 * mad
    threshold = median + k * mad_std
    return threshold


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
    reject_pct: float = None,  # kept for API compat, but unused in v4
    subsample_step: int = None,
) -> Tuple[PatchCoreModel, Set]:
    """Run self-validation loop with adaptive MAD-based threshold."""
    rounds = rounds or config.SELF_VAL_ROUNDS
    mad_k = getattr(config, 'SELF_VAL_MAD_K', 3.5)
    max_reject_pct = getattr(config, 'SELF_VAL_MAX_REJECT_PCT', 5.0)
    
    out_dir = config.OUTPUT_DIR / spec_key / f"group_{group_id}" / "self_val"
    ckpt_dir = config.CHECKPOINT_DIR / spec_key / f"group_{group_id}"
    ensure_dir(out_dir)

    excluded_keys: Set[tuple] = set()
    model = PatchCoreModel()

    for rnd in range(rounds):
        print(f"\n  \U0001f50d Self-validation round {rnd}/{rounds - 1} "
              f"(excluded: {len(excluded_keys)} tiles) \U0001f50d")

        # Check for feature checkpoint from previous crashed run
        ckpt_path = ckpt_dir / f"round_{rnd}_features.npz"
        ckpt = load_checkpoint(ckpt_path)
        
        if ckpt is not None:
            features = ckpt["features"]
            tile_keys_flat = ckpt["tile_keys"]
            print(f"  Resumed {len(features):,} features from checkpoint")
        else:
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
            
            tile_keys_flat = np.array([str(dataset.get_key(i)) for i in range(len(dataset))])
            save_checkpoint(ckpt_path, {
                "features": features,
                "tile_keys": tile_keys_flat,
            })

        print(f"  Fitting coreset on {len(features):,} features...")
        model.fit(features)
        scores = model.score(features)

        remove_checkpoint(ckpt_path)

        # Compute adaptive threshold
        threshold = _compute_mad_threshold(scores, k=mad_k)
        
        # Safety cap: never reject more than max_reject_pct
        pct_threshold = np.percentile(scores, 100 - max_reject_pct)
        effective_threshold = max(threshold, pct_threshold)
        
        reject_mask = scores >= effective_threshold
        n_reject = reject_mask.sum()
        reject_ratio = n_reject / len(scores) * 100
        
        # Stats for logging
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))
        mad_std = 1.4826 * mad
        
        print(f"  Score stats: median={median_score:.4f}, MAD_std={mad_std:.4f}")
        print(f"  MAD threshold (k={mad_k}): {threshold:.4f}")
        if effective_threshold > threshold:
            print(f"  Safety cap applied: {pct_threshold:.4f} (max {max_reject_pct}%)")
        print(f"  Effective threshold: {effective_threshold:.4f}, "
              f"rejecting {n_reject:,} tiles ({reject_ratio:.2f}%)")

        _save_histogram_v4(scores, out_dir / f"round_{rnd}_hist.png", rnd,
                           threshold, effective_threshold, median_score, mad_std, mad_k)

        if n_reject == 0:
            print("  Converged (no rejections). Distribution is clean.")
            break

        new_excluded = set()
        for idx in np.where(reject_mask)[0]:
            key_str = tile_keys_flat[idx]
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


def _save_histogram_v4(scores: np.ndarray, path: Path, round_num: int,
                       mad_threshold: float, effective_threshold: float,
                       median: float, mad_std: float, k: float):
    plt.figure(figsize=(12, 5))
    plt.hist(scores, bins=100, edgecolor="black", alpha=0.7)
    plt.title(f"Anomaly Score Distribution - Round {round_num} (Adaptive MAD Threshold)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    
    # Reference: old 95th percentile (for comparison)
    pct95 = np.percentile(scores, 95)
    plt.axvline(pct95, color='gray', linestyle=':', alpha=0.5,
                label=f'95th pct (old): {pct95:.4f}')
    
    # Median line
    plt.axvline(median, color='blue', linestyle='-', alpha=0.4,
                label=f'Median: {median:.4f}')
    
    # MAD threshold
    plt.axvline(mad_threshold, color='orange', linestyle='--',
                label=f'MAD threshold (k={k}): {mad_threshold:.4f}')
    
    # Effective threshold (may differ if safety cap applied)
    if abs(effective_threshold - mad_threshold) > 1e-6:
        plt.axvline(effective_threshold, color='red', linestyle='--',
                    label=f'Effective (capped): {effective_threshold:.4f}')
    else:
        plt.axvline(effective_threshold, color='red', linestyle='--',
                    label=f'Effective: {effective_threshold:.4f}')
    
    n_rejected = (scores >= effective_threshold).sum()
    reject_pct = n_rejected / len(scores) * 100
    plt.text(0.02, 0.95, f'Rejected: {n_rejected:,} ({reject_pct:.2f}%)\n'
             f'MAD_std: {mad_std:.4f}',
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
