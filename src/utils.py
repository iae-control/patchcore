"""Utility functions - v4: date-stratified sampling + half-res support."""
import re
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from src import config


def discover_all_specs() -> Dict[str, list]:
    """Scan NAS and return {spec_key: [folder_dicts]} for all valid specs."""
    nas = config.NAS_ROOT
    if not nas.exists():
        raise FileNotFoundError(f"NAS root not found: {nas}")

    specs: Dict[str, list] = {}

    for entry in sorted(nas.iterdir()):
        if not entry.is_dir():
            continue
        
        if re.match(r'^\d{8}$', entry.name):
            try:
                subs = sorted(entry.iterdir())
            except PermissionError:
                continue
            for sub in subs:
                if sub.is_dir():
                    _add_if_valid(sub, specs)
        else:
            _add_if_valid(entry, specs)

    return specs


def _add_if_valid(path: Path, specs: Dict[str, list]):
    """Check folder validity and add to specs dict."""
    if not (path / "camera_1").is_dir():
        return
    
    parsed = config.parse_spec(path.name)
    if parsed is None:
        return
    
    raw_spec, w_mm, h_mm = parsed
    sk = config.spec_key(raw_spec)
    
    entry = {
        "path": path,
        "name": path.name,
        "raw_spec": raw_spec,
        "spec_key": sk,
        "width_mm": w_mm,
        "height_mm": h_mm,
    }
    
    if sk not in specs:
        specs[sk] = []
    specs[sk].append(entry)


def _extract_date(folder_name: str) -> str:
    """Extract date (YYYYMMDD) from folder name. Returns 'unknown' if not found."""
    m = re.match(r'^(\d{8})', folder_name)
    if m:
        return m.group(1)
    return "unknown"


def get_trainable_specs(all_specs: Dict[str, list]) -> Tuple[list, list]:
    """Split specs into trainable (>=MIN_FOLDERS) and sparse (<MIN_FOLDERS)."""
    trainable = []
    sparse = []
    
    for sk, folders in sorted(all_specs.items()):
        n = len(folders)
        w = folders[0]["width_mm"]
        h = folders[0]["height_mm"]
        if n >= config.MIN_FOLDERS:
            trainable.append((sk, n, w, h))
        else:
            sparse.append((sk, n, w, h))
    
    return trainable, sparse


def build_fallback_map(trainable, sparse) -> Dict[str, str]:
    """Map each sparse spec to nearest trainable spec by mm distance."""
    available = {sk: (w, h) for sk, _, w, h in trainable}
    fallback = {}
    for sk, _, w, h in sparse:
        nearest, dist = config.find_nearest_spec(w, h, available)
        fallback[sk] = nearest
    return fallback


def adaptive_subsample(
    folders: list,
    cam_ids: list,
    target_images: int = None,
) -> Tuple[list, int]:
    """Date-stratified folder sampling + adaptive subsample rate.
    
    v4: Ensures equal representation across production dates.
    """
    target = target_images or config.TARGET_IMAGES_PER_MODEL
    max_folders = config.FOLDER_SAMPLE_MAX
    
    n_cams = len(cam_ids)
    n_folders = len(folders)
    EST_IMAGES_PER_FOLDER_PER_CAM = 700
    
    total_available = n_folders * n_cams * EST_IMAGES_PER_FOLDER_PER_CAM
    
    if total_available <= target:
        return folders, 1
    
    if n_folders <= max_folders:
        sampled = folders
    else:
        # === Date-stratified sampling ===
        # Group folders by production date
        by_date = defaultdict(list)
        for f in folders:
            date = _extract_date(f["name"])
            by_date[date].append(f)
        
        dates = sorted(by_date.keys())
        n_dates = len(dates)
        
        # Allocate quota per date (equal split, remainder distributed)
        base_per_date = max_folders // n_dates
        remainder = max_folders % n_dates
        
        rng = random.Random(42)
        sampled = []
        
        for i, date in enumerate(dates):
            date_folders = by_date[date]
            quota = base_per_date + (1 if i < remainder else 0)
            
            if len(date_folders) <= quota:
                sampled.extend(date_folders)
            else:
                # Random sample within this date
                sampled.extend(rng.sample(date_folders, quota))
        
        # If we got fewer than max_folders (some dates had few folders),
        # fill remaining slots from dates with surplus
        if len(sampled) < max_folders:
            already = set(f["name"] for f in sampled)
            remaining_pool = [f for f in folders if f["name"] not in already]
            shortfall = max_folders - len(sampled)
            if remaining_pool and shortfall > 0:
                sampled.extend(rng.sample(remaining_pool, min(shortfall, len(remaining_pool))))
        
        sampled = sorted(sampled, key=lambda f: f["name"])
        
        print(f"  Date-stratified sampling: {n_dates} dates, "
              f"{len(sampled)} folders (from {n_folders} total)")
    
    images_if_all = len(sampled) * n_cams * EST_IMAGES_PER_FOLDER_PER_CAM
    step = max(1, round(images_if_all / target))
    
    actual_estimate = images_if_all // step
    print(f"  Adaptive subsample: {len(sampled)} folders, step={step}, "
          f"est. {actual_estimate:,} images (target: {target:,})")
    
    return sampled, step


def _natural_sort_key(path: Path):
    """Natural sort: '10_99.jpg' < '10_100.jpg'."""
    import re as _re
    parts = _re.split(r'(\d+)', path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_image_paths(folder: Path, cam_id: int, subsample_step: int = None) -> list:
    """Get sorted image paths for a camera, trimming head/tail."""
    cam_dir = folder / f"camera_{cam_id}"
    if not cam_dir.is_dir():
        return []
    
    images = sorted(
        [p for p in cam_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
        key=_natural_sort_key
    )
    
    if len(images) <= config.TRIM_HEAD + config.TRIM_TAIL:
        return []
    images = images[config.TRIM_HEAD : len(images) - config.TRIM_TAIL]
    
    step = subsample_step or getattr(config, 'IMAGE_SUBSAMPLE', 1)
    if step > 1:
        images = images[::step]
    
    return images


def tile_positions(img_w, img_h, tile_size, stride):
    """Compute (x, y) top-left positions of tiles."""
    positions = []
    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            positions.append((x, y))
    return positions


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# == Checkpoint helpers ==

def save_checkpoint(ckpt_path: Path, data: dict):
    ensure_dir(ckpt_path.parent)
    np.savez_compressed(str(ckpt_path), **data)
    print(f"  Checkpoint saved: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(ckpt_path: Path) -> Optional[dict]:
    if not ckpt_path.exists():
        return None
    print(f"  Resuming from checkpoint: {ckpt_path}")
    data = dict(np.load(str(ckpt_path), allow_pickle=True))
    return data


def remove_checkpoint(ckpt_path: Path):
    if ckpt_path.exists():
        ckpt_path.unlink()


# == Training progress tracker ==

def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress_path: Path, progress: dict):
    ensure_dir(progress_path.parent)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
