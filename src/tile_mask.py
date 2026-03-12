"""Automatic tile mask generation - v4: half-resolution resize."""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm import tqdm
from src import config
from src.utils import get_image_paths, tile_positions, ensure_dir


def compute_tile_mask(folder: dict, cam_ids: List[int]) -> np.ndarray:
    """Compute boolean mask for tile positions using first N images.
    Resizes images to half resolution before computing.
    
    Returns: 1D bool array of length num_tiles. True = valid tile.
    """
    positions = tile_positions(
        config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
        config.TILE_SIZE, config.TILE_STRIDE
    )
    num_tiles = len(positions)
    brightness_accum = np.zeros(num_tiles, dtype=np.float64)
    count = 0

    resize_to = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

    all_samples = []
    for cam_id in cam_ids:
        images = get_image_paths(folder["path"], cam_id)
        all_samples.extend(images[:config.MASK_SAMPLE_COUNT])

    for img_path in tqdm(all_samples, desc=f"  Mask {folder['name']}", 
                         leave=False, unit="img", dynamic_ncols=True):
        img = Image.open(img_path).convert("L")
        # Resize to half resolution
        if img.size != resize_to:
            img = img.resize(resize_to, Image.LANCZOS)
        img = np.array(img)
        for t_idx, (tx, ty) in enumerate(positions):
            tile = img[ty:ty + config.TILE_SIZE, tx:tx + config.TILE_SIZE]
            brightness_accum[t_idx] += tile.mean()
        count += 1

    if count == 0:
        return np.ones(num_tiles, dtype=bool)

    avg_brightness = brightness_accum / count
    mask = avg_brightness >= config.MASK_BRIGHTNESS_THRESHOLD
    return mask


def load_or_compute_masks(
    folders: List[dict],
    cam_ids: List[int],
    group_name: str,
) -> Dict[str, np.ndarray]:
    """Load cached masks or compute them. Returns {folder_name: mask}."""
    mask_dir = config.OUTPUT_DIR / "masks" / group_name
    ensure_dir(mask_dir)

    masks = {}
    for folder in tqdm(folders, desc="Tile masks", leave=False, 
                       unit="folder", dynamic_ncols=True):
        fname = folder["name"]
        cache_path = mask_dir / f"{fname}.npy"
        if cache_path.exists():
            masks[fname] = np.load(cache_path)
        else:
            mask = compute_tile_mask(folder, cam_ids)
            np.save(cache_path, mask)
            masks[fname] = mask
    return masks
