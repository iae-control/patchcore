"""Tile dataset - v4: half-resolution + RAM preload."""
import numpy as np
import psutil
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src import config
from src.ram_lock import RAMPreloadLock
from src.utils import get_image_paths, tile_positions

# RAM limit for image cache (bytes)
RAM_CACHE_LIMIT = 150 * 1024**3  # 150 GB


class TileDataset(Dataset):
    """Dataset that yields tiles from H-beam camera images with RAM preload."""

    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    RESIZE_TO = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)  # (960, 600)

    def __init__(
        self,
        folders: List[dict],
        cam_ids: List[int],
        mirror_cam_id: int,
        tile_mask: Optional[dict] = None,
        exclude_keys: Optional[Set] = None,
        subsample_step: int = None,
    ):
        self.tiles_info: List[dict] = []
        self.positions = tile_positions(
            config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
            config.TILE_SIZE, config.TILE_STRIDE
        )
        self.mirror_cam_id = mirror_cam_id
        self.exclude_keys = exclude_keys or set()
        self._img_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = False

        # Collect unique image paths
        unique_paths = set()

        for folder in folders:
            folder_name = folder["name"]
            mask = tile_mask.get(folder_name) if tile_mask else None

            for cam_id in cam_ids:
                images = get_image_paths(folder["path"], cam_id,
                                        subsample_step=subsample_step)
                for img_idx, img_path in enumerate(images):
                    for tile_idx, (tx, ty) in enumerate(self.positions):
                        if mask is not None and not mask[tile_idx]:
                            continue
                        key = (folder_name, cam_id, img_idx, tile_idx)
                        key_str = str(key)
                        if key_str in self.exclude_keys or key in self.exclude_keys:
                            continue
                        self.tiles_info.append({
                            "path": img_path,
                            "folder": folder_name,
                            "cam_id": cam_id,
                            "img_idx": img_idx,
                            "tile_idx": tile_idx,
                            "tile_x": tx,
                            "tile_y": ty,
                            "mirror": cam_id == mirror_cam_id,
                        })
                        unique_paths.add(img_path)

        # Preload images into RAM
        self._preload(sorted(unique_paths))

    def _preload(self, paths: list):
        """Preload and resize images into RAM cache (with cross-process lock)."""
        with RAMPreloadLock():
            self._preload_inner(paths)

    def _preload_inner(self, paths: list):
        """Actual preload logic."""
        avail = psutil.virtual_memory().available
        limit = min(RAM_CACHE_LIMIT, int(avail * 0.85))
        # Estimate per-image size: 960*600*3 = ~1.7MB numpy array
        est_per_img = config.IMAGE_WIDTH * config.IMAGE_HEIGHT * 3
        max_images = limit // est_per_img
        n = min(len(paths), max_images)

        if n < len(paths):
            print(f"  RAM preload: {n}/{len(paths)} images (limit {limit/1024**3:.0f}GB)")
        else:
            print(f"  RAM preload: all {n} images ({n * est_per_img / 1024**3:.1f}GB)")

        loaded = 0
        for p in paths[:n]:
            try:
                img = Image.open(p).convert("RGB")
                if img.size != self.RESIZE_TO:
                    img = img.resize(self.RESIZE_TO, Image.LANCZOS)
                self._img_cache[p] = np.array(img)
                loaded += 1
            except Exception:
                continue

        self._cache_enabled = loaded > 0
        print(f"  RAM preload done: {loaded} images cached")

    def _get_image(self, path: str) -> Image.Image:
        """Get image from cache or disk."""
        if self._cache_enabled and path in self._img_cache:
            return Image.fromarray(self._img_cache[path])
        # Fallback to disk
        img = Image.open(path).convert("RGB")
        if img.size != self.RESIZE_TO:
            img = img.resize(self.RESIZE_TO, Image.LANCZOS)
        return img

    def clear_cache(self):
        """Free RAM cache."""
        self._img_cache.clear()
        self._cache_enabled = False

    def __len__(self):
        return len(self.tiles_info)

    def __getitem__(self, idx):
        info = self.tiles_info[idx]
        img = self._get_image(info["path"])

        tile = img.crop((
            info["tile_x"], info["tile_y"],
            info["tile_x"] + config.TILE_SIZE,
            info["tile_y"] + config.TILE_SIZE
        ))
        if info["mirror"]:
            tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
        tensor = self.TRANSFORM(tile)
        meta = {
            "folder": info["folder"],
            "cam_id": info["cam_id"],
            "img_idx": info["img_idx"],
            "tile_idx": info["tile_idx"],
        }
        return tensor, meta

    def get_key(self, idx) -> Tuple:
        info = self.tiles_info[idx]
        img_name = Path(info["path"]).name if "path" in info else "img_%05d" % info["img_idx"]
        return (info["folder"], img_name, info["cam_id"], info["img_idx"], info["tile_idx"])
