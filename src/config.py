"""H-beam PatchCore configuration - v4: half-res + TensorRT + date-stratified."""
import os
import re
import math
from pathlib import Path

# == Paths ==
NAS_ROOT = Path(os.path.expanduser("~/nas_storage"))
PROJECT_ROOT = Path(os.path.expanduser("~/patchcore"))
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# == Camera groups (symmetric pairs, second cam is mirrored) ==
CAMERA_GROUPS = {
    1: {"name": "flange_top_inner",    "cams": [1, 10], "desc": "상부 플랜지 내면"},
    2: {"name": "fillet_top",          "cams": [2, 9],  "desc": "상부 필릿"},
    3: {"name": "flange_outer",        "cams": [3, 8],  "desc": "플랜지 외면"},
    4: {"name": "fillet_bottom",       "cams": [4, 7],  "desc": "하부 필릿"},
    5: {"name": "flange_bottom_inner", "cams": [5, 6],  "desc": "하부 플랜지 내면"},
}

MIRROR_CAM_INDEX = 1  # index in cams list (0-based)

# == Image / Tile settings ==
IMAGE_WIDTH_ORIGINAL = 1920
IMAGE_HEIGHT_ORIGINAL = 1200
RESIZE_SCALE = 0.5  # half resolution
IMAGE_WIDTH = int(IMAGE_WIDTH_ORIGINAL * RESIZE_SCALE)   # 960
IMAGE_HEIGHT = int(IMAGE_HEIGHT_ORIGINAL * RESIZE_SCALE)  # 600
TILE_SIZE = 256
TILE_STRIDE = 256

# == Head/tail trimming ==
TRIM_HEAD = 100
TRIM_TAIL = 100

# == Individual spec training ==
MIN_FOLDERS = 30
TARGET_IMAGES_PER_MODEL = 50000
FOLDER_SAMPLE_MAX = 500
IMAGE_SUBSAMPLE = 5

# == Tile mask ==
MASK_SAMPLE_COUNT = 20
MASK_BRIGHTNESS_THRESHOLD = 30

# == PatchCore ==
BACKBONE = "wide_resnet50_2"
FEATURE_LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.01
CORESET_PROJECTION_DIM = 128
CORESET_BATCH_SIZE = 200000

# == Self-validation ==
SELF_VAL_ROUNDS = 3
SELF_VAL_REJECT_PERCENT = 5.0

# == GPU ==
DEVICE = "cuda"
BATCH_SIZE = 512
NUM_WORKERS = 16
USE_DATA_PARALLEL = True

# == ONNX / TensorRT ==
USE_ONNX_TRT = True  # Use ONNX Runtime with TensorRT EP for inference
ONNX_MODEL_PATH = PROJECT_ROOT / "cache" / "backbone_half.onnx"
TRT_CACHE_DIR = PROJECT_ROOT / "cache" / "trt_engines"

# == W-beam inch-to-mm conversion ==
W_INCH_TO_MM = {
    "W10": 254, "W12": 305, "W14": 356, "W16": 406,
    "W18": 457, "W21": 533, "W24": 610, "W27": 686, "W30": 762,
}

EXCLUDE_PREFIXES = ["C", "SP"]


def parse_spec(folder_name: str):
    """Extract spec from folder name -> (raw_spec, width_mm, height_mm) or None."""
    m = re.search(r'[\s_]((H|W\d+x|HP\d+x)?(\d+)x(\d+))', folder_name)
    if not m:
        return None
    
    raw = m.group(1)
    prefix = m.group(2) or ""
    d1 = int(m.group(3))
    d2 = int(m.group(4))
    
    for ex in EXCLUDE_PREFIXES:
        if raw.startswith(ex) and not raw.startswith("CP"):
            return None
    
    wm = re.match(r'^W(\d+)x', raw)
    if wm:
        w_key = f"W{wm.group(1)}"
        width_mm = W_INCH_TO_MM.get(w_key)
        if width_mm is None:
            return None
        height_mm = width_mm
        return (raw, width_mm, height_mm)
    
    hpm = re.match(r'^HP(\d+)x', raw)
    if hpm:
        hp_key = f"W{hpm.group(1)}"
        width_mm = W_INCH_TO_MM.get(hp_key, d1)
        height_mm = width_mm
        return (raw, width_mm, height_mm)
    
    width_mm = d1
    height_mm = d2
    return (raw, width_mm, height_mm)


def spec_key(raw_spec: str) -> str:
    """Normalize spec name to canonical key."""
    if raw_spec.startswith("H") and not raw_spec.startswith("HP"):
        bare = raw_spec[1:]
        if re.match(r'^\d+x\d+', bare):
            return bare
    return raw_spec


def find_nearest_spec(target_w, target_h, available_specs):
    """Find nearest available spec by Euclidean distance on (width, height) mm."""
    best_key = None
    best_dist = float('inf')
    for sk, (w, h) in available_specs.items():
        dist = math.sqrt((target_w - w) ** 2 + (target_h - h) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_key = sk
    return best_key, best_dist
SELF_VAL_MAD_K = 3.5
SELF_VAL_MAX_REJECT_PCT = 5.0
