# PatchCore H-beam Surface Defect Detection

PatchCore-based anomaly detection system for H-beam surface defect inspection at steel manufacturing lines.

## Overview

This system detects surface defects on H-beam steel products using [PatchCore](https://arxiv.org/abs/2106.08265) (CVPR 2022), an unsupervised anomaly detection method that requires **only normal images** for training — no defect labels needed.

### Key Features

- **Tile-based detection**: Splits 1920x1200 images into 256x256 tiles, preserving fine defect details
- **Per-spec models**: 53 H-beam specifications x 5 camera groups = 265 dedicated models
- **Self-validation**: Automatic 3-round data refinement that removes defect-contaminated images without labels
- **Ens-MAX ensemble**: 8 independent statistical metrics with z-score MAX decision for robust defect judgment
- **Fast model switching**: Shared CNN backbone (WideResNet-50) + swappable memory banks per specification

## Architecture

```
H-beam image (1920x1200)
    |\n    v
Tile split (256x256 patches)
    |\n    v
Feature extraction (WideResNet-50, layers 2+3)
    |\n    v
kNN distance to memory bank (coreset)
    |\n    v
Anomaly score map -> Ens-MAX judgment
```

## Project Structure

```
src/
  config.py          # Configuration (NAS paths, camera groups, thresholds)
  dataset.py         # NAS data loading with RAM preload
  patchcore.py       # Core PatchCore implementation
  self_validation.py # Self-validation data refinement
  tile_mask.py       # Dynamic tile masking per camera group
  utils.py           # Spec discovery, trainable spec filtering
train_v4_reorder.py  # Main training script (large-first ordering)
train_gpu1_reverse.py# Reverse-order training for dual GPU setup
inference.py         # Single-image inference with visualization
eval_all_v3.py       # Batch evaluation across all specs
monitor.py           # Training progress monitor
scan_nas.py          # NAS folder scanner
```

## Training

### Single spec
```bash
CUDA_VISIBLE_DEVICES=0 python train_v4_reorder.py --spec 700x300 --resume
```

### All specs (dual GPU)
```bash
# GPU 0: large specs first
CUDA_VISIBLE_DEVICES=0 nohup python -u train_v4_reorder.py --all --resume >> train_gpu0.log 2>&1 &

# GPU 1: small specs first (reverse order)
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gpu1_reverse.py >> train_gpu1.log 2>&1 &
```

### Training output
```
output/{spec_key}/group_{1-5}/
  memory_bank.npy    # Coreset features
  threshold.json     # MAD-based threshold (k=3.5)
  self_val_stats.json# Self-validation statistics
```

## Inference

```python
from src.patchcore import PatchCoreModel

model = PatchCoreModel("output/700x300/group_1")
score_map, max_score = model.predict(image)
```

## Camera Groups

| Group | Cameras | Surface |
|-------|---------|--------|
| 1 | 1, 10 | Web front |
| 2 | 2, 9 | Flange top outer |
| 3 | 3, 8 | Flange top inner |
| 4 | 4, 7 | Fillet bottom |
| 5 | 5, 6 | Flange bottom inner |

## Self-Validation

The self-validation module automatically detects and removes defect-contaminated tiles from training data through iterative refinement:

1. **Round 0**: Train on all data, compute anomaly scores
2. **Round 1**: Exclude high-score tiles (MAD threshold, k=3.5), retrain
3. **Round 2**: Final refinement pass

This eliminates the need for manual defect labeling in training data.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- NVIDIA GPU with 24GB+ VRAM (tested on A40, L40S)
- 128GB+ RAM (for NAS image preloading)

## Hardware

Developed and tested on:
- 2x NVIDIA A40 (48GB each)
- 256GB RAM
- NFS-mounted NAS (Synology) for image storage

## References

- [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) - Roth et al., CVPR 2022
- [WideResNet](https://arxiv.org/abs/1605.07146) - Zagoruyko & Komodakis, 2016

## License

MIT

## Author

Sanghyuk Jung (jsh@iae.re.kr)  
Institute for Advanced Engineering (IAE)
