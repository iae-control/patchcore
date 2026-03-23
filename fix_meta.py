"""Fix training_meta.json that failed due to float32 serialization error."""
import json
import numpy as np

output_dir = '/home/dk-sdd/patchcore/output_v6/596x199/group_1'
mb = np.load(f'{output_dir}/memory_bank.npy')
print(f'Memory bank: {mb.shape}')

meta = {
    "version": "v6-tile",
    "group_id": 1,
    "cam_ids": [1, 10],
    "mirror_cam": 10,
    "tile_size": 128,
    "tile_stride": 128,
    "image_resolution": [1920, 1200],
    "tiles_per_image": 135,
    "tiles_grid": [15, 9],
    "total_images": 28977,
    "total_tiles_extracted": 3842063,
    "tiles_after_selfval": 3842044,
    "memory_bank_shape": list(mb.shape),
    "coreset_ratio": 0.01,
    "self_val_rounds": 1,
    "self_val_mad_k": 3.5,
    "threshold_mad": 0.6812,
}

with open(f'{output_dir}/training_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Meta saved successfully.')
print(f'Threshold: {meta["threshold_mad"]}')
