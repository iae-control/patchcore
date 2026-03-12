#!/usr/bin/env python3
"""GPU1 reverse-order trainer. Runs specs from smallest to largest."""
import subprocess, sys, json
from pathlib import Path

SPECS_REVERSE = [
    '4x30', '4x34', '150x75', '194x150', '200x200', '203x203',
    '208x202', '206x204', '244x175', '210x205', '216x206',
    '300x150', '222x209', '294x200', '298x201', '350x175',
    '250x250', '250x255', '253x254', '260x256', '396x199',
    '199x396', '400x200', '200x400', '340x250', '446x199',
    '450x200', '300x300', '300x305', 'W12x8x40', '310x310',
    '496x199', '500x200', '506x201', '390x300', '596x199',
    '600x200', '606x201', '350x350', '612x202', '350x357',
    '440x300', '481x300', '488x300', '400x400', '406x403',
    '414x405', '428x407',
]

progress_path = Path('/home/dk-sdd/patchcore/output/training_progress.json')
log_path = Path('/home/dk-sdd/patchcore/train_gpu1_reverse.log')

for spec in SPECS_REVERSE:
    # Check if all 5 groups done
    all_done = True
    for g in range(1, 6):
        model = Path(f'/home/dk-sdd/patchcore/output/{spec}/group_{g}/memory_bank.npy')
        if not model.exists():
            all_done = False
            break
    if all_done:
        print(f'SKIP {spec} (all groups done)', flush=True)
        continue
    
    print(f'>>> Training {spec}', flush=True)
    cmd = [
        '/home/dk-sdd/patchcore/venv/bin/python', '-u',
        '/home/dk-sdd/patchcore/train_v4_reorder.py',
        '--spec', spec, '--resume'
    ]
    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    
    with open(log_path, 'a') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    
    if proc.returncode != 0:
        print(f'FAIL {spec} (exit {proc.returncode})', flush=True)
    else:
        print(f'DONE {spec}', flush=True)

print('=== GPU1 reverse trainer finished ===')
