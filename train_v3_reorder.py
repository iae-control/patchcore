#!/usr/bin/env python3
import argparse,sys,time,json,torch
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from src import config
from src.utils import discover_all_specs, get_trainable_specs, build_fallback_map, adaptive_subsample, ensure_dir, load_progress, save_progress
from src.tile_mask import load_or_compute_masks
from src.patchcore import FeatureExtractor
from src.self_validation import self_validation_loop
PRIORITY_SPECS = ['700X300', '700x300']
def train_spec_camera(spec_key, folders, group_id, extractor, subsample_step):
    grp = config.CAMERA_GROUPS[group_id]
    cam_ids = grp['cams']; mirror_cam_id = cam_ids[config.MIRROR_CAM_INDEX]; group_name = grp['name']
    print(f'\n{{"="*60}}\nSpec: {spec_key} | Group {group_id}: {grp["desc"]} ({group_name})\n  Cameras: {cam_ids}, Folders: {len(folders)}, Step: {subsample_step}\n{{"="*60}}')
    model_path = config.OUTPUT_DIR / spec_key / f'group_{group_id}' / 'memory_bank.npy'
    if model_path.exists(): print(f'  Already trained -> SKIP'); return True
    print('\n[1/2] Computing tile masks...')
    tile_masks = load_or_compute_masks(folders, cam_ids, group_name)
    print('\n[2/2] Self-validation training...')
    try:
        model, excluded = self_validation_loop(folders=folders, cam_ids=cam_ids, mirror_cam_id=mirror_cam_id, tile_masks=tile_masks, spec_key=spec_key, group_id=group_id, group_name=group_name, extractor=extractor, subsample_step=subsample_step)
        print(f'  OK {spec_key}/group_{group_id} Excluded: {len(excluded)}')
        return True
    except Exception as e:
        print(f'  FAIL {spec_key}/group_{group_id}: {e}')
        import traceback; traceback.print_exc()
        return False
def reorder_specs(trainable):
    priority = []; rest = []
    for item in trainable:
        sk = item[0]
        if any(sk.upper() == p.upper() for p in PRIORITY_SPECS): priority.append(item)
        else: rest.append(item)
    rest.sort(key=lambda x: x[2]*x[3], reverse=True)
    reordered = priority + rest
    print('\n=== Training Order (large first, 700x300 priority) ===')
    for i, (sk, n, w, h) in enumerate(reordered):
        tag = ' * PRIORITY' if any(sk.upper() == p.upper() for p in PRIORITY_SPECS) else ''
        print(f'  {i+1:3d}. {sk:20s} {n:5d} folders ({w}x{h}mm){tag}')
    return reordered
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=str); parser.add_argument('--group', type=int)
    parser.add_argument('--all', action='store_true'); parser.add_argument('--list', action='store_true')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(f'PatchCore v3 REORDERED - 700x300 PRIORITY\nStarted: {datetime.now()}')
    print('Scanning NAS...'); t0=time.time(); all_specs=discover_all_specs(); elapsed=time.time()-t0
    print(f'Scan: {sum(len(v) for v in all_specs.values()):,} folders, {len(all_specs)} specs in {elapsed:.1f}s')
    trainable, sparse = get_trainable_specs(all_specs)
    fallback_map = build_fallback_map(trainable, sparse)
    trainable = reorder_specs(trainable)
    if args.list: return
    if not args.spec and not args.all: parser.error('Need --spec or --all or --list')
    device = config.DEVICE
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; config.DEVICE = 'cpu'
    else: print(f'Device: {device} ({torch.cuda.get_device_name(0)})')
    print('Loading backbone...'); extractor = FeatureExtractor(device=device)
    if args.spec:
        specs_to_train = [(sk,n,w,h) for sk,n,w,h in trainable if sk == args.spec]
        if not specs_to_train: print(f'Unknown: {args.spec}'); sys.exit(1)
    else: specs_to_train = trainable
    cam_groups = [args.group] if args.group else list(range(1, 6))
    total_jobs = len(specs_to_train) * len(cam_groups)
    print(f'Plan: {len(specs_to_train)} specs x {len(cam_groups)} cams = {total_jobs} models, Resume={args.resume}')
    if args.dry_run:
        for sk,n,w,h in specs_to_train:
            for gid in cam_groups:
                mp = config.OUTPUT_DIR / sk / f'group_{gid}' / 'memory_bank.npy'
                print(f'  {sk}/group_{gid}: {"EXISTS" if mp.exists() else "TODO"}')
        return
    progress_path = config.OUTPUT_DIR / 'training_progress.json'
    progress = load_progress(progress_path)
    completed=0; failed=0; skipped=0
    for si, (sk, n_folders, w, h) in enumerate(specs_to_train):
        folders = all_specs[sk]
        for gid in cam_groups:
            job_key = f'{sk}/group_{gid}'
            if args.resume and job_key in progress['completed']: skipped+=1; continue
            print(f'\n>>> [{si*len(cam_groups)+cam_groups.index(gid)+1}/{total_jobs}] {job_key}')
            cam_ids = config.CAMERA_GROUPS[gid]['cams']
            sampled, step = adaptive_subsample(folders, cam_ids)
            t_start = time.time()
            ok = train_spec_camera(sk, sampled, gid, extractor, step)
            elapsed = time.time() - t_start
            if ok: completed+=1; progress['completed'].append(job_key)
            else: failed+=1; progress.setdefault('failed',[]).append(job_key)
            save_progress(progress_path, progress)
            print(f'  Time: {elapsed/60:.1f}min Done:{completed} Fail:{failed} Skip:{skipped}')
    print(f'\nTRAINING COMPLETE: Done={completed} Fail={failed} Skip={skipped}')
if __name__ == '__main__': main()
