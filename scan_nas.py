#!/usr/bin/env python3
import os, json
NAS = os.path.expanduser("~/nas_storage")
specs = ['199x396','200x200','203x203','206x204','200x400','150x75','194x150']
train_dates = {'199x396':'20250617','200x200':'20250527','203x203':'20251102',
               '206x204':'20251102','200x400':'20250617','150x75':'20251105','194x150':'20251103'}
result = {}
# Scan date folders
dates = sorted([d for d in os.listdir(NAS) if d.isdigit() and len(d)==8 and os.path.isdir(os.path.join(NAS,d))])
for spec in specs:
    result[spec] = {'train_date': train_dates[spec], 'test_folders': []}
    for date in dates:
        dp = os.path.join(NAS, date)
        try:
            subs = os.listdir(dp)
        except:
            continue
        for s in subs:
            if spec not in s:
                continue
            sp = os.path.join(dp, s)
            cam1 = os.path.join(sp, 'camera_1')
            if os.path.isdir(cam1):
                try:
                    cnt = len([f for f in os.listdir(cam1) if f.endswith('.jpg')])
                except:
                    cnt = 0
                if cnt > 50:
                    is_train = date == train_dates[spec]
                    result[spec]['test_folders'].append({
                        'date': date, 'path': sp, 'cam1_count': cnt, 'is_train': is_train
                    })
                    if not is_train and len([f for f in result[spec]['test_folders'] if not f['is_train']]) >= 3:
                        break
    print(f"{spec}: {len(result[spec]['test_folders'])} folders found")
    for f in result[spec]['test_folders']:
        tag = "TRAIN" if f['is_train'] else "TEST"
        print(f"  [{tag}] {f['date']}/{os.path.basename(f['path'])} ({f['cam1_count']} imgs)")
with open(os.path.expanduser("~/patchcore/nas_scan.json"), "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved to ~/patchcore/nas_scan.json")
