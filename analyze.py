import json, numpy as np

with open('/home/dk-sdd/patchcore/experiment_v2/results.json') as f:
    ALL = json.load(f)

SPOT = ['spot','big_spot','stain','bright']
LINE = ['scratch','thick_scr','crack','multi_scr']

methods = list(ALL.keys())
thresholds = [1.2, 1.3, 1.5, 2.0]

header = f"{'Method':<25}"
for t in thresholds:
    header += f"  thr={t}(S/L/T)  "
print(header)
print("-"*120)

for m in methods:
    row = f"{m:<25}"
    for thr in thresholds:
        sd = sl = 0
        for dk in SPOT:
            for ti, s, n in ALL[m][dk]:
                r = s/n if n != 0 else s
                if m.startswith('Adaptive'):
                    if s > thr: sd += 1
                else:
                    if r > thr: sd += 1
        for dk in LINE:
            for ti, s, n in ALL[m][dk]:
                r = s/n if n != 0 else s
                if m.startswith('Adaptive'):
                    if s > thr: sl += 1
                else:
                    if r > thr: sl += 1
        row += f"  {sd:>2}/{sl:>2}/{sd+sl:>2}       "
    print(row)
