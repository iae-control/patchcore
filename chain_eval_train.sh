#!/bin/bash
set -e
cd ~/patchcore
echo [CHAIN] Waiting for eval to finish...
while pgrep -f eval_all_v3 > /dev/null 2>&1; do sleep 10; done
echo [CHAIN] Eval done. Starting reordered training...
echo [CHAIN] 700x300 PRIORITY, large specs first
nohup ./venv/bin/python -u train_v3_reorder.py --all --resume > train_v3_reorder.log 2>&1 &
echo [CHAIN] Training started PID=$!
