#!/bin/bash
cd ~/patchcore
source venv/bin/activate
export PYTHONUNBUFFERED=1
python -u train.py --all --resume 2>&1 | tee train_v3.log
