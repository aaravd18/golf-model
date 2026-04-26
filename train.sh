#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
pip install brotli
# pip install runpod

rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128


HESSIAN_CLIP_LAMBDA=0.175 \
LOOP_PHASE2_AT=0.65 \
PARALLEL_RESIDUAL_START=7 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=1200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 

# sleep 300

# python3 -c "
# import os, runpod
# runpod.api_key = os.environ['RUNPOD_API_KEY']
# runpod.stop_pod(os.environ['RUNPOD_POD_ID'])
# "