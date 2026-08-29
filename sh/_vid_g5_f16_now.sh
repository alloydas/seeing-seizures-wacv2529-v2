#!/bin/bash
# Pull vid_g5_f16 forward out of sh/_abl_frames_grading.sh and run it on GPU1 now.
#
# That script's GPU1 stream runs 5-class 8 -> 16 -> 32 -> 64 strictly sequentially, so
# f16 would wait for vid_g5_f8 (7/12) to finish. With load down to ~42 on 32 cores
# after yesterday's backlog cleared, the machine has room for a second job per GPU.
#
# f16 chosen because it pairs with vid_g3_f16 already running on GPU0: that gives both
# grading tasks at the same frame budget, which is the row most directly comparable to
# the detection anchor (0.9681 +/- 0.0024 at T=16).
#
# No double-launch risk: the stream checks results.json and a live-process guard before
# starting any cell, so it will skip f16 and move to f32.
#
# Params identical to the queued definition (bs 16 keeps frames-per-batch at 256).
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g5_f16

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
    && { echo "[$(date +%F_%T)] $name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --frames 16 \
      --epochs 12 --batch_size 16 --lr 3e-4 --workers 5 \
      --seed 42 --split_seed 49 \
      --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
  rc=$?
  if [ -f "output/seed_runs/$name/results.json" ]; then
    echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
  tail -3 "$L/$name.log"
  sleep 120
done
echo "[$(date +%F_%T)] GAVE UP on $name"
