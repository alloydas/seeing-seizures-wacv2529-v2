#!/bin/bash
# Pull vid_g3_f32 forward out of sh/_abl_frames_grading.sh and run it on GPU0 now.
#
# Next cell in that script's GPU0 stream (3-class: f8 done -> f16 running -> f32 -> f64).
# GPU0 is carrying one job to GPU1's two, so this is the balanced place to add work.
#
# No double-launch risk: the stream checks results.json and a live-process guard before
# starting any cell, so it will skip f32 and move to f64.
#
# Params identical to the queued definition (bs 8 keeps frames-per-batch at 256).
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g3_f32

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
    && { echo "[$(date +%F_%T)] $name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --group3 --frames 32 \
      --epochs 12 --batch_size 8 --lr 3e-4 --workers 5 \
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
