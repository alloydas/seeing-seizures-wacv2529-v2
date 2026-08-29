#!/bin/bash
# Pull vid_f32_s42 forward out of the queue and run it on GPU1 now.
#
# It was queued in sh/_gpu1_tail.sh behind vid_f64_s42 + vid_g3_s42, i.e. it would not
# have started until ~Thu 01:45. Running it now is the pick that makes the most of the
# other jobs already in flight: with vid_f1_s42 (T=1) and vid_f4_s42 (T=4) running and
# the four detection seeds giving T=16 (0.9681 +/- 0.0024), T=32 completes a
# temporal-sampling row measured entirely under CURRENT code --
#     T = 1, 4, 16(x4 seeds), 32, 64
# which is what makes vid_f64_s42's eventual number interpretable at all. The stored
# 8/16/32 row is Jul-25, pre-refactor, and cannot be mixed with these.
#
# No double-launch risk: sh/_gpu1_tail.sh checks both results.json and a live-process
# guard before it would start this itself.
#
# Params identical to the queued definition and to the original sweep
# (sh/_video_more.sh:15): --frames 32 --batch_size 8 --lr 3e-4, seed 42, split_seed 49.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_f32_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
    && { echo "[$(date +%F_%T)] $name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --group2 --frames 32 \
      --epochs 12 --batch_size 8 --lr 3e-4 --workers 4 \
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
