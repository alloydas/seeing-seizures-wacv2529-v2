#!/bin/bash
# Pull vid_g3_f64 forward out of sh/_abl_frames_grading.sh and run it on GPU1 now.
# Second-to-last cell of the frames-grading ablation; only vid_g5_f64 remains after it.
#
# GPU1 chosen: it is carrying two jobs to GPU0's three.
#
# No double-launch risk: the stream checks results.json and a live-process guard.
# Params identical to the queued definition (bs 4 keeps frames-per-batch at 256).
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g3_f64

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
    && { echo "[$(date +%F_%T)] $name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --group3 --frames 64 \
      --epochs 12 --batch_size 4 --lr 3e-4 --workers 5 \
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
