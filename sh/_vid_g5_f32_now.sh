#!/bin/bash
# Pull vid_g5_f32 forward out of sh/_abl_frames_grading.sh and run it now.
#
# Chosen over the two f64 cells because it completes the 8/16/32 row for 5-class,
# matching what 3-class will have once vid_g3_f32 lands. That gives the full shape of
# both grading curves; the f64 cells (~36 h each) then only test a plateau that
# detection has already measured as flat-to-negative (T=32 0.9716 -> T=64 0.9707).
#
# Placed on GPU0 rather than GPU1: vid_g3_f16 is at 10/12 and about to free a slot
# there, keeping the machine at two jobs per GPU. GPU assignment does not affect the
# result -- same seed, same split -- so this does not break comparability with the
# rest of the sweep.
#
# No double-launch risk: the stream checks results.json and a live-process guard.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g5_f32

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
    && { echo "[$(date +%F_%T)] $name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --frames 32 \
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
