#!/bin/bash
# Extend the temporal-sampling ablation (Table 5, tab:frames) past 32 frames.
#
# Runs TWO things, sequentially on GPU1:
#   1. frames=64  -- the new point being asked for
#   2. frames=32  -- re-run under CURRENT code, because the stored 0.972 was written
#                    2026-07-25, before the 2026-08-03 train_pooled.py edit. Without
#                    this, a fresh 64 number would be compared against a stale 32.
#
# The 16-frame anchor already exists under current code: the four detection seed runs
# are frames=16 by default and give macro-F1 0.9681 +/- 0.0024.
#
# Matching the original sweep's hyperparameters (sh/_video_more.sh:14-15):
#   f8  -> --batch_size 16 --lr 3e-4
#   f32 -> --batch_size 8  --lr 3e-4
# so f64 uses --batch_size 4 to keep frames-per-batch constant (64x4 = 32x8).
#
# Runtime: the Jul-25 f32 run took ~6 h (00:15 -> 06:14) on a quieter machine. f64
# processes 2x the frames per epoch, so expect ~12 h, plus contention from GPU0's
# three video trainers.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
EEG="eeg_bin_s42 eeg_g3_s42 eeg_tcn_g5_s42"

eeg_busy(){ for e in $EEG; do pgrep -f "seed_runs/$e\b" > /dev/null && return 0; done; return 1; }

echo "[$(date +%F_%T)] waiting for GPU1 EEG jobs to clear"
while eeg_busy; do sleep 120; done
echo "[$(date +%F_%T)] GPU1 clear"

run_retry(){          # $1=name  $2=frames  $3=batch
  local name=$1 fr=$2 bs=$3
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    echo "[$(date +%F_%T)] START $name (frames=$fr bs=$bs, try $try)"
    python3 train_pooled.py --arch r2plus1d --frames "$fr" --group2 --epochs 12 \
        --batch_size "$bs" --lr 3e-4 --workers 5 --seed 42 --split_seed 49 \
        --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
    rc=$?
    if [ -f "output/seed_runs/$name/results.json" ]; then
      echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; return 0
    fi
    echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
    tail -3 "$L/$name.log"
    sleep 120
  done
  echo "[$(date +%F_%T)] GAVE UP on $name"; return 1
}

run_retry vid_f64_s42 64 4
run_retry vid_f32_s42 32 8
echo "[$(date +%F_%T)] frames ablation extension done"
