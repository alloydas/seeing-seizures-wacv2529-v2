#!/bin/bash
# Sequential GPU1 queue. Replaces sh/_abl_frames64.sh (stopped while still waiting;
# nothing had launched yet) to put the validation run ahead of the frames extension.
#
# Order and rationale:
#   1. vid_g3_s42  -- video 3-class at seed 42. The claim "the video 3-class baseline
#      (0.7765) is stale" currently rests only on it sitting far from the two new seeds
#      (0.7972 +/- 0.0027). That is the same reasoning that just proved WRONG for EEG
#      3-class: its baseline looked 4 sigma off at n=3, but adding seed 42 widened the
#      spread to +/-0.0194 and the baseline landed 0.1 sigma from the mean. So this
#      needs a same-seed reproduction before anything in the paper changes. It also
#      takes the video 3-class cell from n=2 to n=3.
#   2. vid_f64_s42 -- 64 frames, the temporal-sampling extension.
#   3. vid_f32_s42 -- 32 frames re-run under current code, so 64 is not compared
#      against the stale Jul-25 0.972.
#
# Waits for eeg_tcn_g5_s42 (ep25/30 at 11:54) to release GPU1 first.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs

busy(){ pgrep -f "seed_runs/eeg_tcn_g5_s42\b" > /dev/null; }

echo "[$(date +%F_%T)] waiting for eeg_tcn_g5_s42 to release GPU1"
while busy; do sleep 60; done
echo "[$(date +%F_%T)] GPU1 free"

run_retry(){          # $1=name  $2..=extra train args
  local name=$1; shift
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    echo "[$(date +%F_%T)] START $name (try $try)"
    python3 train_pooled.py --arch r2plus1d "$@" --epochs 12 \
        --workers 5 --seed 42 --split_seed 49 \
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

# 1. 64-frame temporal-sampling extension (requested first)
run_retry vid_f64_s42 --group2 --frames 64 --batch_size 4 --lr 3e-4
# 2. video 3-class seed-42 reproduction (matches the grading runs: default frames/lr)
run_retry vid_g3_s42 --group3 --batch_size 16
# 3. 32 frames re-run under current code, so 64 is not compared against the stale
#    Jul-25 0.972
run_retry vid_f32_s42 --group2 --frames 32 --batch_size 8 --lr 3e-4
echo "[$(date +%F_%T)] gpu1 queue done"
