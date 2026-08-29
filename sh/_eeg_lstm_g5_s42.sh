#!/bin/bash
# EEG LSTM, 5-class, seed 42 -- last GPU-trainable probe of tab:eegarch reproducibility.
# Original command (sh/_eeg_lstm.sh:8): --arch lstm, no group flag, defaults.
#
# After this only RF and XGBoost remain unverified, and those are sklearn/xgboost --
# CPU-only, so they contend with video decode rather than GPU memory and are better
# run once the video queue drains.
#
# Chosen as an EEG run on purpose: sh/_abl_frames_grading.sh gates its GPU1 stream on
# `pgrep -f "python3 train_pooled.py"`, which does NOT match train_pooled_eeg.py, so
# this adds no delay to the frames ablation -- a video run here would.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_lstm_g5_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " && { echo "$name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch lstm --seed 42 --split_seed 49 \
      --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
  rc=$?
  if [ -f "output/seed_runs/$name/results.json" ]; then
    echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
  tail -3 "$L/$name.log"
  sleep 180
done
echo "[$(date +%F_%T)] GAVE UP on $name"
