#!/bin/bash
cd /path/to/repo
L=logs/seed_runs
run(){ echo "[$(date +%F_%T)] START $1"; shift; "$@"; echo "[$(date +%F_%T)] END rc=$?"; }
( export CUDA_VISIBLE_DEVICES=0
  for s in 1 2; do
    run vid_s$s python3 train_pooled.py --arch r2plus1d --group2 --epochs 12 --batch_size 16 \
        --workers 10 --seed $s --split_seed 49 --output output/seed_runs/vid_s$s > $L/vid_s$s.log 2>&1
  done; echo "GPU0 seed runs done" ) &
( export CUDA_VISIBLE_DEVICES=1
  for s in 1 2; do
    run eeg_s$s python3 train_pooled_eeg.py --arch gru --group2 --seed $s --split_seed 49 \
        --output output/seed_runs/eeg_s$s > $L/eeg_s$s.log 2>&1
  done; echo "GPU1 seed runs done" ) &
wait
echo "[$(date +%F_%T)] seed runs done"
