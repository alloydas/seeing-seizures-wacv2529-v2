#!/bin/bash
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
# 3-class (--group3), seeds 1,2,3 in parallel (GRU tiny)
for s in 1 2 3; do
  python3 train_pooled_eeg.py --arch gru --group3 --seed $s --split_seed 49 \
    --output output/seed_runs/eeg_g3_s$s > $L/eeg_g3_s$s.log 2>&1 &
done
wait
echo "[$(date +%F_%T)] eeg 3-class seeds done"
# 5-class (no group flag), seeds 1,2,3 in parallel
for s in 1 2 3; do
  python3 train_pooled_eeg.py --arch gru --seed $s --split_seed 49 \
    --output output/seed_runs/eeg_g5_s$s > $L/eeg_g5_s$s.log 2>&1 &
done
wait
echo "[$(date +%F_%T)] eeg 5-class seeds done"
