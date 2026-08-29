#!/bin/bash
# Ablation 1: subject-wise 5-fold CV, 2-class detection.
# Video R(2+1)D folds split across both GPUs; EEG GRU folds after, on GPU1.
cd /path/to/repo
L=logs/abl_subj
run(){ local name=$1 g=$2; shift 2; echo "[$(date +%F_%T)] START $name (gpu $g)"
  CUDA_VISIBLE_DEVICES=$g "$@"; echo "[$(date +%F_%T)] END $name rc=$?"; }

# GPU0: video folds 0,1,2
( for k in 0 1 2; do
    run "vid_f$k" 0 python3 train_pooled.py --split subject --fold $k --n_folds 5 \
        --group2 --epochs 10 --batch_size 16 --workers 12 --output abl_subj/vid_fold$k > $L/vid_fold$k.log 2>&1
  done
  echo "[$(date +%F_%T)] GPU0 video folds done" ) &

# GPU1: video folds 3,4 then EEG folds 0-4
( for k in 3 4; do
    run "vid_f$k" 1 python3 train_pooled.py --split subject --fold $k --n_folds 5 \
        --group2 --epochs 10 --batch_size 16 --workers 12 --output abl_subj/vid_fold$k > $L/vid_fold$k.log 2>&1
  done
  for k in 0 1 2 3 4; do
    run "eeg_f$k" 1 python3 train_pooled_eeg.py --split subject --fold $k --n_folds 5 \
        --group2 --epochs 25 --output abl_subj/eeg_fold$k > $L/eeg_fold$k.log 2>&1
  done
  echo "[$(date +%F_%T)] GPU1 video+eeg folds done" ) &
wait
echo "[$(date +%F_%T)] ablation-1 (subject-wise) driver done"
