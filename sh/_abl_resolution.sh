#!/bin/bash
# Resolution ablation: R(2+1)D detection at 224 and 448 (112 = existing 0.970 baseline).
# Waits for the two Swin RodEpil runs to free the GPUs (checks results.json, not pgrep).
cd /path/to/repo
L=logs/abl_res
echo "[$(date +%F_%T)] waiting for Swin RodEpil to finish..."
while [ ! -f output/rodepil_swin/results.json ] || [ ! -f output/rodepil_swin_s/results.json ]; do sleep 120; done
sleep 30
echo "[$(date +%F_%T)] GPUs free -- running resolution ablation"
run(){ echo "[$(date +%F_%T)] START $1"; shift; CUDA_VISIBLE_DEVICES=0 "$@"; echo "[$(date +%F_%T)] END rc=$?"; }
run res224 python3 train_pooled.py --arch r2plus1d --frames 16 --size 224 --group2 --epochs 10 \
    --batch_size 8 --workers 10 --output output/abl_res/r2p1d_224 > $L/res224.log 2>&1
run res448 python3 train_pooled.py --arch r2plus1d --frames 16 --size 448 --group2 --epochs 10 \
    --batch_size 4 --workers 10 --output output/abl_res/r2p1d_448 > $L/res448.log 2>&1
echo "[$(date +%F_%T)] resolution ablation done"
