#!/bin/bash
# Wait for the VideoMAE runs to release the GPUs, then run Video Swin-T (GPU0)
# and MViTv2-S (GPU1) on 2-class detection -- same seed-49 session-disjoint
# split as R(2+1)D/TimeSformer/VideoMAE. Matched 12-epoch budget.
cd /path/to/repo
L=logs/video_arch; mkdir -p "$L"
echo "[$(date +%F_%T)] waiting for VideoMAE to finish..."
while pgrep -f 'train_pooled_videomae.py' >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%F_%T)] GPUs free -- launching Swin + MViT"
run(){ echo "[$(date +%F_%T)] START $1"; shift; "$@"; echo "[$(date +%F_%T)] END rc=$?"; }

CUDA_VISIBLE_DEVICES=0 python3 train_pooled.py --arch swin --group2 --epochs 12 \
    --batch_size 4 --lr 1e-4 --output vid_swin_bin > "$L/swin_bin.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_pooled.py --arch mvit --group2 --epochs 12 \
    --batch_size 8 --lr 1e-4 --output vid_mvit_bin > "$L/mvit_bin.log" 2>&1 &
P1=$!
wait $P0; echo "[$(date +%F_%T)] swin END rc=$?"
wait $P1; echo "[$(date +%F_%T)] mvit END rc=$?"
echo "[$(date +%F_%T)] video-arch driver done"
