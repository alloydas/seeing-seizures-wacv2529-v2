#!/bin/bash
# Frame-count ablation (R2+1D 8/32; 16 already = 0.970) + more backbones
# (S3D, X3D, MViT-v1, Swin-S, SlowFast) on 2-class detection. Same seed-49 split.
# Waits for the Swin-T/MViT (video_arch) driver to release the GPUs first.
cd /path/to/repo
L=logs/video_more; mkdir -p "$L"
echo "[$(date +%F_%T)] waiting for prior video runs (videomae + swin_t/mvit)..."
while pgrep -f 'train_pooled_videomae.py|train_pooled.py --arch (swin|mvit) ' >/dev/null 2>&1; do sleep 30; done
sleep 5
echo "[$(date +%F_%T)] GPUs free -- launching frame ablation + more backbones"
run(){ echo "[$(date +%F_%T)] START $1"; g=$2; shift 2; CUDA_VISIBLE_DEVICES=$g "$@"; echo "[$(date +%F_%T)] END $1 rc=$?"; }

# GPU0 queue (fast-ish): frame ablation + s3d + mvit_v1
( run r2p1d_f8  0 python3 train_pooled.py --arch r2plus1d --frames 8  --group2 --epochs 12 --batch_size 16 --lr 3e-4 --output vid_r2p1d_f8  > "$L/r2p1d_f8.log"  2>&1
  run r2p1d_f32 0 python3 train_pooled.py --arch r2plus1d --frames 32 --group2 --epochs 12 --batch_size 8  --lr 3e-4 --output vid_r2p1d_f32 > "$L/r2p1d_f32.log" 2>&1
  run s3d       0 python3 train_pooled.py --arch s3d      --group2 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_s3d_bin      > "$L/s3d_bin.log"     2>&1
  run mvit_v1   0 python3 train_pooled.py --arch mvit_v1  --group2 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_mvit_v1_bin  > "$L/mvit_v1_bin.log" 2>&1
  echo "[$(date +%F_%T)] GPU0 more-video queue done" ) &

# GPU1 queue (heavy): x3d + slowfast + swin_s
( run x3d       1 python3 train_pooled.py --arch x3d      --group2 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_x3d_bin      > "$L/x3d_bin.log"     2>&1
  run slowfast  1 python3 train_pooled.py --arch slowfast --group2 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_slowfast_bin > "$L/slowfast_bin.log" 2>&1
  run swin_s    1 python3 train_pooled.py --arch swin_s   --group2 --epochs 12 --batch_size 4 --lr 1e-4 --output vid_swin_s_bin   > "$L/swin_s_bin.log"  2>&1
  echo "[$(date +%F_%T)] GPU1 more-video queue done" ) &
wait
echo "[$(date +%F_%T)] video-more driver done"
