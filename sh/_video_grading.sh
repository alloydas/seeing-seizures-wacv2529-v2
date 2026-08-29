#!/bin/bash
# Run the 8 video backbones that only had detection results on 3-class + 5-class
# grading (R(2+1)D and VideoMAE already done). Same seed-49 session-disjoint split.
cd /path/to/repo
L=logs/video_grading; mkdir -p "$L"
run(){ local name=$1 g=$2; shift 2
  echo "[$(date +%F_%T)] START $name (gpu $g)"; CUDA_VISIBLE_DEVICES=$g "$@"
  echo "[$(date +%F_%T)] END $name rc=$?"; }

# ---- GPU 0 ----
( run swin_g3    0 python3 train_pooled.py --arch swin    --group3 --epochs 12 --batch_size 4 --lr 1e-4 --output vid_swin_g3    > $L/swin_g3.log    2>&1
  run swin_g5    0 python3 train_pooled.py --arch swin             --epochs 12 --batch_size 4 --lr 1e-4 --output vid_swin_g5    > $L/swin_g5.log    2>&1
  run swin_s_g3  0 python3 train_pooled.py --arch swin_s  --group3 --epochs 12 --batch_size 4 --lr 1e-4 --output vid_swin_s_g3  > $L/swin_s_g3.log  2>&1
  run swin_s_g5  0 python3 train_pooled.py --arch swin_s           --epochs 12 --batch_size 4 --lr 1e-4 --output vid_swin_s_g5  > $L/swin_s_g5.log  2>&1
  run s3d_g3     0 python3 train_pooled.py --arch s3d     --group3 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_s3d_g3     > $L/s3d_g3.log     2>&1
  run s3d_g5     0 python3 train_pooled.py --arch s3d              --epochs 12 --batch_size 8 --lr 1e-4 --output vid_s3d_g5     > $L/s3d_g5.log     2>&1
  run mvit_g3    0 python3 train_pooled.py --arch mvit    --group3 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_mvit_g3    > $L/mvit_g3.log    2>&1
  run mvit_g5    0 python3 train_pooled.py --arch mvit             --epochs 12 --batch_size 8 --lr 1e-4 --output vid_mvit_g5    > $L/mvit_g5.log    2>&1
  echo "[$(date +%F_%T)] GPU0 grading queue done" ) &

# ---- GPU 1 ----
( run slowfast_g3 1 python3 train_pooled.py --arch slowfast --group3 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_slowfast_g3 > $L/slowfast_g3.log 2>&1
  run slowfast_g5 1 python3 train_pooled.py --arch slowfast          --epochs 12 --batch_size 8 --lr 1e-4 --output vid_slowfast_g5 > $L/slowfast_g5.log 2>&1
  run x3d_g3      1 python3 train_pooled.py --arch x3d      --group3 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_x3d_g3      > $L/x3d_g3.log      2>&1
  run x3d_g5      1 python3 train_pooled.py --arch x3d               --epochs 12 --batch_size 8 --lr 1e-4 --output vid_x3d_g5      > $L/x3d_g5.log      2>&1
  run mvit_v1_g3  1 python3 train_pooled.py --arch mvit_v1  --group3 --epochs 12 --batch_size 8 --lr 1e-4 --output vid_mvit_v1_g3  > $L/mvit_v1_g3.log  2>&1
  run mvit_v1_g5  1 python3 train_pooled.py --arch mvit_v1           --epochs 12 --batch_size 8 --lr 1e-4 --output vid_mvit_v1_g5  > $L/mvit_v1_g5.log  2>&1
  run tsf_g3      1 python3 train_pooled_timesformer.py --group3 --epochs 12 --batch_size 8 --lr 5e-5 --output vid_tsf_g3 > $L/tsf_g3.log 2>&1
  run tsf_g5      1 python3 train_pooled_timesformer.py          --epochs 12 --batch_size 8 --lr 5e-5 --output vid_tsf_g5 > $L/tsf_g5.log 2>&1
  echo "[$(date +%F_%T)] GPU1 grading queue done" ) &
wait
echo "[$(date +%F_%T)] video-grading driver done"
