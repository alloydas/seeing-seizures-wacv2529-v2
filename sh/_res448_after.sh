#!/bin/bash
cd /path/to/repo
# wait for res224 to finish, then run res448 into a now-empty GPU1 (avoids NVML pressure)
while [ ! -f output/abl_res/r2p1d_224/results.json ]; do sleep 60; done
sleep 20
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
CUDA_VISIBLE_DEVICES=1 python3 train_pooled.py --arch r2plus1d --frames 16 --size 448 \
  --group2 --epochs 10 --batch_size 4 --workers 8 --output output/abl_res/r2p1d_448 > logs/abl_res/res448.log 2>&1
echo "res448 done rc=$?"
