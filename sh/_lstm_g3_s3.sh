#!/bin/bash
set -u
cd /path/to/repo
OUT=output/v3_bestcfg/lstm_g3_s3
for try in 1 2 3; do
  [ -f "$OUT/results.json" ] && break
  echo "[$(date +%F_%T)] START lstm_g3_s3 (try $try)"
  CUDA_VISIBLE_DEVICES=1 python3 train_pooled_eeg.py --arch lstm --group3 \
      --cache cache_bestcfg/seg_w6.0_s3.0_d8.npz --agg logmean \
      --seed 3 --split_seed 49 --output "$OUT" \
      > logs/v3_bestcfg/lstm_g3_s3.log 2>&1
  [ -f "$OUT/results.json" ] && { echo "[$(date +%F_%T)] DONE"; break; }
  echo "[$(date +%F_%T)] failed try $try"; tail -3 logs/v3_bestcfg/lstm_g3_s3.log | cut -c1-140
  sleep 60
done
python3 make_tab_eegarch.py 2>&1 | tail -30
