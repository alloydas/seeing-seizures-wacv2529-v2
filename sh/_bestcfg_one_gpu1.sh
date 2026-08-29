#!/bin/bash
# Runs one pending cell of the 105-run Table 2 sweep explicitly on GPU 1.
#
# WHY BY HAND. sh/_v3_bestcfg_backbones2.sh assigns GPUs round-robin (i % 2) without
# looking at actual occupancy, and it had drifted to 3 jobs on GPU 0 (0.2 GiB free --
# the condition that aborted eeg_bin_fold4 with the NVML assert) against 1 on GPU 1
# (13.1 GiB free). Pinning this cell to GPU 1 rebalances without raising total
# concurrency: the sweep counts ALL train_pooled_eeg processes against its cap of 4, so
# it simply holds its next launch until a slot frees, and its busy()/results.json guards
# make it skip this cell rather than duplicate it.
#
# Same configuration as the sweep: the ablation's winning pair, 6 s / 3 s / 125 Hz with
# logmean clip pooling.
set -u
cd /path/to/repo
CACHE=cache_bestcfg/seg_w6.0_s3.0_d8.npz
OUT=output/v3_bestcfg/gru_g5_s1
L=logs/v3_bestcfg
log(){ echo "[$(date +%F_%T)] $*"; }

[ -f "$OUT/results.json" ] && { log "already done"; exit 0; }
for try in 1 2 3; do
  log "START gru_g5_s1 on GPU1 (try $try)"
  CUDA_VISIBLE_DEVICES=1 python3 train_pooled_eeg.py --arch gru \
      --cache "$CACHE" --agg logmean --seed 1 --split_seed 49 --output "$OUT" \
      > "$L/gru_g5_s1.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "DONE gru_g5_s1"; break; }
  log "retry $try failed -- tail:"; tail -3 "$L/gru_g5_s1.log" | cut -c1-140; sleep 120
done
[ -f "$OUT/results.json" ] && python3 -c "
import json;r=json.load(open('$OUT/results.json'))
print('gru_g5_s1  bal=%.4f  macro_f1=%.4f'%(r['balanced_accuracy'],r['macro_f1']))"
