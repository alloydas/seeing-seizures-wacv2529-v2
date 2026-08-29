#!/bin/bash
# Runs stage-C ablation cell r62 (62.5 Hz) in parallel with the serial
# sh/_abl_eeg_window.sh, on GPU 1. Third companion to _abl_w6/_abl_w8_parallel.sh.
#
# CONFIGURATION IS NOW KNOWN. Stage A finished at 11:27 with w6 (6 s window) winning at
# 0.5340 macro-F1, so the driver fixes BW=6 for stages B and C and its remaining cells
# are fully determined:
#     stage B  s1.5 (6 s / 1.5 s)   s6 (6 s / 6 s)
#     stage C  r250 (decim 4)       r62 (decim 16)
# It walks them in that order, and cell() returns early when results.json exists.
#
# RACE MARGIN. r62 is the LAST cell the driver reaches. It is currently training s1.5
# (started 11:29:59, ~2x the windows of w6, so ~65 min), then s6, then r250 -- it cannot
# reach r62 before ~13:50. This run needs ~2 min of cache build plus ~20 min of training
# (decim 16 halves the samples per window against w6), landing ~12:10. About 1.7 h slack.
#
# Same hyper-parameters as cell(): arch gru, 5-class, seed 42, split_seed 49.
# Cache name matches the driver's own (seg_w6_s3_d16.npz) so a duplicate would reuse it.
set -u
cd /path/to/repo
L=logs/abl_eeg_window
OUT=output/v3_abl_eeg_window/r62
CACHE=cache_abl/seg_w6_s3_d16.npz
GPU=1
mkdir -p "$L" "$OUT"
log(){ echo "[$(date +%F_%T)] $*"; }

if [ -f "$OUT/results.json" ]; then log "r62 already done -- nothing to do"; exit 0; fi

if [ ! -f "$CACHE" ]; then
  log "building r62 cache win=6s stride=3s decim=16 (62.5 Hz)"
  python3 build_stage_segments_pooled.py --win 6 --stride 3 --decim 16 \
      --workers 8 --out "$CACHE" > "$L/r62_build.log" 2>&1
  if [ ! -f "$CACHE" ]; then
    log "CACHE BUILD FAILED -- tail:"; tail -5 "$L/r62_build.log"; exit 1
  fi
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

for try in 1 2 3; do
  [ -f "$OUT/results.json" ] && break
  log "START r62 train on GPU$GPU (try $try)"
  CUDA_VISIBLE_DEVICES=$GPU python3 train_pooled_eeg.py --arch gru \
      --cache "$CACHE" --seed 42 --split_seed 49 --output "$OUT" \
      > "$L/r62.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "r62 DONE"; break; }
  log "r62 FAILED try $try -- tail:"; tail -3 "$L/r62.log"; sleep 120
done

rm -f "$CACHE"; log "removed r62 cache"
[ -f "$OUT/results.json" ] || { log "r62 GAVE UP"; exit 1; }
python3 -c "
import json; r=json.load(open('$OUT/results.json'))
print('r62  bal_acc=%.4f  macro_f1=%.4f  acc=%.4f' % (r['balanced_accuracy'], r['macro_f1'], r['accuracy']))
"
log "===== r62 parallel cell finished ====="
