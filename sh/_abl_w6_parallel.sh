#!/bin/bash
# Runs stage-A ablation cell w6 in parallel with the serial sh/_abl_eeg_window.sh,
# on GPU 0. Companion to sh/_abl_w8_parallel.sh.
#
# RACE MARGIN. The serial driver walks w2 -> w4 -> w6 -> w8 and cell() returns early
# when $out/results.json exists. It is on w2 (ep 11/30, ~3.3 min/ep -> ~10:39), then w4
# (~50 min -> ~11:32), so it cannot reach w6 before ~11:32. This run needs ~2 min of
# cache build plus ~33 min of training (w6/s3 yields ~1/3 the windows of w2/s1), landing
# ~10:15 -- about 1.3 h of slack.
#
# Same hyper-parameters as cell() in the driver -- arch gru, 5-class (no task flag),
# seed 42, split_seed 49 -- so the number is comparable across cells.
#
# Cache is deleted afterwards, mirroring KEEP_CACHE=0; the driver would not delete it,
# because it skips the cell at the results.json check, before the rm.
set -u
cd /path/to/repo
L=logs/abl_eeg_window
OUT=output/v3_abl_eeg_window/w6
CACHE=cache_abl/seg_w6_s3_d8.npz
GPU=0
mkdir -p "$L" "$OUT"
log(){ echo "[$(date +%F_%T)] $*"; }

if [ -f "$OUT/results.json" ]; then log "w6 already done -- nothing to do"; exit 0; fi

if [ ! -f "$CACHE" ]; then
  log "building w6 cache win=6s stride=3s decim=8"
  python3 build_stage_segments_pooled.py --win 6 --stride 3 --decim 8 \
      --workers 8 --out "$CACHE" > "$L/w6_build.log" 2>&1
  if [ ! -f "$CACHE" ]; then
    log "CACHE BUILD FAILED -- tail:"; tail -5 "$L/w6_build.log"; exit 1
  fi
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

for try in 1 2 3; do
  [ -f "$OUT/results.json" ] && break
  log "START w6 train on GPU$GPU (try $try)"
  CUDA_VISIBLE_DEVICES=$GPU python3 train_pooled_eeg.py --arch gru \
      --cache "$CACHE" --seed 42 --split_seed 49 --output "$OUT" \
      > "$L/w6.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "w6 DONE"; break; }
  log "w6 FAILED try $try -- tail:"; tail -3 "$L/w6.log"; sleep 120
done

rm -f "$CACHE"; log "removed w6 cache"
[ -f "$OUT/results.json" ] || { log "w6 GAVE UP"; exit 1; }
python3 -c "
import json; r=json.load(open('$OUT/results.json'))
print('w6  bal_acc=%.4f  macro_f1=%.4f  acc=%.4f' % (r['balanced_accuracy'], r['macro_f1'], r['accuracy']))
"
log "===== w6 parallel cell finished ====="
