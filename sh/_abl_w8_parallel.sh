#!/bin/bash
# Runs ONE stage-A ablation cell (w8) in parallel with the serial sh/_abl_eeg_window.sh,
# to shorten the critical path to the backbone sweep.
#
# WHY w8 SPECIFICALLY. The serial driver walks stage A in the order w2 -> w4 -> w6 -> w8,
# and its cell() returns early when $out/results.json already exists. w8 is the LAST cell
# it reaches, so this run has the largest possible head start (~2 h at the observed
# ~3 min/epoch on w2) before the driver could ever duplicate it. Picking w4 or w6 instead
# would race.
#
# Same hyper-parameters as cell() in the driver -- arch gru, 5-class (no task flag),
# seed 42, split_seed 49 -- so the number is comparable with the other cells.
#
# Cache is deleted afterwards, mirroring KEEP_CACHE=0. The driver would NOT delete it,
# because it skips the cell at the results.json check, before the rm.
set -u
cd /path/to/repo
L=logs/abl_eeg_window
OUT=output/v3_abl_eeg_window/w8
CACHE=cache_abl/seg_w8_s4_d8.npz
mkdir -p "$L" "$OUT"
log(){ echo "[$(date +%F_%T)] $*"; }

if [ -f "$OUT/results.json" ]; then log "w8 already done -- nothing to do"; exit 0; fi

# --- build the w8 cache (CPU, 8 workers; video queue is empty so cores are free) ------
if [ ! -f "$CACHE" ]; then
  log "building w8 cache win=8s stride=4s decim=8"
  python3 build_stage_segments_pooled.py --win 8 --stride 4 --decim 8 \
      --workers 8 --out "$CACHE" > "$L/w8_build.log" 2>&1
  if [ ! -f "$CACHE" ]; then
    log "CACHE BUILD FAILED -- tail:"; tail -5 "$L/w8_build.log"; exit 1
  fi
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

# --- train on GPU 1 --------------------------------------------------------------
for try in 1 2 3; do
  [ -f "$OUT/results.json" ] && break
  log "START w8 train on GPU1 (try $try)"
  CUDA_VISIBLE_DEVICES=1 python3 train_pooled_eeg.py --arch gru \
      --cache "$CACHE" --seed 42 --split_seed 49 --output "$OUT" \
      > "$L/w8.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "w8 DONE"; break; }
  log "w8 FAILED try $try -- tail:"; tail -3 "$L/w8.log"; sleep 120
done

rm -f "$CACHE"; log "removed w8 cache"
[ -f "$OUT/results.json" ] || { log "w8 GAVE UP"; exit 1; }
python3 -c "
import json; r=json.load(open('$OUT/results.json'))
print('w8  bal_acc=%.4f  macro_f1=%.4f  acc=%.4f' % (r['balanced_accuracy'], r['macro_f1'], r['accuracy']))
"
log "===== w8 parallel cell finished ====="
