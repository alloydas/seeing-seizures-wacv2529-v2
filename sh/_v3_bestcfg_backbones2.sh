#!/bin/bash
# Table 2 (tab:eegarch) at the best EEG configuration -- corrected selection.
#
# REPLACES sh/_v3_bestcfg_backbones.sh, which selected wrongly. That script picked the
# cell by the macro-F1 in results.json (which is computed under MEAN clip pooling) and
# picked the pooling rule separately, by balanced accuracy averaged over cells. The two
# choices interact, so the pair it would have selected -- s1.5 + logmean, 0.5532 -- is
# not the best pair available:
#
#     r250 + logmean  0.5706   <- best (was still training when this was written)
#     w6   + logmean  0.5622   <- best completed
#     w2   + logmean  0.5551
#     s1.5 + mean     0.5540   <- why s1.5 looked like the stage-A/B winner
#     s1.5 + logmean  0.5532   <- what the old script would have run
#     w6   + mean     0.5340   <- today's default
#
# Pooling matters more than window length: logmean beats mean on 7 of 8 cells, and is
# worth +0.028 macro-F1 at the default configuration -- for free, since it re-scores
# saved window predictions without retraining. It also reorders the cells, which is why
# the two axes have to be chosen jointly rather than in sequence.
#
# STAGES
#   1. wait for r250 (last ablation cell) to finish
#   2. re-score every cell under all five pooling rules and take the best (cell, rule)
#      pair by macro-F1
#   3. recover that cell's (win, stride, decim) from its build log and rebuild the cache
#   4. 7 backbones x 3 tasks x 5 seeds = 105 runs at that configuration
set -u
cd /path/to/repo
L=logs/v3_bestcfg; mkdir -p "$L" output/v3_bestcfg
ABL=output/v3_abl_eeg_window
SEEDS="1 2 3 5 42"
log(){ echo "[$(date +%F_%T)] $*"; }

# ---------- 1. wait for the last cell ------------------------------------------
log "waiting for the r250 ablation cell"
while [ ! -f "$ABL/r250/results.json" ]; do
  pgrep -f "train_pooled_eeg.py.*abl_eeg_window/r250" > /dev/null || \
    pgrep -f "_abl_eeg_window\.sh" > /dev/null || { log "r250 gone and no results -- aborting"; exit 1; }
  sleep 60
done
log "r250 done"

# ---------- 2. joint (cell, pooling) selection ---------------------------------
python3 sweep_eeg_agg.py --glob "$ABL/*" --json_out "$ABL/agg_sweep.json" \
    > "$L/agg_sweep_final.log" 2>&1
read -r TAG AGG SCORE <<< "$(python3 - <<'PY'
import json, os
d = json.load(open('output/v3_abl_eeg_window/agg_sweep.json'))
best = (-1, None, None)
for k, v in d.items():
    cell = os.path.basename(k.rstrip('/'))
    for rule, m in v.items():
        f1 = m.get('macro_f1')
        if f1 is not None and f1 > best[0]:
            best = (f1, cell, rule)
print(best[1], best[2], f"{best[0]:.4f}")
PY
)"
log "winning pair: cell=$TAG  pooling=$AGG  macro-F1=$SCORE"

read -r WIN STRIDE DECIM <<< "$(python3 - <<PY
import re, os
p = 'logs/abl_eeg_window/${TAG}_build.log'
m = re.search(r'win=([\d.]+)s stride=([\d.]+)s decim=(\d+)', open(p).read()) if os.path.exists(p) else None
print(*(m.groups() if m else ('6','3','8')))
PY
)"
log "configuration: win=${WIN}s stride=${STRIDE}s decim=$DECIM pooling=$AGG"

# ---------- 3. rebuild the winning cache ---------------------------------------
CACHE="cache_bestcfg/seg_w${WIN}_s${STRIDE}_d${DECIM}.npz"
mkdir -p cache_bestcfg
if [ ! -f "$CACHE" ]; then
  log "building $CACHE"
  python3 build_stage_segments_pooled.py --win "$WIN" --stride "$STRIDE" \
      --decim "$DECIM" --workers 8 --out "$CACHE" > "$L/build_bestcfg.log" 2>&1
  [ -f "$CACHE" ] || { log "ABORT: cache build failed"; tail -5 "$L/build_bestcfg.log"; exit 1; }
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

# ---------- 4. the sweep --------------------------------------------------------
busy(){ pgrep -af "python3 train_pooled_eeg" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }

deep(){
  local gpu=$1 arch=$2 flag=$3 tag=$4 seed=$5
  local out="output/v3_bestcfg/${arch}_${tag}_s${seed}"
  [ -f "$out/results.json" ] && return 0
  busy "$out" && return 0
  local try
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch "$arch" $flag \
        --cache "$CACHE" --agg "$AGG" --seed "$seed" --split_seed 49 --output "$out" \
        > "$L/${arch}_${tag}_s${seed}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE ${arch}_${tag}_s${seed}"; return 0; }
    log "retry $try failed ${arch}_${tag}_s${seed}"; sleep 120
  done
  log "GAVE UP ${arch}_${tag}_s${seed}"
}

log "=== classical RF/XGB (CPU) ==="
for arch in rf xgb; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      out="output/v3_bestcfg/${arch}_${tag}_s${seed}"
      [ -f "$out/results.json" ] && continue
      python3 train_pooled_eeg_classical.py --arch "$arch" $flag --cache "$CACHE" \
          --seed "$seed" --split_seed 49 --output "$out" \
          > "$L/${arch}_${tag}_s${seed}.log" 2>&1
    done
  done
done
log "classical done"

log "=== deep backbones, 4 concurrent (2/GPU) ==="
i=0
for arch in gru lstm eegnet conformer tcn; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      while [ "$(pgrep -cf 'python3 train_pooled_eeg\.py')" -ge 4 ]; do sleep 30; done
      gpu=$(( i % 2 )); i=$(( i + 1 ))
      deep "$gpu" "$arch" "$flag" "$tag" "$seed" &
      sleep 10
    done
  done
done
wait

log "===== sweep complete; emitting tab:eegarch ====="
python3 make_tab_eegarch.py 2>&1
log "===== finished ====="
