#!/bin/bash
# EEG backbone table (tab:eegarch) trained at the BEST preprocessing configuration,
# selected empirically by the plan-#8 ablation rather than assumed.
#
# Design (per the user, 2026-08-12): run the EEG ablation first, take the winning
# configuration, THEN benchmark the backbones on it. The previous version of this script
# hardcoded the 6 s / 3 s / 125 Hz default -- which is precisely the assumption the
# ablation exists to test, so a backbone comparison at that setting would inherit an
# arbitrary choice.
#
# STAGES
#  1. wait for sh/_abl_eeg_window.sh (running on the REPAIRED clips) to finish
#  2. read every ablation cell, pick the highest clip macro-F1, and recover its
#     (window, stride, decimation) from that cell's build log
#  3. read agg_sweep.json and pick the best clip-pooling rule (mean/max/topk/logmean/attn)
#  4. rebuild the winning cache -- the ablation deletes caches after use (KEEP_CACHE=0),
#     so the winner has to be regenerated once
#  5. train 7 backbones x 3 tasks x 5 seeds = 105 runs at that configuration
#
# The GRU is INCLUDED this time: its existing 5 seeds are at the 6 s/3 s default, which
# is a different configuration, so they are not reusable here.
#
# Concurrency 4 (2 per GPU) counting all EEG trainers, so it shares the machine with
# anything still running. Six concurrent produced NVML aborts on 2026-08-11.
set -u
cd /path/to/repo
L=logs/v3_bestcfg; mkdir -p "$L" output/v3_bestcfg
ABL=output/v3_abl_eeg_window
SEEDS="1 2 3 5 42"
log(){ echo "[$(date +%F_%T)] $*"; }

# ---------- 1. wait for the ablation ----------------------------------------------
log "waiting for the EEG ablation to finish"
while pgrep -f "_abl_eeg_window\.sh" > /dev/null; do sleep 120; done
log "ablation finished"

# ---------- 2+3. select the winning configuration ---------------------------------
read -r WIN STRIDE DECIM AGG TAG <<< "$(python3 - <<'PY'
import json, glob, os, re
best, btag, bscore = None, None, -1
for d in sorted(glob.glob('output/v3_abl_eeg_window/*')):
    p = os.path.join(d, 'results.json')
    if not os.path.isfile(p):
        continue
    f1 = json.load(open(p))['macro_f1']
    if f1 > bscore:
        bscore, btag = f1, os.path.basename(d)
# recover (win, stride, decim) from that cell's build log; the builder prints
#   "win=4.0s stride=2.0s decim=8 (125 Hz, 500 samples/window) -> cache_abl/..."
win = stride = decim = None
bl = f'logs/abl_eeg_window/{btag}_build.log'
if os.path.exists(bl):
    m = re.search(r'win=([\d.]+)s stride=([\d.]+)s decim=(\d+)', open(bl).read())
    if m:
        win, stride, decim = m.group(1), m.group(2), m.group(3)
# best clip-pooling rule, averaged over ablation cells
agg = 'mean'
aj = 'output/v3_abl_eeg_window/agg_sweep.json'
if os.path.exists(aj):
    data = json.load(open(aj))
    means = {}
    for run, rules in data.items():
        for rule, m in rules.items():
            means.setdefault(rule, []).append(m['balanced_accuracy'])
    if means:
        agg = max(means, key=lambda k: sum(means[k]) / len(means[k]))
print(win or 6, stride or 3, decim or 8, agg, btag or 'none')
PY
)"
log "winner: cell=$TAG  win=${WIN}s stride=${STRIDE}s decim=$DECIM  pooling=$AGG"
if [ "$TAG" = "none" ]; then log "ABORT: no ablation results found"; exit 1; fi

# ---------- 4. rebuild the winning cache ------------------------------------------
CACHE="cache_bestcfg/seg_w${WIN}_s${STRIDE}_d${DECIM}.npz"
mkdir -p cache_bestcfg
if [ ! -f "$CACHE" ]; then
  log "rebuilding winning cache -> $CACHE"
  python3 build_stage_segments_pooled.py --win "$WIN" --stride "$STRIDE" \
      --decim "$DECIM" --workers 8 --out "$CACHE" > "$L/build_bestcfg.log" 2>&1
  [ -f "$CACHE" ] || { log "ABORT: cache build failed"; tail -5 "$L/build_bestcfg.log"; exit 1; }
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

# ---------- 5. backbone sweep at the winning configuration ------------------------
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

log "=== classical RF/XGB (CPU) at the winning configuration ==="
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

log "=== deep backbones, 4 concurrent ==="
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

log "===== Table 2 at the winning configuration ====="
python3 - <<'PY'
import json, glob, os, statistics as st
print(f"{'backbone':11s} {'detection':>18s} {'3-class':>18s} {'5-class':>18s}")
best = {}
for arch in ('gru','lstm','eegnet','conformer','tcn','rf','xgb'):
    row = f'{arch:11s}'
    for tag in ('bin','g3','g5'):
        v = [json.load(open(p))['macro_f1']
             for p in sorted(glob.glob(f'output/v3_bestcfg/{arch}_{tag}_s*/results.json'))]
        if v:
            m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            row += f"  {m:.4f}±{s:.4f} n={len(v)}"
            if m > best.get(tag, (0, ''))[0]:
                best[tag] = (m, arch)
        else:
            row += f"{'--':>18s}"
    print(row)
print("\nBEST EEG BACKBONE PER TASK (this is the row for Table 1):")
for tag, name in (('bin','detection'), ('g3','3-class'), ('g5','5-class')):
    if tag in best:
        print(f"  {name:10s} {best[tag][1]}  {best[tag][0]:.4f}")
PY
log "===== finished ====="
