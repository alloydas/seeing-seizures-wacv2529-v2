#!/bin/bash
# Retrain the EEG backbone table (tab:eegarch, Table 2) on the REPAIRED clips.
#
# Runs after sh/_after_recut_retrain.sh, which repairs the clips, rebuilds the window
# cache as stage_segments_pooled_v3.npz, and retrains the six Table-1 GRU cells.
#
# NOTE ON SCOPE: the request said "table 4 backbones". By the compiled numbering Table 4
# is tab:frames -- the temporal-sampling ablation, which is VIDEO-only and untouched by
# the EDF defect (its 18 cells never read an EDF). The backbone table the repair does
# invalidate is Table 2, tab:eegarch. That is what this retrains. If Table 4 really was
# meant, nothing needs rerunning there.
#
# GRID: 6 backbones x 3 tasks x 5 seeds = 90 runs. The GRU's 15 come from the previous
# stage, so it is excluded here. Seeds {1,2,3,5,42} match every other seed cell.
#
# RF and XGBoost are CPU-only (sklearn/xgboost) and run first, sequentially -- they take
# ~1 min each and would otherwise idle a GPU slot. The four deep backbones then run
# 4-concurrent (2 per GPU); six concurrent produced NVML aborts on 2026-08-11.
#
# Results -> output/v3_tab2/, keeping the pre-repair output/tab2/ intact for comparison.
set -u
cd /path/to/repo
L=logs/v3_tab2; mkdir -p "$L" output/v3_tab2
CACHE=stage_segments_pooled_v3.npz
SEEDS="1 2 3 5 42"
log(){ echo "[$(date +%F_%T)] $*"; }

log "waiting for the Table-1 retrain chain to finish"
while pgrep -f "_after_recut_retrain\.sh" > /dev/null; do sleep 120; done
if [ ! -f "$CACHE" ]; then
  log "ABORT: $CACHE missing -- the chain aborted at its safety gate, so the clips were"
  log "       never verified repaired. Not training on unverified data."
  exit 1
fi
log "chain finished, cache present -- starting backbone sweep"

# ---------------- classical (CPU, ~1 min each) ------------------------------------
for arch in rf xgb; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      out="output/v3_tab2/${arch}_${tag}_s${seed}"
      [ -f "$out/results.json" ] && continue
      python3 train_pooled_eeg_classical.py --arch "$arch" $flag --cache "$CACHE" \
          --seed "$seed" --split_seed 49 --output "$out" \
          > "$L/${arch}_${tag}_s${seed}.log" 2>&1
      [ -f "$out/results.json" ] && log "DONE ${arch}_${tag}_s${seed}" \
                                 || log "FAILED ${arch}_${tag}_s${seed}"
    done
  done
done
log "classical block done"

# ---------------- deep (GPU, 4 concurrent) ----------------------------------------
deep(){
  local gpu=$1 arch=$2 flag=$3 tag=$4 seed=$5
  local out="output/v3_tab2/${arch}_${tag}_s${seed}"
  [ -f "$out/results.json" ] && return 0
  local try
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch "$arch" $flag \
        --cache "$CACHE" --seed "$seed" --split_seed 49 --output "$out" \
        > "$L/${arch}_${tag}_s${seed}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE ${arch}_${tag}_s${seed}"; return 0; }
    log "retry $try failed ${arch}_${tag}_s${seed}"; sleep 120
  done
  log "GAVE UP ${arch}_${tag}_s${seed}"
}

i=0
for arch in lstm eegnet conformer tcn; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      while [ "$(pgrep -cf 'python3 train_pooled_eeg\.py')" -ge 4 ]; do sleep 30; done
      gpu=$(( i % 2 )); i=$(( i + 1 ))
      log "launch ${arch}_${tag}_s${seed} gpu$gpu"
      deep "$gpu" "$arch" "$flag" "$tag" "$seed" &
      sleep 10
    done
  done
done
wait

log "===== Table 2 on repaired data: before vs after ====="
python3 - <<'PY'
import json, os, glob, statistics as st
def cell(pat):
    v = [json.load(open(p))['macro_f1']
         for p in sorted(glob.glob(pat)) if os.path.exists(p)]
    return (len(v), st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else None
print(f"{'backbone':11s} {'task':5s} {'BEFORE (corrupt)':>22s} {'AFTER (repaired)':>22s}")
for arch in ('gru','lstm','eegnet','conformer','tcn','rf','xgb'):
    for tag in ('bin','g3','g5'):
        if arch == 'gru':
            pre = {'bin':'output/seed_runs/eeg_s*/results.json',
                   'g3':'output/seed_runs/eeg_g3_s*/results.json',
                   'g5':'output/seed_runs/eeg_g5_s*/results.json'}[tag]
            post = f'output/v3_seed_runs/eeg_{tag}_s*/results.json'
        else:
            pre  = f'output/tab2/{arch}_{tag}_s*/results.json'
            post = f'output/v3_tab2/{arch}_{tag}_s*/results.json'
        o, n = cell(pre), cell(post)
        fo = f"{o[1]:.4f}+/-{o[2]:.4f} n={o[0]}" if o else "--"
        fn = f"{n[1]:.4f}+/-{n[2]:.4f} n={n[0]}" if n else "--"
        print(f"{arch:11s} {tag:5s} {fo:>22s} {fn:>22s}")
PY
log "===== backbone sweep finished ====="
