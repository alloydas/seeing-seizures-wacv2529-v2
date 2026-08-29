#!/bin/bash
# Autonomous chain: wait for the clip-EDF repair, verify it, rebuild the window cache,
# then retrain the EEG models on the repaired data across BOTH GPUs.
#
# Every EEG number in the paper was computed on clips whose biopotential channel had been
# crushed to 2-4 distinct values by mne.export.export_raw's shared physical range (69.2%
# of 24,214 clips, 2026-08-12 audit). The clips are being rewritten with per-channel
# ranges; this script picks up from there without further input.
#
# STAGES
#   0. wait for recut pass 2 to exit
#   1. SAFETY GATE -- re-quantify and refuse to train if the corpus is still bad.
#      Training on unrepaired data would silently reproduce the very numbers this whole
#      exercise exists to discard, so the chain stops rather than guesses.
#   2. rebuild the pooled window cache from the repaired clips -> stage_segments_pooled_v3.npz
#      (the existing .npz and _v2.npz were both built from corrupted clips and are dead)
#   3. retrain the six Table-1 EEG cells at 5 seeds each = 15 runs, 4 concurrent
#      (2 per GPU). Six concurrent EEG jobs triggered NVML aborts on 2026-08-11, so the
#      cap stays at 4; every run has a retry loop regardless.
#
# Results go to output/v3_seed_runs/ so the pre-repair results stay intact for comparison
# -- the size of the change IS the finding.
set -u
cd /path/to/repo
L=logs/v3; mkdir -p "$L" output/v3_seed_runs
log(){ echo "[$(date +%F_%T)] $*"; }

# ---------- stage 0: wait for the repair ------------------------------------------
log "waiting for the clip-EDF re-cut to finish"
while pgrep -f "recut_clip_edfs\.py" > /dev/null; do sleep 120; done
log "re-cut finished"

# ---------- stage 1: safety gate --------------------------------------------------
log "re-quantifying clip quality"
python3 quantify_clip_edf.py > "$L/quality_after.log" 2>&1
BAD=$(python3 - <<'PY'
import json
rows = json.load(open('output/clip_edf_quality.json'))
bad = sum(1 for r in rows if r['std'] == 0 or r['n_uniq'] < 100)
print(f"{100*bad/len(rows):.1f}")
PY
)
log "degenerate+flat after repair: ${BAD}%  (was 70.3%)"
if python3 -c "import sys; sys.exit(0 if float('$BAD') > 20 else 1)"; then
  log "ABORT: still ${BAD}% degenerate -- not training on unrepaired data."
  log "       inspect $L/quality_after.log and output/clip_edf_quality.json"
  exit 1
fi
log "gate passed"

# ---------- stage 2: rebuild the window cache -------------------------------------
CACHE=stage_segments_pooled_v3.npz
if [ ! -f "$CACHE" ]; then
  log "rebuilding window cache -> $CACHE"
  python3 build_stage_segments_pooled.py --win 6 --stride 3 --decim 8 \
      --workers 8 --out "$CACHE" > "$L/build_cache_v3.log" 2>&1
  if [ ! -f "$CACHE" ]; then
    log "ABORT: cache build failed"; tail -5 "$L/build_cache_v3.log"; exit 1
  fi
fi
log "cache ready ($(du -h "$CACHE" | cut -f1))"

# ---------- stage 3: retrain the Table-1 EEG cells --------------------------------
run(){   # $1=gpu $2=name $3=taskflag $4=seed
  local gpu=$1 name=$2 flag=$3 seed=$4
  local out="output/v3_seed_runs/$name"
  [ -f "$out/results.json" ] && { log "SKIP $name"; return 0; }
  local try
  for try in 1 2 3; do
    log "START $name on gpu$gpu (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch gru $flag \
        --cache "$CACHE" --seed "$seed" --split_seed 49 --output "$out" \
        > "$L/$name.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE $name"; return 0; }
    log "retry $try failed for $name"; sleep 120
  done
  log "GAVE UP $name"
}

log "=== retraining 15 EEG cells (3 tasks x 5 seeds), 4 concurrent ==="
i=0
for spec in "bin --group2" "g3 --group3" "g5 "; do
  set -- $spec; tag=$1; shift; flag="${*:-}"
  for seed in 1 2 3 5 42; do
    while [ "$(pgrep -cf 'python3 train_pooled_eeg\.py')" -ge 4 ]; do sleep 30; done
    gpu=$(( i % 2 )); i=$(( i + 1 ))
    run "$gpu" "eeg_${tag}_s${seed}" "$flag" "$seed" &
    sleep 10
  done
done
wait

log "===== retraining complete -- before/after ====="
python3 - <<'PY'
import json, os, statistics as st
def cell(dirs):
    v = [json.load(open(f'{d}/results.json'))['balanced_accuracy']
         for d in dirs if os.path.exists(f'{d}/results.json')]
    return (len(v), st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else None
old = {'bin': ['output/seed_runs/eeg_s1','output/seed_runs/eeg_s2','output/seed_runs/eeg_s3',
               'output/seed_runs/eeg_s7','output/seed_runs/eeg_bin_s42'],
       'g3':  [f'output/seed_runs/eeg_g3_s{s}' for s in (1,2,3,7,42)],
       'g5':  [f'output/seed_runs/eeg_g5_s{s}' for s in (1,2,3,5,42)]}
name = {'bin':'detection','g3':'3-class','g5':'5-class'}
print(f"{'cell':12s} {'BEFORE (corrupt clips)':>26s} {'AFTER (repaired)':>24s}")
for tag in ('bin','g3','g5'):
    o = cell(old[tag])
    n = cell([f'output/v3_seed_runs/eeg_{tag}_s{s}' for s in (1,2,3,5,42)])
    fo = f"{o[1]:.4f} +/- {o[2]:.4f} (n={o[0]})" if o else "--"
    fn = f"{n[1]:.4f} +/- {n[2]:.4f} (n={n[0]})" if n else "--"
    d  = f"   delta {n[1]-o[1]:+.4f}" if (o and n) else ""
    print(f"{name[tag]:12s} {fo:>26s} {fn:>24s}{d}")
PY
log "===== chain finished ====="
