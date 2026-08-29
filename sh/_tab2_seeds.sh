#!/bin/bash
# Table 2 (tab:eegarch) with mean +/- sd: seed-variance for all seven EEG backbones
# across all three task granularities.
#
# The table is currently every-cell-n=1 at seed 42 ("identical split, seed and
# 12-epoch budget"), so no standard deviation exists for any of its 84 values.
#
# GRID: 6 backbones x 3 tasks x 5 seeds {1,2,3,5,42} = 90 runs.
#   The GRU is excluded because its 5 seeds already exist under current code
#   (output/seed_runs/eeg_{s*,g3_s*,g5_s*}) -- 15 runs saved.
#   Seeds {1,2,3,5,42} match the set used for every other seed cell in the paper,
#   so Table 2's spread is directly comparable to Table 1's.
#
# All 90 are re-run under CURRENT code rather than reusing the July single-seed dirs
# as the seed-42 member: those predate the 2026-08-03 train_pooled_eeg.py edit, and
# although all six reproduced when checked (LSTM 0.0000, EEGNet -0.0001, RF 0.0000,
# XGB 0.0000, TCN +0.004, Conformer +0.016), Conformer's +0.016 is large enough that
# mixing old and new runs inside one mean would blur config drift into seed variance.
#
# CLASSICAL MODELS: RF and XGBoost hardcoded random_state=42; train_pooled_eeg_classical.py
# now takes --seed (default 42, so prior behaviour is unchanged) which flows to both the
# global RNG and the estimator. Without that patch their "sd" would have been 0 by
# construction, which would have looked like a result rather than an artefact.
#
# SCHEDULING: deep runs go 3-way total (not 6 -- see the abort note below); classical runs are CPU-only
# and take ~1-2 min, so they run first and sequentially. EEG training reads the npz
# window cache, so there is no video decode -- five concurrent EEG jobs measured load ~8
# on this 32-core box, versus ~100 for five video jobs.
set -u
cd /path/to/repo
L=logs/tab2; mkdir -p "$L" output/tab2
log(){ echo "[$(date +%F_%T)] $*"; }
SEEDS="1 2 3 5 42"

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

# deep: $1=gpu $2=arch $3=taskflag $4=tag $5=seed
deep(){
  local gpu=$1 arch=$2 flag=$3 tag=$4 seed=$5
  local out="output/tab2/${arch}_${tag}_s${seed}"
  [ -f "$out/results.json" ] && return 0
  busy "$out" && return 0
  local try
  for try in 1 2 3; do
    [ -f "$out/results.json" ] && break
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch "$arch" $flag \
        --seed "$seed" --split_seed 49 --output "$out" \
        > "$L/${arch}_${tag}_s${seed}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE ${arch}_${tag}_s${seed}"; return 0; }
    log "retry $try failed ${arch}_${tag}_s${seed}"; sleep 120
  done
  log "FAILED ${arch}_${tag}_s${seed} after 3 tries"
}

# ---------------- classical first (CPU, minutes) ----------------------------------
log "=== classical RF/XGB: 2 archs x 3 tasks x 5 seeds = 30 runs ==="
for arch in rf xgb; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      out="output/tab2/${arch}_${tag}_s${seed}"
      [ -f "$out/results.json" ] && continue
      python3 train_pooled_eeg_classical.py --arch "$arch" $flag \
          --seed "$seed" --split_seed 49 --output "$out" \
          > "$L/${arch}_${tag}_s${seed}.log" 2>&1
      [ -f "$out/results.json" ] && log "DONE ${arch}_${tag}_s${seed}" \
                                 || log "FAILED ${arch}_${tag}_s${seed}"
    done
  done
done

# ---------------- deep: 4 archs x 3 tasks x 5 seeds = 60 runs ---------------------
log "=== deep backbones: lstm/eegnet/conformer/tcn, 3 per GPU ==="
i=0
for arch in lstm eegnet conformer tcn; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      # cap at 3 concurrent trainers. 6-way produced 3 NVML aborts in 2 minutes
      # on 2026-08-11; NVML is broken by the driver mismatch so memory
      # pressure aborts hard instead of raising a clean OOM.
      while [ "$(pgrep -cf 'python3 train_pooled_eeg.py')" -ge 3 ]; do sleep 30; done
      gpu=$(( i % 2 )); i=$(( i + 1 ))
      log "launch ${arch}_${tag}_s${seed} on gpu${gpu}"
      deep "$gpu" "$arch" "$flag" "$tag" "$seed" &
      sleep 5
    done
  done
done
wait

log "===== Table 2 seed sweep complete ====="
python3 - <<'PY'
import json, os, glob, statistics as st
import numpy as np
from sklearn.metrics import roc_auc_score

def cell(arch, tag):
    """mean +/- sd of P/R/F1/AUROC over the seeds present for one (arch, task)."""
    if arch == 'gru':                      # GRU seeds live with the main seed program
        pre = {'bin': 'output/seed_runs/eeg_s', 'g3': 'output/seed_runs/eeg_g3_s',
               'g5': 'output/seed_runs/eeg_g5_s'}[tag]
        dirs = sorted(glob.glob(pre + '*'))
    else:
        dirs = sorted(glob.glob(f'output/tab2/{arch}_{tag}_s*'))
    P, R, F, A = [], [], [], []
    for d in dirs:
        p = os.path.join(d, 'results.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); pc = r['per_class']
        P.append(sum(v['precision'] for v in pc.values()) / len(pc))
        R.append(sum(v['recall'] for v in pc.values()) / len(pc))
        F.append(r['macro_f1'])
        z = np.load(os.path.join(d, 'val_clip_preds.npz'), allow_pickle=True)
        y, pr = z['y'], z['probs']
        A.append(roc_auc_score(y, pr[:, 1]) if pr.shape[1] == 2
                 else roc_auc_score(y, pr, multi_class='ovr', average='macro'))
    if not F:
        return None
    def ms(v):
        return f'{st.mean(v):.3f}' + (f'$\\pm${st.stdev(v):.3f}' if len(v) > 1 else '  (n=1)')
    return len(F), ms(P), ms(R), ms(F), ms(A)

print(f"{'backbone':11s} {'task':4s} {'n':>2s}  {'P':>14s} {'R':>14s} {'F1':>14s} {'AUC':>14s}")
for arch in ['gru', 'lstm', 'eegnet', 'conformer', 'tcn', 'rf', 'xgb']:
    for tag in ['bin', 'g3', 'g5']:
        c = cell(arch, tag)
        print(f'{arch:11s} {tag:4s} {c[0]:2d}  {c[1]:>14s} {c[2]:>14s} {c[3]:>14s} {c[4]:>14s}'
              if c else f'{arch:11s} {tag:4s}  -- no runs --')
PY
