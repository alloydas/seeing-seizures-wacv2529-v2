#!/bin/bash
# Detection (2-class) subject-disjoint EEG CV on the REPAIRED clips -- the missing
# Table 3 row.
#
# WHY THIS IS NEEDED. sh/_v3_subject_cv.sh only ran "g3 --group3" and "g5", so the
# repaired cohort has no detection folds. The detection row currently in the paper
# (EEG 0.849 +/- 0.055, AUROC 0.940 +/- 0.035) came from sh/_abl_subject.sh, which wrote
# to abl_subj/ -- a directory that no longer exists on disk. So that row is both
# corrupt-era (every EEG clip read before 2026-08-11 hit the mne.export defect) and
# unreproducible. Fig.~\ref{fig:stability} is built from its per-fold values, so it
# inherits the same problem.
#
# Concurrency counts ALL train_pooled_eeg processes so this shares the box with the
# ablation's w4 cell. Cap 4 (six concurrent produced NVML aborts on 2026-08-11).
set -u
cd /path/to/repo
CACHE=stage_segments_pooled_v3.npz
L=logs/v3_subject_cv; mkdir -p "$L" output/v3_subject_cv
log(){ echo "[$(date +%F_%T)] $*"; }

[ -f "$CACHE" ] || { log "ABORT: $CACHE missing"; exit 1; }
busy(){ pgrep -af "python3 train_pooled_eeg" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }

fold(){
  local gpu=$1 f=$2
  local out="output/v3_subject_cv/eeg_bin_fold${f}"
  [ -f "$out/results.json" ] && { log "SKIP fold${f} (done)"; return 0; }
  busy "$out" && { log "SKIP fold${f} (already running)"; return 0; }
  local try
  for try in 1 2 3; do
    log "START eeg_bin_fold${f} gpu$gpu (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch gru --group2 \
        --split subject --fold "$f" --n_folds 5 --cache "$CACHE" \
        --seed 42 --split_seed 49 --output "$out" \
        > "$L/eeg_bin_fold${f}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE eeg_bin_fold${f}"; return 0; }
    log "retry $try failed eeg_bin_fold${f}"; sleep 120
  done
  log "GAVE UP eeg_bin_fold${f}"
}

log "=== detection subject-disjoint EEG CV, repaired data ==="
i=0
for f in 0 1 2 3 4; do
  while [ "$(pgrep -cf 'python3 train_pooled_eeg\.py')" -ge 4 ]; do sleep 30; done
  gpu=$(( i % 2 )); i=$(( i + 1 ))
  fold "$gpu" "$f" &
  sleep 10
done
wait

log "===== Table 3 detection EEG row (repaired) ====="
python3 - <<'PY'
import json, glob, os, statistics as st
import numpy as np
from sklearn.metrics import roc_auc_score
f1s, aus = [], []
for f in range(5):
    d = f'output/v3_subject_cv/eeg_bin_fold{f}'
    if not os.path.exists(f'{d}/results.json'): continue
    f1s.append(json.load(open(f'{d}/results.json'))['macro_f1'])
    z = np.load(f'{d}/val_clip_preds.npz', allow_pickle=True)
    aus.append(roc_auc_score(z['y'], z['probs'][:, 1]))
if f1s:
    print(f"  per-fold macro-F1: {' '.join(f'{x:.3f}' for x in f1s)}")
    print(f"  macro-F1 {st.mean(f1s):.4f} +/- {st.stdev(f1s) if len(f1s)>1 else 0:.4f}  n={len(f1s)}")
    print(f"  AUROC    {st.mean(aus):.4f} +/- {st.stdev(aus) if len(aus)>1 else 0:.4f}")
    print("  video reference (corrupt-era, also unreproducible): 0.948 +/- 0.016, AUROC 0.985 +/- 0.006")
PY
log "===== finished ====="
