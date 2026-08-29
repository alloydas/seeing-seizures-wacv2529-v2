#!/bin/bash
# Detection (2-class) subject-disjoint VIDEO CV -- the other half of the missing
# Table 3 detection row.
#
# WHY. The row currently in the paper reads video 0.948 +/- 0.016 / AUROC 0.985 +/- 0.006
# and EEG 0.849 +/- 0.055. Both halves came from sh/_abl_subject.sh, which wrote to
# abl_subj/vid_fold* and abl_subj/eeg_fold* -- a directory that no longer exists, so
# neither number is reproducible. sh/_v3_subject_cv_bin.sh is regenerating the EEG half;
# without this script the row would pair a fresh EEG number against an unverifiable video
# one, and Fig.~\ref{fig:stability} compares the two modalities' per-fold spread directly.
#
# Video clips were never touched by the mne.export defect (it only rewrote eeg.edf), so
# this is a reproducibility fix, not a correctness one -- the numbers should land close
# to the published 0.948.
#
# RECIPE. Matched to the video *grading* folds in output/subject_cv/vid_g3_fold* from
# sh/_master_queue.sh -- 12 epochs, batch 16, seed 42, split_seed 49 -- not the 10 epochs
# the vanished _abl_subject.sh used, so that all v3 subject-CV rows share one recipe.
#
# GPU 0 only: GPU 1 is carrying the ablation's s1.5 + r62 and eeg_bin_fold4. Two video
# trainers at a time; video is CPU-decode-bound, so --workers 8 each on a 32-core box.
set -u
cd /path/to/repo
L=logs/v3_subject_cv; mkdir -p "$L" output/v3_subject_cv
log(){ echo "[$(date +%F_%T)] $*"; }

busy(){ pgrep -af "python3 train_pooled\.py" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }

fold(){
  local f=$1
  local out="output/v3_subject_cv/vid_bin_fold${f}"
  [ -f "$out/results.json" ] && { log "SKIP fold${f} (done)"; return 0; }
  busy "$out" && { log "SKIP fold${f} (already running)"; return 0; }
  local try
  for try in 1 2 3; do
    log "START vid_bin_fold${f} (try $try)"
    CUDA_VISIBLE_DEVICES=0 python3 train_pooled.py --arch r2plus1d --group2 \
        --split subject --fold "$f" --n_folds 5 --epochs 12 --batch_size 16 \
        --workers 8 --seed 42 --split_seed 49 --output "$out" \
        > "$L/vid_bin_fold${f}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE vid_bin_fold${f}"; return 0; }
    log "retry $try failed vid_bin_fold${f} -- tail:"
    tail -3 "$L/vid_bin_fold${f}.log" | cut -c1-140
    sleep 180
  done
  log "GAVE UP vid_bin_fold${f}"
}

log "=== detection subject-disjoint VIDEO CV, 2 concurrent on GPU 0 ==="
for f in 0 1 2 3 4; do
  while [ "$(pgrep -cf 'python3 train_pooled\.py')" -ge 2 ]; do sleep 60; done
  fold "$f" &
  sleep 20
done
wait

log "===== Table 3 detection row, both modalities (repaired/regenerated) ====="
python3 - <<'PY'
import json, os, statistics as st
import numpy as np
from sklearn.metrics import roc_auc_score
def row(pre, npz):
    f1s, aus = [], []
    for f in range(5):
        d = f'output/v3_subject_cv/{pre}_fold{f}'
        if not os.path.exists(f'{d}/results.json'): continue
        f1s.append(json.load(open(f'{d}/results.json'))['macro_f1'])
        p = f'{d}/{npz}'
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            aus.append(roc_auc_score(z['y'], z['probs'][:, 1]))
    if not f1s: return None
    return (len(f1s), st.mean(f1s), st.stdev(f1s) if len(f1s) > 1 else 0.0,
            st.mean(aus) if aus else float('nan'),
            st.stdev(aus) if len(aus) > 1 else 0.0, f1s)
for pre, npz, name in (('vid_bin', 'val_preds.npz', 'Video'),
                       ('eeg_bin', 'val_clip_preds.npz', 'EEG')):
    r = row(pre, npz)
    if r:
        n, m, s, am, asd, f1s = r
        print(f"  {name:6s} macro-F1 {m:.4f} +/- {s:.4f}   AUROC {am:.4f} +/- {asd:.4f}  n={n}")
        print(f"         per-fold: {' '.join(f'{x:.3f}' for x in f1s)}")
print("  paper (unreproducible): Video 0.948 +/- 0.016 / 0.985 +/- 0.006,"
      "  EEG 0.849 +/- 0.055 / 0.940 +/- 0.035")
PY
log "===== finished ====="
