#!/bin/bash
# Waits for eeg_bin_fold4, then emits everything needed to update Table 3's detection
# row and regenerate fig:stability.
set -u
cd /path/to/repo
OUT=output/v3_subject_cv/eeg_bin_fold4
log(){ echo "[$(date +%F_%T)] $*"; }
log "waiting for eeg_bin_fold4"
while [ ! -f "$OUT/results.json" ]; do
  pgrep -f "train_pooled_eeg.py.*eeg_bin_fold4" > /dev/null || \
    pgrep -f "_bin_fold4_gpu1.sh" > /dev/null || { log "fold4 no longer running and no results"; exit 1; }
  sleep 120
done
log "fold4 done"
python3 make_stability_fig.py 2>&1
log "=== Table 3 detection row, repaired EEG ==="
python3 - <<'PY'
import json, os, statistics as st
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef
f1s, aus, mccs = [], [], []
for f in range(5):
    d = f'output/v3_subject_cv/eeg_bin_fold{f}'
    f1s.append(json.load(open(f'{d}/results.json'))['macro_f1'])
    z = np.load(f'{d}/val_clip_preds.npz', allow_pickle=True)
    aus.append(roc_auc_score(z['y'], z['probs'][:, 1]))
    mccs.append(matthews_corrcoef(z['y'], z['pred']))
def ms(v): return st.mean(v), st.stdev(v)
for nm, v in (('macro-F1', f1s), ('AUROC', aus), ('MCC', mccs)):
    m, s = ms(v)
    print(f"  {nm:9s} {m:.4f} +/- {s:.4f}   per-fold: {' '.join(f'{x:.3f}' for x in v)}")
print("\n  currently in paper (corrupt-era): 0.849 +/- 0.055 / 0.940 +/- 0.035 / 0.712 +/- 0.101")
PY
