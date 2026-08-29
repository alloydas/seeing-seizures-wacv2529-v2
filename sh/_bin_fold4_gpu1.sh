#!/bin/bash
# eeg_bin_fold4 retry on GPU 1.
# All three of the driver's attempts ran on GPU 0 between 13:15 and 13:20, while the
# RN243 swin sweep held GPU 0 down to ~1.1 GiB free. Each died in ~26 s with the NVML
# assert at CUDACachingAllocator.cpp:806 -- the allocator's OOM path, which cannot raise
# a clean OOM while the host NVML driver/library mismatch persists. GPU 1 has ~9 GiB free.
set -u
cd /path/to/repo
OUT=output/v3_subject_cv/eeg_bin_fold4
log(){ echo "[$(date +%F_%T)] $*"; }
for try in 1 2 3; do
  [ -f "$OUT/results.json" ] && break
  log "START eeg_bin_fold4 on GPU1 (try $try)"
  CUDA_VISIBLE_DEVICES=1 python3 train_pooled_eeg.py --arch gru --group2 \
      --split subject --fold 4 --n_folds 5 --cache stage_segments_pooled_v3.npz \
      --seed 42 --split_seed 49 --output "$OUT" \
      > logs/v3_subject_cv/eeg_bin_fold4.log 2>&1
  [ -f "$OUT/results.json" ] && { log "DONE"; break; }
  log "try $try failed -- tail:"; tail -3 logs/v3_subject_cv/eeg_bin_fold4.log | cut -c1-140; sleep 180
done
log "===== Table 3 detection EEG row (repaired, 5 folds) ====="
python3 - <<'PY'
import json, os, statistics as st
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
    print("  per-fold macro-F1: " + " ".join(f"{x:.3f}" for x in f1s))
    print(f"  macro-F1 {st.mean(f1s):.4f} +/- {st.stdev(f1s) if len(f1s)>1 else 0:.4f}  n={len(f1s)}")
    print(f"  AUROC    {st.mean(aus):.4f} +/- {st.stdev(aus) if len(aus)>1 else 0:.4f}")
PY
