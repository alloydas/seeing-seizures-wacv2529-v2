#!/bin/bash
# Train one video and one EEG model that CAN be paired per clip, then run the paired
# significance test that was previously impossible.
#
# Background: video and EEG validation sets differ (5289 vs 5250 clips) and neither
# prediction file carried a clip identifier -- video saved a positional `idx`, EEG only
# `sub` -- so video-vs-EEG McNemar could not be computed and C1 rested on unpaired
# bootstrap CIs. Both trainers now save clip paths, and the rebuilt cache
# (stage_segments_pooled_v2.npz) stores clip_path with deterministic ordering.
#
# 5-class chosen: it is the contested cell (video 0.6432 +/- 0.0055 vs EEG
# 0.4483 +/- 0.0267 session-disjoint) and the one where video's margin is largest.
# Seed 42 / split_seed 49 to match the rest of the project.
#
# EEG first (~1 h) so the cheap half lands early; video (~4 h) after. Both on GPU1 --
# r2p1d_448 still holds GPU0 and is the heaviest job in the project.
#
# NOTE: the two splits are computed independently (split_sessions runs over each
# modality's own item list), so the val sets need not coincide. The point of the paths
# is precisely that the paired test can then run on their INTERSECTION rather than
# assuming they match. If the intersection turns out small, that is itself the finding
# and the unpaired CIs remain the honest fallback.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/paired; mkdir -p "$L" output/paired
log(){ echo "[$(date +%F_%T)] $*"; }

run(){   # $1=name  $2=outdir  $3.. = cmd
  local name=$1 out=$2; shift 2
  [ -f "$out/results.json" ] && { log "SKIP $name (done)"; return 0; }
  local try
  for try in 1 2; do
    log "START $name (try $try)"
    "$@" > "$L/$name.log" 2>&1
    rc=$?
    [ -f "$out/results.json" ] && { log "DONE $name (rc=$rc)"; return 0; }
    log "FAILED $name try $try (rc=$rc); tail:"; tail -4 "$L/$name.log"; sleep 60
  done
  log "GAVE UP $name"; return 1
}

run eeg_g5_v2 output/paired/eeg_g5_v2 \
    python3 train_pooled_eeg.py --arch gru --cache stage_segments_pooled_v2.npz \
      --seed 42 --split_seed 49 --output output/paired/eeg_g5_v2

run vid_g5_v2 output/paired/vid_g5_v2 \
    python3 train_pooled.py --arch r2plus1d --epochs 12 --batch_size 16 --workers 8 \
      --seed 42 --split_seed 49 --output output/paired/vid_g5_v2

log "===== paired video-vs-EEG test ====="
python3 - <<'PY'
import numpy as np, os
from scipy.stats import binomtest
from sklearn.metrics import f1_score

v, e = 'output/paired/vid_g5_v2/val_preds.npz', 'output/paired/eeg_g5_v2/val_clip_preds.npz'
if not (os.path.exists(v) and os.path.exists(e)):
    print('  predictions missing'); raise SystemExit

V, E = np.load(v, allow_pickle=True), np.load(e, allow_pickle=True)
if 'path' not in V.files or 'path' not in E.files:
    print(f'  no clip paths (video={V.files}, eeg={E.files}) -- pairing unavailable')
    raise SystemExit

# EEG paths come from the cache and are clip DIRECTORIES; video items may be the
# video.mp4 inside them. Normalise to the directory before intersecting.
def norm(a):
    return np.array([p[:-len('/video.mp4')] if p.endswith('/video.mp4') else p for p in a])

vp, ep = norm(V['path']), norm(E['path'])
vi = {p: i for i, p in enumerate(vp)}
common = [p for p in ep if p in vi]
print(f'  video val={len(vp)}  EEG val={len(ep)}  INTERSECTION={len(common)}')
if len(common) < 100:
    print('  intersection too small for a meaningful paired test'); raise SystemExit

ei = {p: i for i, p in enumerate(ep)}
vidx = np.array([vi[p] for p in common]); eidx = np.array([ei[p] for p in common])
yv, ye = V['y'][vidx], E['y'][eidx]
print(f'  labels agree on the intersection: {(yv == ye).all()}')
pv, pe = V['pred'][vidx], E['pred'][eidx]
cv, ce = (pv == yv).astype(int), (pe == ye).astype(int)
n10 = int(((cv == 1) & (ce == 0)).sum()); n01 = int(((cv == 0) & (ce == 1)).sum())
p = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
print(f'\n  video right / EEG wrong = {n10}')
print(f'  EEG right / video wrong = {n01}')
print(f'  McNemar exact p = {p:.4g}  -> {"SIGNIFICANT" if p < 0.05 else "not significant"}')
print(f'\n  macro-F1 on the paired subset: video {f1_score(yv, pv, average="macro", zero_division=0):.4f}'
      f'   EEG {f1_score(ye, pe, average="macro", zero_division=0):.4f}')
PY
