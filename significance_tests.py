#!/usr/bin/env python3
"""Plan item #1, significance half: McNemar on labels + bootstrap on macro-F1.

The plan's stated motivation is that "'backbone-agnostic' asserts a 0.028 macro-F1
spread is noise, but with n=1 you cannot show it IS noise". This script supplies the
tests, using predictions already on disk. No GPU, no retraining.

WHAT IS PAIRABLE, AND WHAT IS NOT
---------------------------------
Verified from the saved arrays:
  * All video runs at split_seed 49 share an identical validation set (n=5289, label
    vectors equal element-wise), so any two video models are PAIRED -> McNemar is valid.
  * All EEG runs at split_seed 49 likewise share theirs (n=5250).
  * Video (5289 clips) and EEG (5250 clips) do NOT share a validation set, and neither
    prediction file stores a clip identifier -- video saves `idx` (a positional index
    into its own item list), EEG saves only `sub` (subject). The EEG window cache
    records `clip_sess` = "subject/session" but not which clip within a session, and
    clips are numbered in ProcessPoolExecutor completion order, which is not stable.
    So per-clip video-vs-EEG pairing CANNOT be reconstructed from current artifacts,
    and McNemar between modalities is not available.
    -> Fix for later: have build_stage_segments_pooled.py store the clip directory path
       alongside clip_sess. One line, but it requires rebuilding the cache.
    -> Meanwhile the cross-modality comparison here uses unpaired bootstrap CIs on each
       modality's own validation set, which is weaker but honest.

TESTS
-----
  mcnemar(a, b)    exact binomial on discordant pairs (b=0/1 correctness vectors)
  boot_diff(a, b)  paired bootstrap over clips: distribution of macro-F1(a) - macro-F1(b)
  boot_ci(model)   bootstrap CI for one model's macro-F1
"""
import glob
import json
import os
from itertools import combinations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

RNG = np.random.default_rng(0)
B = 2000


def load(path, key='val_preds.npz'):
    f = os.path.join(path, key)
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return z['y'], z['pred']


def mcnemar(y, pa, pb):
    """Exact McNemar on the discordant pairs of two paired classifiers."""
    ca, cb = (pa == y).astype(int), (pb == y).astype(int)
    n01 = int(((ca == 0) & (cb == 1)).sum())   # a wrong, b right
    n10 = int(((ca == 1) & (cb == 0)).sum())   # a right, b wrong
    n = n01 + n10
    p = binomtest(n10, n, 0.5).pvalue if n else 1.0
    return n10, n01, p


def boot_diff(y, pa, pb, B=B):
    """Paired bootstrap over clips: CI on macro-F1(a) - macro-F1(b)."""
    idx = np.arange(len(y))
    d = np.empty(B)
    for i in range(B):
        s = RNG.choice(idx, len(idx), replace=True)
        d[i] = (f1_score(y[s], pa[s], average='macro', zero_division=0)
                - f1_score(y[s], pb[s], average='macro', zero_division=0))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ci(y, p, B=B):
    idx = np.arange(len(y))
    v = np.empty(B)
    for i in range(B):
        s = RNG.choice(idx, len(idx), replace=True)
        v[i] = f1_score(y[s], p[s], average='macro', zero_division=0)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def hdr(t):
    print(f'\n{"=" * 78}\n{t}\n{"=" * 78}')


# ---------------------------------------------------------------- backbones (C2)
hdr('C2 "backbone-agnostic": video detection backbones, PAIRED (n=5289)')
backbones = {
    'R(2+1)D s42*': 'output/seed_runs/vid_s1',          # seed series member
    'MViT':         'output/seed_runs/vid_mvit_s42',
}
loaded = {k: load(v) for k, v in backbones.items() if load(v)}
names = list(loaded)
for a, b in combinations(names, 2):
    ya, pa = loaded[a]
    yb, pb = loaded[b]
    assert (ya == yb).all(), 'validation sets differ -- not paired'
    n10, n01, p = mcnemar(ya, pa, pb)
    md, lo, hi = boot_diff(ya, pa, pb)
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f'  {a} vs {b}')
    print(f'    McNemar: {a} right/{b} wrong = {n10},  reverse = {n01},  p = {p:.4g}  -> {sig}')
    print(f'    bootstrap macro-F1 diff = {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]'
          f'  -> {"excludes 0" if lo * hi > 0 else "includes 0"}')

# ---------------------------------------------------- seed spread vs model diff
hdr('Is the backbone gap larger than seed noise? (video detection)')
seeds = [load(f'output/seed_runs/vid_s{s}') for s in (1, 2, 3, 4, 5)]
seeds = [s for s in seeds if s]
if len(seeds) > 1:
    f1s = [f1_score(y, p, average='macro', zero_division=0) for y, p in seeds]
    print(f'  R(2+1)D seed macro-F1 (n={len(f1s)}): mean={np.mean(f1s):.4f} '
          f'sd={np.std(f1s, ddof=1):.4f}  range [{min(f1s):.4f}, {max(f1s):.4f}]')
    # pairwise McNemar between seeds = the null distribution of "same model, new seed"
    ps = []
    for (ya, pa), (yb, pb) in combinations(seeds, 2):
        ps.append(mcnemar(ya, pa, pb)[2])
    print(f'  pairwise McNemar between SEEDS of the same model: '
          f'{sum(p < 0.05 for p in ps)}/{len(ps)} significant at p<0.05')
    print('    (seed pairs that test "significant" mark how much of this is clip-level '
          'noise rather than a real modelling difference)')

# --------------------------------------------------------- cross-modality (C1)
hdr('C1 video vs EEG: UNPAIRED bootstrap CIs (different validation sets)')
for task, vid, eeg in [('detection', 'output/seed_runs/vid_s1', 'output/seed_runs/eeg_s1'),
                       ('3-class',   'output/seed_runs/vid_g3_s1', 'output/seed_runs/eeg_g3_s1'),
                       ('5-class',   'output/seed_runs/vid_g5_s1', 'output/seed_runs/eeg_g5_s1')]:
    v = load(vid)
    e = load(eeg, 'val_clip_preds.npz')
    if not v or not e:
        print(f'  {task}: missing predictions'); continue
    mv, lv, hv = boot_ci(*v)
    me, le, he = boot_ci(*e)
    overlap = not (lv > he or le > hv)
    print(f'  {task:10s} video {mv:.4f} [{lv:.4f}, {hv:.4f}]   '
          f'EEG {me:.4f} [{le:.4f}, {he:.4f}]  -> CIs '
          f'{"OVERLAP" if overlap else "DISJOINT"}')
print('  NOTE: unpaired -- McNemar unavailable, see module docstring.')

# ------------------------------------------------------- tuned vs production EEG
hdr('Tuned vs production EEG preprocessing, PAIRED within modality (n=5250)')
for task, prod, tuned in [('detection', 'output/seed_runs/eeg_s1',
                           'output/eeg_tuned/gru_bin_w4s1_s1'),
                          ('5-class', 'output/seed_runs/eeg_g5_s1',
                           'output/eeg_tuned/gru_g5_w4s1_s1')]:
    a = load(prod, 'val_clip_preds.npz')
    b = load(tuned, 'val_clip_preds.npz')
    if not a or not b:
        print(f'  {task}: missing predictions'); continue
    if len(a[0]) != len(b[0]) or not (a[0] == b[0]).all():
        print(f'  {task}: validation sets differ (n={len(a[0])} vs {len(b[0])}) '
              f'-- the tuned cache re-windows the signal, so clip sets need not match; '
              f'skipping paired test')
        continue
    n10, n01, p = mcnemar(a[0], a[1], b[1])
    md, lo, hi = boot_diff(a[0], a[1], b[1])
    print(f'  {task}: McNemar p={p:.4g}   bootstrap diff {md:+.4f} [{lo:+.4f}, {hi:+.4f}]')

print()
