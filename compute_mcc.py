#!/usr/bin/env python3
"""Matthews Correlation Coefficient (MCC) for every completed cell.

MCC generalises to K classes as Gorodkin's R_K: +1 perfect, 0 chance, negative worse
than chance. It is the metric of choice here because it is the only common one that
accounts for all four confusion-matrix quadrants and stays near 0 for a classifier that
merely exploits the class prior -- which matters a lot on this cohort, where 5-class
validation has 2688 non-seizure clips against 34 Stage-5 clips.

Contrast with what the paper currently reports:
  * accuracy      inflated by the majority class (5-class video accuracy ~0.90 while
                  balanced accuracy is ~0.65)
  * macro-F1      ignores true negatives entirely
  * balanced acc  averages recall only, so it rewards over-prediction of rare classes

Reads y/pred from val_preds.npz (video) and val_clip_preds.npz (EEG); no retraining.
"""
import glob
import json
import os
import statistics as st

import numpy as np
from sklearn.metrics import matthews_corrcoef


def mcc_of(d):
    for fn in ("val_preds.npz", "val_clip_preds.npz"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return float(matthews_corrcoef(z["y"], z["pred"]))
    return None


def agg(dirs):
    v = [m for m in (mcc_of(d) for d in dirs) if m is not None]
    if not v:
        return None
    return len(v), st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def show(title, groups):
    print(f'\n{title}\n' + '-' * len(title))
    for lbl, dirs in groups:
        a = agg(dirs)
        if not a:
            print(f'  {lbl:22s} -- no runs --'); continue
        n, m, s = a
        print(f'  {lbl:22s} n={n}  MCC = {m:.4f}' + (f' +/- {s:.4f}' if n > 1 else ''))


S = 'output/seed_runs'
show('Session-disjoint, 5 seeds (Table 1 cells)', [
    ('video detection', [f'{S}/vid_s{i}' for i in (1, 2, 3, 4, 5)]),
    ('video 3-class',   [f'{S}/vid_g3_s{i}' for i in (1, 2, 3, 4, 42)]),
    ('video 5-class',   [f'{S}/vid_g5_s{i}' for i in (1, 2, 3, 4, 5)]),
    ('EEG detection',   [f'{S}/eeg_s{i}' for i in (1, 2, 3, 7)] + [f'{S}/eeg_bin_s42']),
    ('EEG 3-class',     [f'{S}/eeg_g3_s{i}' for i in (1, 2, 3, 7, 42)]),
    ('EEG 5-class',     [f'{S}/eeg_g5_s{i}' for i in (1, 2, 3, 5, 42)]),
])

show('Subject-disjoint 5-fold CV (Table 3 cells)', [
    ('video 3-class', [f'output/subject_cv/vid_g3_fold{f}' for f in range(5)]),
    ('video 5-class', [f'output/subject_cv/vid_g5_fold{f}' for f in range(5)]),
    ('EEG 3-class',   [f'output/subject_cv/eeg_g3_fold{f}' for f in range(5)]),
    ('EEG 5-class',   [f'output/subject_cv/eeg_g5_fold{f}' for f in range(5)]),
])

show('EEG backbones, 5-class (Table 2 rows, seeds so far)', [
    (a, sorted(glob.glob(f'output/tab2/{a}_g5_s*')) if a != 'gru'
        else sorted(glob.glob(f'{S}/eeg_g5_s*')))
    for a in ('gru', 'lstm', 'eegnet', 'conformer', 'tcn', 'rf', 'xgb')
])

show('Temporal sampling, 5-class (Table 4 column)', [
    (f'T={t}', [f'{S}/vid_g5_f{t}']) for t in (1, 4, 8, 16, 32, 64)
])

# how much does MCC disagree with the metrics already in the paper?
print('\nMCC vs reported metrics, video 5-class seed 1')
print('-' * 46)
z = np.load(f'{S}/vid_g5_s1/val_preds.npz', allow_pickle=True)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
y, p = z['y'], z['pred']
print(f'  accuracy          {accuracy_score(y, p):.4f}')
print(f'  balanced accuracy {balanced_accuracy_score(y, p):.4f}')
print(f'  macro-F1          {f1_score(y, p, average="macro", zero_division=0):.4f}')
print(f'  MCC               {matthews_corrcoef(y, p):.4f}')
