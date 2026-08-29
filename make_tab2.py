#!/usr/bin/env python3
"""Generate the LaTeX body of Table 2 (tab:eegarch) as mean +/- sd over seeds.

Reads results.json + val_clip_preds.npz for every (backbone, task, seed) on disk and
emits the tabular rows. Run with --check to see completeness without emitting LaTeX.

Seed sources:
  GRU  -> output/seed_runs/eeg_{s*,g3_s*,g5_s*}   (the main seed programme, 5 seeds)
  rest -> output/tab2/{arch}_{tag}_s{seed}        (sh/_tab2_seeds.sh, 5 seeds each)

AUROC is recomputed from the saved probabilities rather than read from results.json,
which stores only accuracy / balanced_accuracy / macro_f1.
"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np
from sklearn.metrics import roc_auc_score

ARCHS = [
    ('gru',       r'GRU~\cite{ganguly2025}'),
    ('lstm',      r'LSTM~\cite{hochreiter1997lstm}'),
    ('eegnet',    r'EEGNet~\cite{lawhern2018eegnet}'),
    ('conformer', r'EEG Conformer~\cite{song2022conformer}'),
    ('tcn',       r'TCN~\cite{bai2018tcn}'),
    ('rf',        r'Random Forest~\cite{breiman2001rf}'),
    ('xgb',       r'XGBoost~\cite{chen2016xgboost}'),
]
TASKS = ['bin', 'g3', 'g5']
GRU_PREFIX = {'bin': 'output/seed_runs/eeg_s',
              'g3':  'output/seed_runs/eeg_g3_s',
              'g5':  'output/seed_runs/eeg_g5_s'}


def dirs_for(arch, tag):
    if arch == 'gru':
        # eeg_s* also matches eeg_s7 etc.; exclude the 3-/5-class prefixes
        out = [d for d in glob.glob(GRU_PREFIX[tag] + '*') if os.path.isdir(d)]
        if tag == 'bin':
            out = [d for d in out if 'g3' not in d and 'g5' not in d]
            out += [d for d in glob.glob('output/seed_runs/eeg_bin_s*') if os.path.isdir(d)]
        return sorted(set(out))
    return sorted(glob.glob(f'output/tab2/{arch}_{tag}_s*'))


def metrics(d):
    p = os.path.join(d, 'results.json')
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    pc = r['per_class']
    P = sum(v['precision'] for v in pc.values()) / len(pc)
    R = sum(v['recall'] for v in pc.values()) / len(pc)
    z = np.load(os.path.join(d, 'val_clip_preds.npz'), allow_pickle=True)
    y, pr = z['y'], z['probs']
    A = (roc_auc_score(y, pr[:, 1]) if pr.shape[1] == 2
         else roc_auc_score(y, pr, multi_class='ovr', average='macro'))
    return P, R, r['macro_f1'], A


def cell(arch, tag):
    vals = [m for m in (metrics(d) for d in dirs_for(arch, tag)) if m]
    if not vals:
        return None
    cols = list(zip(*vals))
    return len(vals), [(st.mean(c), st.stdev(c) if len(c) > 1 else 0.0) for c in cols]


def fmt(mean, sd, best=False, second=False, show_sd=True):
    if not show_sd:
        sd = 0.0
    body = f'{mean:.3f}' + (r'\,{\scriptsize$\pm$' + f'{sd:.3f}' + '}' if sd > 0 else '')
    if best:
        return r'\textbf{' + f'{mean:.3f}' + '}' + (r'\,{\scriptsize$\pm$' + f'{sd:.3f}' + '}' if sd > 0 else '')
    if second:
        return r'\underline{' + f'{mean:.3f}' + '}' + (r'\,{\scriptsize$\pm$' + f'{sd:.3f}' + '}' if sd > 0 else '')
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report completeness only')
    ap.add_argument('--metrics', default='PRFA',
                    help='which metrics to emit per task: subset of P R F A')
    ap.add_argument('--sd-metrics', default='FA', dest='sd_metrics',
                    help='which of those carry a +/- sd (Option B: F1 and AUROC only; '
                         'P and R stay as plain means so the table keeps its 13 columns)')
    a = ap.parse_args()

    data = {}
    for arch, _ in ARCHS:
        for tag in TASKS:
            data[(arch, tag)] = cell(arch, tag)

    if a.check:
        print(f"{'backbone':11s} " + ''.join(f'{t:>26s}' for t in TASKS))
        for arch, _ in ARCHS:
            line = f'{arch:11s} '
            for tag in TASKS:
                c = data[(arch, tag)]
                line += f'{"n=" + str(c[0]) + " F1 " + format(c[1][2][0], ".3f") + "+/-" + format(c[1][2][1], ".3f"):>26s}' if c else f'{"-- none --":>26s}'
            print(line)
        n_done = sum(1 for c in data.values() if c and c[0] >= 5)
        print(f'\ncells with 5 seeds: {n_done}/21')
        return

    # per-column best / second (by mean), over the metrics requested
    order = {'P': 0, 'R': 1, 'F': 2, 'A': 3}
    idxs = [order[m] for m in a.metrics]
    rank = {}
    for tag in TASKS:
        for mi in idxs:
            vals = [(arch, data[(arch, tag)][1][mi][0])
                    for arch, _ in ARCHS if data[(arch, tag)]]
            vals.sort(key=lambda x: -x[1])
            if vals:
                rank[(tag, mi)] = (vals[0][0], vals[1][0] if len(vals) > 1 else None)

    for arch, label in ARCHS:
        cells = []
        for tag in TASKS:
            c = data[(arch, tag)]
            for mi in idxs:
                if not c:
                    cells.append('--'); continue
                m, s = c[1][mi]
                b, sec = rank.get((tag, mi), (None, None))
                show = 'PRFA'[mi] in a.sd_metrics
                cells.append(fmt(m, s, best=(arch == b), second=(arch == sec),
                                 show_sd=show))
        print(f'{label:40s} & ' + ' & '.join(cells) + r' \\')


if __name__ == '__main__':
    main()
