#!/usr/bin/env python3
"""
Emits ready-to-paste LaTeX rows for tab:eegarch from the backbone sweep in
output/v3_bestcfg/, with best/second-best markup computed per column.

Table body is the seed-42 run of each cell, matching how tab:results and tab:vidarch
are built; the five-seed spread is printed separately as a check on whether any
ordering in the table is inside seed noise.
"""
import argparse, glob, json, os, statistics as st

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

ROWS = [
    ("GRU~\\cite{ganguly2025}",                 "gru",       "deep"),
    ("LSTM~\\cite{hochreiter1997lstm}",         "lstm",      "deep"),
    ("EEGNet~\\cite{lawhern2018eegnet}",        "eegnet",    "deep"),
    ("EEG Conformer~\\cite{song2022conformer}", "conformer", "deep"),
    ("TCN~\\cite{bai2018tcn}",                  "tcn",       "deep"),
    ("Random Forest~\\cite{breiman2001rf}",     "rf",        "classical"),
    ("XGBoost~\\cite{chen2016xgboost}",         "xgb",       "classical"),
]
TASKS = ("bin", "g3", "g5")


def metrics(d):
    p = os.path.join(d, "val_clip_preds.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    y, pred, P = z["y"], z["pred"], z["probs"]
    auc = (roc_auc_score(y, P[:, 1]) if P.shape[1] == 2
           else roc_auc_score(y, P, multi_class="ovr", average="macro"))
    return {
        "P": precision_score(y, pred, average="macro", zero_division=0),
        "R": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/v3_bestcfg")
    ap.add_argument("--seed", default="42")
    a = ap.parse_args()

    cells, missing = {}, []
    for _, arch, _ in ROWS:
        for t in TASKS:
            d = f"{a.base}/{arch}_{t}_s{a.seed}"
            m = metrics(d)
            if m is None:
                missing.append(f"{arch}_{t}_s{a.seed}")
            cells[(arch, t)] = m

    if missing:
        print(f"MISSING {len(missing)} seed-{a.seed} cells:")
        for x in missing:
            print("   ", x)
        print("\nrefusing to emit a partial table.")
        return

    # best / second-best per (task, metric) column
    mark = {}
    for t in TASKS:
        for k in ("P", "R", "F1", "AUC", "MCC"):
            vals = sorted(((cells[(arch, t)][k], arch) for _, arch, _ in ROWS), reverse=True)
            mark[(t, k, vals[0][1])] = "b"
            if len(vals) > 1:
                mark[(t, k, vals[1][1])] = "u"

    def fmt(arch, t, k):
        v = cells[(arch, t)][k]
        s = f"{v:.3f}"
        m = mark.get((t, k, arch))
        return f"\\textbf{{{s}}}" if m == "b" else (f"\\underline{{{s}}}" if m == "u" else s)

    print("\n" + "=" * 78)
    print("LaTeX rows for tab:eegarch  (seed 42, best configuration)")
    print("=" * 78)
    width = max(len(n) for n, _, _ in ROWS) + 1
    last = None
    for name, arch, kind in ROWS:
        if kind != last:
            label = "Deep" if kind == "deep" else "Classical (band-power $+$ statistical features)"
            print(f"\\midrule" if last else "")
            print(f"\\multicolumn{{16}}{{@{{}}l}}{{\\itshape {label}}}\\\\")
            last = kind
        cellstr = " & ".join(fmt(arch, t, k) for t in TASKS
                             for k in ("P", "R", "F1", "AUC", "MCC"))
        print(f"{name:<{width}} & {cellstr} \\\\")

    print("\n" + "=" * 78)
    print("five-seed macro-F1 (mean +/- sd) -- is any ordering inside seed noise?")
    print("=" * 78)
    print(f"{'backbone':14s} {'detection':>18s} {'3-class':>18s} {'5-class':>18s}")
    for name, arch, _ in ROWS:
        line = f"{arch:14s}"
        for t in TASKS:
            v = []
            for p in sorted(glob.glob(f"{a.base}/{arch}_{t}_s*/results.json")):
                v.append(json.load(open(p))["macro_f1"])
            if v:
                m = st.mean(v)
                s = st.stdev(v) if len(v) > 1 else 0.0
                line += f"  {m:.4f}±{s:.4f} n={len(v)}"
            else:
                line += f"{'--':>18s}"
        print(line)


if __name__ == "__main__":
    main()
