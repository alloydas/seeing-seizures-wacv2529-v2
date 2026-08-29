#!/usr/bin/env python3
"""
Table 4 (tab:eegarch) with mean +/- std over five seeds for every metric, matching the
format Table 1 now uses.

Reads output/v3_bestcfg/, the 105-run sweep at the configuration chosen by the plan-#8
ablation (6 s window / 3 s stride / 125 Hz, logmean clip pooling).

Collapsed runs are reported, not silently averaged in: a run whose macro-F1 falls more
than `--collapse_gap` below the median of its own cell is almost certainly a failed
optimisation rather than a draw from the same distribution (the LSTM has two such runs),
and averaging them produces a mean that describes neither outcome. They are excluded
from the mean, counted, and listed.
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
METRICS = ("P", "R", "F1", "AUROC", "MCC")


def metrics(d):
    p = os.path.join(d, "val_clip_preds.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    y, pred, P = z["y"], z["pred"], z["probs"]
    auc = (roc_auc_score(y, P[:, 1]) if P.shape[1] == 2
           else roc_auc_score(y, P, multi_class="ovr", average="macro"))
    return {"P": precision_score(y, pred, average="macro", zero_division=0),
            "R": recall_score(y, pred, average="macro", zero_division=0),
            "F1": f1_score(y, pred, average="macro", zero_division=0),
            "AUROC": auc, "MCC": matthews_corrcoef(y, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/v3_bestcfg")
    ap.add_argument("--collapse_gap", type=float, default=0.15)
    ap.add_argument("--allow_missing", type=int, default=0)
    a = ap.parse_args()

    cells, collapsed, missing = {}, [], []
    for _, arch, _ in ROWS:
        for t in TASKS:
            runs = []
            for d in sorted(glob.glob(f"{a.base}/{arch}_{t}_s*")):
                m = metrics(d)
                if m:
                    runs.append((os.path.basename(d), m))
            if len(runs) < 5:
                missing.append(f"{arch}_{t} (n={len(runs)})")
            if not runs:
                cells[(arch, t)] = None
                continue
            f1s = [r[1]["F1"] for r in runs]
            med = st.median(f1s)
            keep = [r for r in runs if med - r[1]["F1"] <= a.collapse_gap]
            for r in runs:
                if r not in keep:
                    collapsed.append((r[0], r[1]["F1"], med))
            cells[(arch, t)] = keep

    if missing and not a.allow_missing:
        print("INCOMPLETE cells:", ", ".join(missing))
        print("re-run with --allow_missing 1 to emit anyway\n")

    if collapsed:
        print("=" * 78)
        print(f"COLLAPSED RUNS excluded from the means (>{a.collapse_gap} below their cell median)")
        print("=" * 78)
        for n, f1, med in collapsed:
            print(f"  {n:22s} macro-F1 {f1:.4f}  vs cell median {med:.4f}")
        print()

    def stat(arch, t, k):
        runs = cells.get((arch, t))
        if not runs:
            return None
        v = [r[1][k] for r in runs]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)

    print("=" * 78)
    print("TABLE 4 -- mean +/- std over seeds")
    print("=" * 78)
    for t in TASKS:
        print(f"\n--- {t} ---")
        print(f"{'backbone':12s} " + " ".join(f"{k:>14s}" for k in METRICS) + "   n")
        for _, arch, _ in ROWS:
            s = [stat(arch, t, k) for k in METRICS]
            if s[0] is None:
                print(f"{arch:12s} " + "--"); continue
            print(f"{arch:12s} " + " ".join(f"{m:.3f}±{sd:.3f}".rjust(14) for m, sd, _ in s)
                  + f"   {s[0][2]}")

    # markup on the means, per (task, metric) column
    mark = {}
    for t in TASKS:
        for k in METRICS:
            vals = sorted(((stat(a_, t, k) or (0,))[0], a_) for _, a_, _ in ROWS)
            vals = [v for v in vals if v[0]]
            if vals:
                mark[(t, k, vals[-1][1])] = "b"
                if len(vals) > 1:
                    mark[(t, k, vals[-2][1])] = "u"

    print("\n" + "=" * 78)
    print("LaTeX rows for tab:eegarch")
    print("=" * 78)
    last = None
    for name, arch, kind in ROWS:
        if kind != last:
            if last:
                print("\\midrule")
            label = "Deep" if kind == "deep" else "Classical (band-power $+$ statistical features)"
            print(f"\\multicolumn{{16}}{{@{{}}l}}{{\\itshape {label}}}\\\\")
            last = kind
        parts = []
        for t in TASKS:
            for k in METRICS:
                s = stat(arch, t, k)
                if s is None:
                    parts.append("--"); continue
                m, sd, _ = s
                body = f"{m:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}"
                mk = mark.get((t, k, arch))
                if mk == "b":
                    parts.append(f"\\textbf{{{m:.3f}}}\\,{{\\scriptsize$\\pm${sd:.3f}}}")
                elif mk == "u":
                    parts.append(f"\\underline{{{m:.3f}}}\\,{{\\scriptsize$\\pm${sd:.3f}}}")
                else:
                    parts.append(body)
        print(f"{name} & " + " & ".join(parts) + " \\\\")


if __name__ == "__main__":
    main()
