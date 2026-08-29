#!/usr/bin/env python3
"""
Table 1 (tab:results) with mean +/- std over five seeds for EVERY metric, not just
macro-F1.

Video is scored from each run's val_preds.npz; EEG is re-scored from val_window_preds.npz
under the pooling rule chosen by the plan-#8 sweep (logmean by default), so the EEG
numbers here match Tables 2 and 3.

Prints the per-seed values, the mean +/- std table, and paste-ready LaTeX rows with
best/second-best markup resolved on the means -- with ties (differences that vanish at
three decimals) left unmarked rather than arbitrarily broken.
"""
import argparse, glob, json, os, statistics as st, sys

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRICS = ["P", "R", "F1", "AUROC", "MCC"]
TASKS = [("bin", "2-class (detection)"), ("g3", "3-class (severity)"),
         ("g5", "5-class (severity)")]

VID = {"bin": "output/seed_runs/vid_s*",
       "g3":  "output/seed_runs/vid_g3_s*",
       "g5":  "output/seed_runs/vid_g5_s*"}
EEG = {t: f"output/v3_seed_runs/eeg_{t}_s*" for t, _ in TASKS}


def _m(y, pred, prob):
    auc = (roc_auc_score(y, prob[:, 1]) if prob.shape[1] == 2
           else roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    return {"P": precision_score(y, pred, average="macro", zero_division=0),
            "R": recall_score(y, pred, average="macro", zero_division=0),
            "F1": f1_score(y, pred, average="macro", zero_division=0),
            "AUROC": auc,
            "MCC": matthews_corrcoef(y, pred)}


def video_runs(pat):
    out = []
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "val_preds.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        out.append((os.path.basename(d), _m(z["y"], z["pred"], z["probs"])))
    return out


def eeg_runs(pat, agg):
    from train_pooled_eeg import aggregate_clip
    out = []
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "val_window_preds.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        cp = aggregate_clip(z["probs"], z["cid"], list(z["clips"]), agg, 0.25, 1.0)
        out.append((os.path.basename(d), _m(z["y"], cp.argmax(1), cp)))
    return out


def agg_stats(runs):
    return {k: (st.mean([r[1][k] for r in runs]),
                st.stdev([r[1][k] for r in runs]) if len(runs) > 1 else 0.0)
            for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="logmean")
    a = ap.parse_args()

    table = {}
    for t, name in TASKS:
        v = video_runs(VID[t])
        e = eeg_runs(EEG[t], a.agg)
        table[t] = {"Video": (v, agg_stats(v)), "EEG": (e, agg_stats(e))}
        print(f"\n--- {name} ---")
        for mod in ("Video", "EEG"):
            runs, _ = table[t][mod]
            print(f"  {mod:6s} n={len(runs)}: " +
                  "  ".join(f"{n.split('_')[-1]}={m['F1']:.4f}" for n, m in runs))

    print("\n" + "=" * 92)
    print(f"TABLE 1 -- mean +/- std over five seeds (EEG pooling: {a.agg})")
    print("=" * 92)
    print(f"{'task':22s} {'mod':6s} " + " ".join(f"{k:>14s}" for k in METRICS))
    for t, name in TASKS:
        for mod in ("Video", "EEG"):
            _, s = table[t][mod]
            print(f"{name:22s} {mod:6s} " +
                  " ".join(f"{s[k][0]:.3f}±{s[k][1]:.3f}" .rjust(14) for k in METRICS))

    print("\n" + "=" * 92)
    print("LaTeX rows (markup on the means; ties at three decimals left unmarked)")
    print("=" * 92)
    for t, name in TASKS:
        rows = {}
        for mod in ("Video", "EEG"):
            _, s = table[t][mod]
            rows[mod] = s
        cells = {}
        for mod in ("Video", "EEG"):
            other = "EEG" if mod == "Video" else "Video"
            parts = []
            for k in METRICS:
                m, sd = rows[mod][k]
                om = rows[other][k][0]
                body = f"{m:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}"
                if abs(m - om) < 5e-4:          # identical once printed
                    parts.append(body)
                elif m > om:
                    parts.append(f"\\textbf{{{m:.3f}}}\\,{{\\scriptsize$\\pm${sd:.3f}}}")
                else:
                    parts.append(f"\\underline{{{m:.3f}}}\\,{{\\scriptsize$\\pm${sd:.3f}}}")
            cells[mod] = " & ".join(parts)
        print(f"\\multirow{{2}}{{*}}{{{name}}} & Video & {cells['Video']} \\\\")
        print(f"{'':<{len(name)+22}} & EEG   & {cells['EEG']} \\\\")
        if t != "g5":
            print("\\midrule")

    json.dump({t: {m: table[t][m][1] for m in ("Video", "EEG")} for t, _ in TASKS},
              open("tex_wacv/tab1_meansd.json", "w"), indent=2)
    print("\nwrote tex_wacv/tab1_meansd.json")


if __name__ == "__main__":
    main()
