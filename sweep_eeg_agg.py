#!/usr/bin/env python3
"""Re-sweep the clip-aggregation axis of plan item #8 WITHOUT retraining.

train_pooled_eeg.py now saves val_window_preds.npz (per-window probabilities at the
best epoch). Clip pooling happens strictly after the model runs, so every aggregation
variant can be scored from that file -- the mean/max/topk/logmean/attn comparison
costs seconds instead of one training run each.

Usage:
    python3 sweep_eeg_agg.py output/seed_runs/eeg_g5_s42 [more dirs ...]
    python3 sweep_eeg_agg.py --glob 'output/seed_runs/eeg_win*'
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pooled_eeg import aggregate_clip  # noqa: E402  (single source of truth)

AGGS = ["mean", "max", "topk", "logmean", "attn"]


def score_dir(d, aggs, topk_frac, temp):
    f = os.path.join(d, "val_window_preds.npz")
    if not os.path.exists(f):
        return None, f"no val_window_preds.npz (retrain needed: it is only written by runs after this patch)"
    z = np.load(f, allow_pickle=True)
    P, cid, clips, y = z["probs"], z["cid"], z["clips"], z["y"]
    rows = {}
    for how in aggs:
        cp = aggregate_clip(P, cid, list(clips), how, topk_frac, temp)
        pred = cp.argmax(1)
        rows[how] = dict(
            balanced_accuracy=float(balanced_accuracy_score(y, pred)),
            macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
            accuracy=float(accuracy_score(y, pred)),
        )
    return rows, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="run directories to score")
    ap.add_argument("--glob", default=None, help="glob of run directories")
    ap.add_argument("--aggs", nargs="+", default=AGGS, choices=AGGS)
    ap.add_argument("--topk_frac", type=float, default=0.25)
    ap.add_argument("--agg_temp", type=float, default=1.0)
    ap.add_argument("--json_out", default=None, help="write all results to this JSON")
    a = ap.parse_args()

    dirs = list(a.dirs) + (sorted(globmod.glob(a.glob)) if a.glob else [])
    if not dirs:
        ap.error("give at least one run directory or --glob")

    allres, skipped = {}, []
    hdr = f"{'run':34s}" + "".join(f"{h:>12s}" for h in a.aggs)
    print(hdr); print("-" * len(hdr))
    for d in dirs:
        rows, err = score_dir(d, a.aggs, a.topk_frac, a.agg_temp)
        if rows is None:
            skipped.append((d, err)); continue
        allres[d] = rows
        print(f"{os.path.basename(d.rstrip('/')):34s}"
              + "".join(f"{rows[h]['balanced_accuracy']:12.4f}" for h in a.aggs))

    if allres:
        print("\n(balanced accuracy; macro-F1 and accuracy are in the JSON)")
        best = {}
        for d, rows in allres.items():
            b = max(rows, key=lambda h: rows[h]["balanced_accuracy"])
            best[b] = best.get(b, 0) + 1
        print("best aggregation per run:", ", ".join(f"{k} x{v}" for k, v in
                                                     sorted(best.items(), key=lambda kv: -kv[1])))
    for d, err in skipped:
        print(f"SKIPPED {d}: {err}")

    if a.json_out and allres:
        with open(a.json_out, "w") as fh:
            json.dump(allres, fh, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
