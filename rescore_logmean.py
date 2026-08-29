#!/usr/bin/env python3
"""
Re-scores the Table 1 and Table 3 EEG runs under logmean clip pooling.

The plan-#8 sweep found logmean beats the default mean pooling on 7 of 8 configurations
(+0.028 macro-F1 at the winning configuration), and the Table 2 backbone sweep is being
run under logmean. Tables 1 and 3 currently report the GRU under mean pooling, so the
same model appears with different numbers in different tables. This re-scores them from
each run's saved val_window_preds.npz -- no retraining, and it reuses train_pooled_eeg's
own aggregate_clip() so the pooling math is identical to the sweep's.

Emits every number the two tables and fig:stability need.
"""
import argparse, glob, json, os, statistics as st, sys

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
                             accuracy_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pooled_eeg import aggregate_clip


def score(d, how="logmean", topk_frac=0.25, temp=1.0):
    f = os.path.join(d, "val_window_preds.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    P, cid, clips, y = z["probs"], z["cid"], z["clips"], z["y"]
    cp = aggregate_clip(P, cid, list(clips), how, topk_frac, temp)
    pred = cp.argmax(1)
    auc = (roc_auc_score(y, cp[:, 1]) if cp.shape[1] == 2
           else roc_auc_score(y, cp, multi_class="ovr", average="macro"))
    return dict(
        P=precision_score(y, pred, average="macro", zero_division=0),
        R=recall_score(y, pred, average="macro", zero_division=0),
        F1=f1_score(y, pred, average="macro", zero_division=0),
        AUC=auc,
        MCC=matthews_corrcoef(y, pred),
        bal=balanced_accuracy_score(y, pred),
        acc=accuracy_score(y, pred),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="logmean")
    a = ap.parse_args()

    out = {"agg": a.agg, "table1": {}, "table3": {}}

    print("=" * 76)
    print(f"TABLE 1 (session-disjoint) -- EEG GRU under {a.agg} vs mean")
    print("=" * 76)
    print(f"{'task':10s} {'':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s} {'MCC':>7s}")
    for tag, name in (("bin", "detection"), ("g3", "3-class"), ("g5", "5-class")):
        d = f"output/v3_seed_runs/eeg_{tag}_s42"
        new = score(d, a.agg)
        old = score(d, "mean")
        if not new:
            print(f"  {name}: no window preds"); continue
        print(f"{name:10s} {'mean':>6s} {old['P']:7.3f} {old['R']:7.3f} {old['F1']:7.3f} "
              f"{old['AUC']:7.3f} {old['MCC']:7.3f}")
        print(f"{'':10s} {a.agg:>6s} {new['P']:7.3f} {new['R']:7.3f} {new['F1']:7.3f} "
              f"{new['AUC']:7.3f} {new['MCC']:7.3f}")
        # five-seed macro-F1 under the new rule
        f1s = []
        for p in sorted(glob.glob(f"output/v3_seed_runs/eeg_{tag}_s*")):
            s = score(p, a.agg)
            if s:
                f1s.append(s["F1"])
        m = st.mean(f1s); sd = st.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"{'':10s} {'5-seed':>6s} F1 = {m:.4f} +/- {sd:.4f}  (n={len(f1s)})\n")
        out["table1"][tag] = {"seed42": new, "mean_pool_seed42": old,
                              "five_seed_f1": [m, sd, len(f1s)]}

    print("=" * 76)
    print(f"TABLE 3 (subject-disjoint, 5 folds) -- EEG under {a.agg}")
    print("=" * 76)
    for tag, name in (("bin", "detection"), ("g3", "3-class"), ("g5", "5-class")):
        rows = []
        for f in range(5):
            d = f"output/v3_subject_cv/eeg_{tag}_fold{f}"
            s = score(d, a.agg)
            if s:
                rows.append(s)
        if not rows:
            print(f"  {name}: none"); continue
        def ms(k):
            v = [r[k] for r in rows]
            return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        f1m, f1s_ = ms("F1"); aum, aus = ms("AUC"); mcm, mcs = ms("MCC")
        print(f"  {name:10s} n={len(rows)}  F1 {f1m:.4f}+/-{f1s_:.4f}   "
              f"AUROC {aum:.4f}+/-{aus:.4f}   MCC {mcm:.4f}+/-{mcs:.4f}")
        perfold = " ".join(f"{r['F1']:.3f}" for r in rows)
        print(f"             per-fold F1: {perfold}")
        out["table3"][tag] = {"f1": [f1m, f1s_], "auc": [aum, aus], "mcc": [mcm, mcs],
                              "per_fold_f1": [r["F1"] for r in rows]}

    json.dump(out, open(f"tex_wacv/rescore_{a.agg}.json", "w"), indent=2)
    print(f"\nwrote tex_wacv/rescore_{a.agg}.json")


if __name__ == "__main__":
    main()
