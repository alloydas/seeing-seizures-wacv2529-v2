"""
Classical-ML EEG baselines (Random Forest, XGBoost) on the pooled 20-subject
cache, using the SAME per-window z-score, session-disjoint split (seed 49) and
window->clip mean-pool evaluation as the deep EEG models (train_pooled_eeg.py),
so they slot directly into the tab:eegarch comparison. CPU-only.

  python train_pooled_eeg_classical.py --arch rf  --group2 --output output/eeg_rf_bin
  python train_pooled_eeg_classical.py --arch xgb --group3 --output output/eeg_xgb_g3
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import train_pooled_eeg as TE

FS = 125.0
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 60)]  # delta theta alpha beta gamma


def extract_features(segs):
    """Per-window EEG features (band powers + time-domain stats) -- the standard
    representation for classical ML on EEG, far more tree-friendly than raw
    amplitude. segs: (N, L) -> (N, 5 bands + ~9 stats)."""
    N, L = segs.shape
    freqs = np.fft.rfftfreq(L, 1.0 / FS)
    psd = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    total = psd.sum(1, keepdims=True) + 1e-8
    bp = np.stack([psd[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in BANDS], 1)
    bp = bp / total                                              # relative band power
    diff = np.diff(segs, axis=1)
    stats = np.stack([
        segs.mean(1), segs.std(1), np.sqrt((segs ** 2).mean(1)),   # mean, std, RMS
        np.abs(diff).sum(1),                                       # line length
        (np.abs(diff) > diff.std(1, keepdims=True)).sum(1),        # ~activity
        ((segs[:, :-1] * segs[:, 1:]) < 0).sum(1),                 # zero crossings
        segs.min(1), segs.max(1), np.ptp(segs, axis=1),            # min, max, range
    ], 1)
    return np.concatenate([bp, stats], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["rf", "xgb"], required=True)
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--group3", action="store_true")
    ap.add_argument("--group2", action="store_true")
    ap.add_argument("--cache", default="stage_segments_pooled.npz")
    ap.add_argument("--max_train", type=int, default=0, help="subsample train windows (0=all)")
    ap.add_argument("--raw", action="store_true", help="use raw windows instead of features")
    ap.add_argument("--seed", type=int, default=42,
                    help="init/data-order seed (for seed-variance runs)")
    ap.add_argument("--output", default="output/eeg_classical_out")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.cache, allow_pickle=True)
    segs, wlab, cid, clab, csess = d["segs"], d["wlab"], d["clip_id"], d["clip_lab"], d["clip_sess"]
    if a.group2:
        wlab, clab = TE.GROUP2[wlab], TE.GROUP2[clab]; TE.NAMES, TE.NC = TE.NAMES2, 2; print("*** classical EEG 2-class ***")
    elif a.group3:
        wlab, clab = TE.GROUP3[wlab], TE.GROUP3[clab]; TE.NAMES, TE.NC = TE.NAMES3, 3; print("*** classical EEG 3-class ***")
    else:
        TE.NAMES, TE.NC = TE.NAMES5, 5; print("*** classical EEG 5-class ***")
    NC = TE.NC

    # per-window z-score (identical to the deep models)
    m = segs.mean(1, keepdims=True); sd = segs.std(1, keepdims=True); sd[sd == 0] = 1.0
    segs = ((segs - m) / sd).astype(np.float32)
    if not a.raw:
        segs = extract_features(segs); print(f"features: {segs.shape[1]}-d per window (band powers + stats)")

    val_sess, seed = TE.split_sessions(list(csess), [int(x) for x in clab], a.split_seed)
    clip_is_val = np.array([s in val_sess for s in csess])
    win_is_val = clip_is_val[cid]

    Xtr, ytr = segs[~win_is_val], wlab[~win_is_val]
    Xva = segs[win_is_val]
    if a.max_train and len(Xtr) > a.max_train:
        idx = np.random.choice(len(Xtr), a.max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"subjects: {len(set(s.split('/')[0] for s in csess))}  windows: train={len(Xtr)} "
          f"val={len(Xva)}  seed={seed}", flush=True)
    print(f"  train window labels: {np.bincount(ytr, minlength=NC).tolist()}", flush=True)

    freq = np.array([max(int((ytr == i).sum()), 1) for i in range(NC)], float)
    w = 1.0 / freq; w = w / w.mean()

    if a.arch == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced",
                                     random_state=a.seed)
        clf.fit(Xtr, ytr)
    else:
        from xgboost import XGBClassifier
        sw = w[ytr.astype(int)]
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            tree_method="hist", n_jobs=-1, random_state=a.seed)
        clf.fit(Xtr, ytr, sample_weight=sw)
    print("trained; predicting val windows...", flush=True)

    P = clf.predict_proba(Xva)
    if P.shape[1] < NC:                       # a class missing from train -> pad
        full = np.zeros((len(P), NC), np.float32); full[:, clf.classes_.astype(int)] = P; P = full

    cid_va = cid[win_is_val]; val_clips = np.unique(cid_va)
    cy = np.array([int(clab[c]) for c in val_clips])
    csub = np.array([csess[c].split("/")[0] for c in val_clips])
    cp = np.stack([P[cid_va == c].mean(0) for c in val_clips]); cpred = cp.argmax(1)

    np.savez(out / "val_clip_preds.npz", y=cy, probs=cp, pred=cpred, sub=csub)
    res = TE.report(cy, cpred, cp, f"CLASSICAL EEG {a.arch.upper()} ({len(set(csub))} subjects), clip level")
    res["per_subject"] = TE.per_subject(cy, cpred, csub)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
