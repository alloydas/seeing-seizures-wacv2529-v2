#!/usr/bin/env python3
"""
Tests the mechanism proposed for video's failures: that it misses seizures whose motor
expression is subtle. Measures per-clip motion energy (mean absolute frame difference on
a downscaled grayscale sample) and relates it to whether each model got the clip right.

Sparse-samples frames rather than decoding whole clips, so it stays cheap on a loaded box.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NAMES = ["non-seizure", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]


def motion_energy(clip_dir, n_frames=24, size=96):
    """Mean |frame difference| over n_frames evenly spaced samples."""
    f = os.path.join(clip_dir, "video.mp4")
    if not os.path.exists(f):
        return clip_dir, np.nan
    cap = cv2.VideoCapture(f)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 4:
        cap.release()
        return clip_dir, np.nan
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    prev, diffs = None, []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return clip_dir, (float(np.mean(diffs)) if diffs else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=900)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tex_wacv/motion_vs_error.json")
    a = ap.parse_args()

    import train_pooled as tp
    items = tp.discover()
    tr, va, valsess, seed = tp.split_sessions(items, 49)

    zv = np.load("output/classifier_pooled20_g5/val_preds.npz", allow_pickle=True)
    ze = np.load("output/v3_seed_runs/eeg_g5_s42/val_clip_preds.npz", allow_pickle=True)
    vpath = np.array([os.path.normpath(os.path.dirname(va[i][0])) for i in zv["idx"]])
    epath = np.array([os.path.normpath(p) for p in ze["path"]])

    vkey = {p: i for i, p in enumerate(vpath)}
    rows = [(vkey[p], j) for j, p in enumerate(epath) if p in vkey]
    vi = np.array([r[0] for r in rows]); ei = np.array([r[1] for r in rows])
    paths = epath[ei]
    y = ze["y"][ei]
    vok = zv["pred"][vi] == zv["y"][vi]
    eok = ze["pred"][ei] == ze["y"][ei]
    print(f"{len(rows)} shared clips")

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(rows), size=min(a.sample, len(rows)), replace=False)
    todo = [paths[i] for i in sel]
    print(f"measuring motion on {len(todo)} clips with {a.workers} workers")

    energy = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (p, e) in enumerate(ex.map(motion_energy, todo, chunksize=4)):
            energy[p] = e
            if (k + 1) % 150 == 0:
                print(f"  {k+1}/{len(todo)}", flush=True)

    m = np.array([energy.get(paths[i], np.nan) for i in sel])
    ok = ~np.isnan(m)
    m, ys = m[ok], y[sel][ok]
    v, e = vok[sel][ok], eok[sel][ok]
    print(f"usable: {ok.sum()}")

    res = {"n": int(ok.sum()), "by_class": {}, "overall": {}}
    res["overall"] = {
        "median_motion_video_right": float(np.median(m[v])) if v.any() else None,
        "median_motion_video_wrong": float(np.median(m[~v])) if (~v).any() else None,
        "median_motion_eeg_right": float(np.median(m[e])) if e.any() else None,
        "median_motion_eeg_wrong": float(np.median(m[~e])) if (~e).any() else None,
    }
    print("\nmotion energy (mean |frame diff|), median:")
    print(f"  video right {res['overall']['median_motion_video_right']:.3f}   "
          f"video wrong {res['overall']['median_motion_video_wrong']:.3f}")
    print(f"  EEG   right {res['overall']['median_motion_eeg_right']:.3f}   "
          f"EEG   wrong {res['overall']['median_motion_eeg_wrong']:.3f}")

    print("\nper true class: median motion, and video recall in low/high motion halves")
    print(f"{'class':13s} {'n':>4s} {'motion':>8s} {'vid lo':>7s} {'vid hi':>7s} {'eeg lo':>7s} {'eeg hi':>7s}")
    for c in range(5):
        mc = ys == c
        if mc.sum() < 12:
            continue
        med = np.median(m[mc])
        lo, hi = mc & (m <= med), mc & (m > med)
        row = {"n": int(mc.sum()), "median_motion": float(med),
               "video_recall_low": float(v[lo].mean()) if lo.any() else None,
               "video_recall_high": float(v[hi].mean()) if hi.any() else None,
               "eeg_recall_low": float(e[lo].mean()) if lo.any() else None,
               "eeg_recall_high": float(e[hi].mean()) if hi.any() else None}
        res["by_class"][NAMES[c]] = row
        print(f"{NAMES[c]:13s} {row['n']:4d} {med:8.3f} "
              f"{row['video_recall_low']:7.3f} {row['video_recall_high']:7.3f} "
              f"{row['eeg_recall_low']:7.3f} {row['eeg_recall_high']:7.3f}")

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
