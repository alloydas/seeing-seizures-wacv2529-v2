#!/usr/bin/env python3
"""Overlay video and EEG seizure confidence for one session on a single wall clock.

Reads the trace.npz files written by session_eval.py. Because every trace is stored in
absolute epoch seconds -- the whole point of session_eval's align step -- a video sweep
and an EEG sweep of the same session can be plotted together even when they were run
separately, at different window sizes and strides.

  python3 make_session_timeline.py --trace output/session_eval_RN208/trace.npz \
                                   --trace output/session_eval_RN208_eeg/trace.npz \
                                   --out output/session_RN208_timeline.pdf
"""
import argparse, os
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load(paths):
    """Merge any number of traces. Later files fill modalities the earlier ones lack."""
    vt = vp = et = ep = None
    ann = np.empty((0, 2))
    for p in paths:
        z = np.load(p)
        if z["t_video"].size and vt is None:
            vt, vp = z["t_video"], z["p_video"]
        if z["t_eeg"].size and et is None:
            et, ep = z["t_eeg"], z["p_eeg"]
        if z["ann"].size and ann.size == 0:
            ann = z["ann"].reshape(-1, 2)
    return vt, vp, et, ep, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True,
                    help="trace.npz from session_eval.py; repeat to merge video + EEG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=0,
                    help="odd window for a centred moving average on the traces")
    a = ap.parse_args()

    vt, vp, et, ep, ann = load(a.trace)
    if vt is None and et is None:
        raise SystemExit("[fatal] no video and no EEG samples in the given traces")

    def sm(y):
        if a.smooth and a.smooth > 1 and y is not None and y.size > a.smooth:
            k = np.ones(a.smooth) / a.smooth
            return np.convolve(y, k, mode="same")
        return y

    # one shared origin so both modalities sit on the same hours-from-start axis
    t0 = min([t[0] for t in (vt, et) if t is not None and t.size])
    start = datetime.fromtimestamp(t0)
    h = lambda t: (t - t0) / 3600.0

    n = sum(x is not None for x in (vt, et))
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    i = 0
    for t, p, name, col in ((vt, vp, "Video (SlowFast-R50)", "#1f77b4"),
                            (et, ep, "EEG (GRU)", "#d62728")):
        if t is None:
            continue
        ax = axes[i]; i += 1
        ax.plot(h(t), sm(p), lw=0.6, color=col, label=f"{name}  p(seizure)")
        ax.axhline(a.thr, ls="--", lw=0.8, color="0.4")
        for k, (s, e) in enumerate(ann):
            ax.axvspan(h(s), h(e), color="green", alpha=0.30, lw=0,
                       label="annotated seizure" if k == 0 else None)
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel("p(seizure)")
        ax.set_title(name, loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].set_xlabel(f"hours from {start:%Y-%m-%d %H:%M:%S} (local)")
    if a.title:
        fig.suptitle(a.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

    # a short numeric summary, so the figure is never the only artifact
    for t, p, name in ((vt, vp, "video"), (et, ep, "eeg")):
        if t is None:
            continue
        inside = np.zeros(t.shape, bool)
        for s, e in ann:
            inside |= (t >= s) & (t <= e)
        span = (t.max() - t.min()) / 3600.0
        print(f"  {name:6s} n={t.size:6d}  span={span:6.2f} h  "
              f"mean p={p.mean():.3f}  p>={a.thr}: {(p >= a.thr).mean()*100:5.2f}% of windows  "
              f"| inside GT: n={inside.sum()} mean p={p[inside].mean():.3f}" if inside.any()
              else f"  {name:6s} n={t.size:6d}  span={span:6.2f} h  mean p={p.mean():.3f}  "
                   f"p>={a.thr}: {(p >= a.thr).mean()*100:5.2f}% of windows | no GT in range")


if __name__ == "__main__":
    main()
