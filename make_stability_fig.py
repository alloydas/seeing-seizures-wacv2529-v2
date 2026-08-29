#!/usr/bin/env python3
"""
Regenerates fig/fig_stability.pdf -- cross-subject stability of detection macro-F1
under subject-disjoint 5-fold CV.

No generator for this figure existed on disk; the published version was produced
ad hoc in an earlier session. This reproduces its design (open circles = folds,
diamond = mean, whiskers and shaded band = +/-1 sigma) from the fold results, so it
can be regenerated whenever the folds change.

Sources:
  video  output/abl_subj/vid_fold{0..4}/results.json          (unaffected by the
                                                               clip-EDF defect)
  EEG    output/v3_subject_cv/eeg_bin_fold{0..4}/results.json  (repaired clips)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VID = "output/abl_subj/vid_fold{f}/results.json"
EEG = "output/v3_subject_cv/eeg_bin_fold{f}/results.json"


def folds(tmpl, n=5, agg=None):
    """agg=None -> the run's own results.json (mean pooling, video).
    agg='logmean' -> re-score the saved window predictions (EEG)."""
    out = []
    for f in range(n):
        p = tmpl.format(f=f)
        if agg:
            import sys, os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from rescore_logmean import score
            s = score(_os.path.dirname(p), agg)
            if s: out.append(s["F1"])
        elif os.path.exists(p):
            out.append(json.load(open(p))["macro_f1"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tex_wacv/fig/fig_stability.pdf")
    ap.add_argument("--require", type=int, default=5,
                    help="refuse to write unless both modalities have this many folds")
    a = ap.parse_args()

    v, e = folds(VID), folds(EEG, agg="logmean")
    print(f"video folds n={len(v)}: {np.round(v,4)}")
    print(f"EEG   folds n={len(e)}: {np.round(e,4)}")
    if len(v) < a.require or len(e) < a.require:
        raise SystemExit(f"refusing to write: need {a.require} folds each, "
                         f"have {len(v)} video / {len(e)} EEG")

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(3.9, 1.95))
    rng = np.random.default_rng(0)

    for x, vals, col, name in ((0, v, "#3B7EA1", "Video"), (1, e, "#E08A1E", "EEG")):
        m, s = vals.mean(), vals.std(ddof=1)
        ax.add_patch(plt.Rectangle((x - 0.30, m - s), 0.60, 2 * s,
                                   color=col, alpha=0.16, lw=0))
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x + jitter, vals, s=16, facecolors="none",
                   edgecolors=col, linewidths=0.9, zorder=3)
        ax.errorbar(x, m, yerr=s, fmt="D", color=col, markersize=4.5,
                    capsize=3.5, elinewidth=1.3, capthick=1.3, zorder=4,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.text(x, m + s + 0.012, f"{m:.3f} $\\pm$ {s:.3f}", ha="center",
                fontsize=8, color=col)

    lo = min(v.min(), e.min()); hi = max(v.max(), e.max())
    pad = max(0.05, (hi - lo) * 0.45)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Video", "EEG"])
    ax.set_ylabel("Detection macro-F1 (per fold)", fontsize=8)
    ax.grid(axis="y", color="#cccccc", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")

    ratio = e.std(ddof=1) / v.std(ddof=1)
    print(f"\nvideo {v.mean():.4f} +/- {v.std(ddof=1):.4f}  range [{v.min():.3f}, {v.max():.3f}]")
    print(f"EEG   {e.mean():.4f} +/- {e.std(ddof=1):.4f}  range [{e.min():.3f}, {e.max():.3f}]")
    print(f"sigma ratio EEG/video = {ratio:.2f}x")


if __name__ == "__main__":
    main()
