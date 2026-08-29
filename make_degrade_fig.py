#!/usr/bin/env python3
"""
Regenerates fig/fig_degrade.pdf -- macro-F1 against task granularity for both modalities.

No generator existed on disk; the published version was made ad hoc and still carried
the pre-repair, mean-pooled, seed-42 numbers. This reads tex_wacv/tab1_meansd.json
(written by make_tab1_meansd.py) so the figure and Table 1 cannot drift apart again, and
adds the +/-1 std band that Table 1 now reports.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["2-class\n(detection)", "3-class", "5-class"]
ORDER = ["bin", "g3", "g5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tex_wacv/tab1_meansd.json")
    ap.add_argument("--out", default="tex_wacv/fig/fig_degrade.pdf")
    a = ap.parse_args()

    d = json.load(open(a.src))
    x = np.arange(3)
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(3.9, 2.1))

    for mod, col, ls, mk in (("Video", "#1f6fb4", "-", "o"), ("EEG", "#e08a1e", "--", "x")):
        m = np.array([d[t][mod]["F1"][0] for t in ORDER])
        s = np.array([d[t][mod]["F1"][1] for t in ORDER])
        ax.plot(x, m, ls, color=col, marker=mk, markersize=4.5, lw=1.4, label=mod, zorder=3)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.16, lw=0)
        for xi, mi in zip(x, m):
            off = 0.028 if mod == "Video" else -0.045
            ax.annotate(f"{mi:.3f}", (xi, mi), textcoords="offset points",
                        xytext=(0, 9 if mod == "Video" else -13),
                        ha="center", fontsize=7.5, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Task granularity", fontsize=8)
    ax.set_ylabel("Macro-F1", fontsize=8)
    ax.set_ylim(0.45, 1.06)
    ax.set_xlim(-0.35, 2.35)
    ax.grid(axis="y", color="#cccccc", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")
    for t, lab in zip(ORDER, ["detection", "3-class", "5-class"]):
        v = d[t]["Video"]["F1"]; e = d[t]["EEG"]["F1"]
        print(f"  {lab:10s} video {v[0]:.3f}±{v[1]:.3f}   EEG {e[0]:.3f}±{e[1]:.3f}   "
              f"gap {v[0]-e[0]:+.3f}")


if __name__ == "__main__":
    main()
