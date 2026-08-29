#!/usr/bin/env python3
"""
Confusion matrices for the video and EEG models at all three granularities, plus a
paired analysis of *where* each modality fails and evidence for *why*.

Outputs (into tex_wacv/fig/ unless --outdir given):
    fig_confusion.pdf        2x3 grid of row-normalised confusion matrices
    fig_failure_modes.pdf    ordinal error structure, rarity effect, paired disagreement
    failure_analysis.json    every number quoted, for the paper

The video runs predate the `path=` patch in train_pooled.py, so their val_preds.npz
carries only `idx`. We rebuild the val list with the same discover()/split_sessions()
call the training run used and index it by `idx`; the reconstruction is verified by
checking that the recovered labels match the stored `y` exactly, and we refuse to use
it otherwise.
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASKS = {
    "bin": ("Detection", ["non-seizure", "seizure"]),
    "g3":  ("3-class",   ["non-seizure", "mild (S2-3)", "severe (S4-5)"]),
    "g5":  ("5-class",   ["non-seizure", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]),
}
VID = "output/classifier_pooled20_{t}/val_preds.npz"
EEG = "output/v3_seed_runs/eeg_{t}_s42/val_clip_preds.npz"


# ---------------------------------------------------------------- loading
def load(path, want_path=False):
    z = np.load(path, allow_pickle=True)
    d = {"y": z["y"], "pred": z["pred"], "probs": z["probs"]}
    for k in ("path", "sub", "idx"):
        if k in z:
            d[k] = z[k]
    return d


def recover_video_paths(task, d):
    """Rebuild the video val list and map idx -> clip dir. Verified against stored y."""
    import train_pooled as tp
    grp = {"bin": "group2", "g3": "group3", "g5": None}[task]
    tp.NC = {"bin": 2, "g3": 3, "g5": 5}[task]
    # discover() reads module-level flags for the grouping; set them the way main() does
    for flag, val in (("GROUP2", grp == "group2"), ("GROUP3", grp == "group3")):
        if hasattr(tp, flag):
            setattr(tp, flag, val)
    try:
        items = tp.discover()
        tr, va, valsess, seed = tp.split_sessions(items, 49)
    except Exception as e:                       # discovery signature differs -> give up
        return None, f"discovery failed: {e}"
    if len(va) == 0:
        return None, "empty val list"
    idx = d.get("idx")
    if idx is None or idx.max() >= len(va):
        return None, f"idx out of range (max {None if idx is None else idx.max()} vs {len(va)})"
    # va stores the .mp4; the EEG side keys on the clip directory that contains it
    paths = np.array([os.path.dirname(va[i][0]) for i in idx])
    labels = np.array([va[i][1] for i in idx])
    if not np.array_equal(labels, d["y"]):
        return None, "recovered labels do not match stored y"
    return paths, "ok"


# ---------------------------------------------------------------- figures
def confusion_grid(data, out):
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2b6cb0"])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for r, mod in enumerate(("Video", "EEG")):
        for c, t in enumerate(("bin", "g3", "g5")):
            ax = axes[r, c]
            d = data[mod][t]
            names = TASKS[t][1]
            cm = confusion_matrix(d["y"], d["pred"], labels=range(len(names)))
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})",
                            ha="center", va="center", fontsize=7.5,
                            color="white" if cmn[i, j] > 0.55 else "#222")
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5)
            ax.set_yticklabels(names, fontsize=7.5)
            mf1 = f1_score(d["y"], d["pred"], average="macro")
            ax.set_title(f"{mod} — {TASKS[t][0]}  (macro-F1 {mf1:.3f})", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("true", fontsize=8.5)
            ax.set_xlabel("predicted", fontsize=8.5)
    fig.suptitle("Row-normalised confusion matrices (fraction of each true class; counts in parentheses)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def failure_modes(data, res, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # (a) ordinal error distance at 5-class -------------------------------
    ax = axes[0]
    w = 0.38
    for k, mod in enumerate(("Video", "EEG")):
        h = res["ordinal"][mod]["hist"]
        ax.bar(np.arange(len(h)) + (k - 0.5) * w, h, w,
               label=f"{mod} (mean |Δstage| {res['ordinal'][mod]['mean_dist']:.2f})",
               color=["#2b6cb0", "#c05621"][k])
    ax.set_xlabel("stage distance of the error  |true − predicted|")
    ax.set_ylabel("fraction of errors")
    ax.set_title("(a) How far wrong, at 5-class")
    ax.set_xticks(range(len(res["ordinal"]["Video"]["hist"])))
    ax.legend(fontsize=8)

    # (b) per-class recall vs support -------------------------------------
    ax = axes[1]
    names = TASKS["g5"][1]
    x = np.arange(len(names))
    for k, mod in enumerate(("Video", "EEG")):
        ax.bar(x + (k - 0.5) * w, res["per_class"][mod]["recall"], w,
               label=mod, color=["#2b6cb0", "#c05621"][k])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for i, n in enumerate(res["per_class"]["support"]):
        ax.text(i, 1.02, f"n={n}", ha="center", fontsize=7, color="#555")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("recall")
    ax.set_title("(b) Where the recall goes, at 5-class")
    ax.legend(fontsize=8)

    # (c) paired disagreement ---------------------------------------------
    ax = axes[2]
    p = res.get("paired")
    if p and p.get("n"):
        cats = ["both\nright", "video right\nEEG wrong", "EEG right\nvideo wrong", "both\nwrong"]
        vals = [p["both_right"], p["vid_only"], p["eeg_only"], p["both_wrong"]]
        ax.bar(cats, vals, color=["#2f855a", "#2b6cb0", "#c05621", "#822727"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}\n({100*v/p['n']:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(f"clips (n={p['n']} shared)")
        ax.set_title("(c) Paired outcomes on shared clips, 5-class")
        ax.set_ylim(0, max(vals) * 1.25)
    else:
        ax.text(0.5, 0.5, "paired analysis unavailable\n(video paths unrecoverable)",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- analysis
def ordinal_stats(d):
    y, pred = d["y"], d["pred"]
    err = y != pred
    dist = np.abs(y - pred)[err]
    maxd = 5
    hist = np.array([(dist == k).sum() for k in range(maxd)], dtype=float)
    hist = hist / max(hist.sum(), 1)
    return {"mean_dist": float(dist.mean()) if err.any() else 0.0,
            "hist": hist.tolist(), "n_err": int(err.sum()), "n": int(len(y))}


def per_class(d, ncls):
    y, pred = d["y"], d["pred"]
    rec, prec, sup = [], [], []
    for c in range(ncls):
        m = y == c
        sup.append(int(m.sum()))
        rec.append(float((pred[m] == c).mean()) if m.any() else 0.0)
        pm = pred == c
        prec.append(float((y[pm] == c).mean()) if pm.any() else 0.0)
    return {"recall": rec, "precision": prec, "support": sup}


def clip_meta(path):
    """Seizure duration and stage from the clip's info.txt."""
    f = os.path.join(path, "info.txt")
    if not os.path.exists(f):
        return {}
    out = {}
    for line in open(f, errors="ignore"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if "dur (s)" in k or k == "Duration (s)":
            try: out["seizure_dur"] = float(v)
            except ValueError: pass
        elif "label" in k.lower() and "Stage" in v:
            out["stage"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="tex_wacv/fig")
    ap.add_argument("--max_meta", type=int, default=4000,
                    help="cap on clips read for duration metadata")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    data = {"Video": {}, "EEG": {}}
    for t in TASKS:
        data["Video"][t] = load(VID.format(t=t))
        data["EEG"][t] = load(EEG.format(t=t))
        print(f"{t}: video n={len(data['Video'][t]['y'])}  eeg n={len(data['EEG'][t]['y'])}")

    res = {"ordinal": {}, "per_class": {}, "confusion": {}, "note": {}}

    # confusion matrices, every task
    for mod in ("Video", "EEG"):
        for t in TASKS:
            d = data[mod][t]
            n = len(TASKS[t][1])
            cm = confusion_matrix(d["y"], d["pred"], labels=range(n))
            res["confusion"].setdefault(mod, {})[t] = cm.tolist()
        res["ordinal"][mod] = ordinal_stats(data[mod]["g5"])
        res["per_class"][mod] = per_class(data[mod]["g5"], 5)
    res["per_class"]["support"] = res["per_class"]["Video"]["support"]

    confusion_grid(data, os.path.join(a.outdir, "fig_confusion.pdf"))
    print("wrote fig_confusion.pdf")

    # ---- paired analysis on shared clips --------------------------------
    vpaths, why = recover_video_paths("g5", data["Video"]["g5"])
    res["note"]["video_path_recovery"] = why
    print("video path recovery:", why)
    paired = None
    if vpaths is not None:
        ep = data["EEG"]["g5"]["path"]
        vkey = {os.path.normpath(p): i for i, p in enumerate(vpaths)}
        rows = [(vkey[os.path.normpath(p)], j) for j, p in enumerate(ep)
                if os.path.normpath(p) in vkey]
        if rows:
            vi = np.array([r[0] for r in rows]); ei = np.array([r[1] for r in rows])
            vok = data["Video"]["g5"]["pred"][vi] == data["Video"]["g5"]["y"][vi]
            eok = data["EEG"]["g5"]["pred"][ei] == data["EEG"]["g5"]["y"][ei]
            paired = {"n": int(len(rows)),
                      "both_right": int((vok & eok).sum()),
                      "vid_only": int((vok & ~eok).sum()),
                      "eeg_only": int((~vok & eok).sum()),
                      "both_wrong": int((~vok & ~eok).sum())}
            # what distinguishes the clips only video gets right?
            ytrue = data["EEG"]["g5"]["y"][ei]
            paired["by_true_class"] = {
                TASKS["g5"][1][c]: {
                    "n": int((ytrue == c).sum()),
                    "vid_only": int((vok & ~eok & (ytrue == c)).sum()),
                    "eeg_only": int((~vok & eok & (ytrue == c)).sum()),
                } for c in range(5)}
            # duration of the seizure each clip is matched to
            sub = ep[ei][:a.max_meta]
            durs = np.array([clip_meta(p).get("seizure_dur", np.nan) for p in sub])
            ok = ~np.isnan(durs)
            if ok.sum() > 50:
                vv, ee = vok[:len(durs)][ok], eok[:len(durs)][ok]
                paired["duration"] = {
                    "n": int(ok.sum()),
                    "median_all": float(np.median(durs[ok])),
                    "median_eeg_wrong": float(np.median(durs[ok][~ee])) if (~ee).any() else None,
                    "median_eeg_right": float(np.median(durs[ok][ee])) if ee.any() else None,
                    "median_vid_wrong": float(np.median(durs[ok][~vv])) if (~vv).any() else None,
                    "median_vid_right": float(np.median(durs[ok][vv])) if vv.any() else None,
                }
    res["paired"] = paired

    failure_modes(data, res, os.path.join(a.outdir, "fig_failure_modes.pdf"))
    print("wrote fig_failure_modes.pdf")

    with open(os.path.join(a.outdir, "..", "failure_analysis.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- console summary -------------------------------------------------
    print("\n" + "=" * 74)
    print("WHERE EACH MODALITY FAILS (5-class)")
    print("=" * 74)
    names = TASKS["g5"][1]
    print(f"{'class':14s} {'n':>5s} {'video rec':>10s} {'EEG rec':>9s} {'gap':>8s}")
    for i, nm in enumerate(names):
        v = res["per_class"]["Video"]["recall"][i]
        e = res["per_class"]["EEG"]["recall"][i]
        print(f"{nm:14s} {res['per_class']['support'][i]:5d} {v:10.3f} {e:9.3f} {v-e:+8.3f}")
    print(f"\nordinal error distance (mean |true-pred| over errors):")
    for mod in ("Video", "EEG"):
        o = res["ordinal"][mod]
        print(f"  {mod:6s} {o['mean_dist']:.2f}   ({o['n_err']}/{o['n']} wrong)")
    if paired:
        print(f"\npaired on {paired['n']} shared clips:")
        print(f"  both right {paired['both_right']}   video-only {paired['vid_only']}   "
              f"EEG-only {paired['eeg_only']}   both wrong {paired['both_wrong']}")
        if "duration" in paired:
            d = paired["duration"]
            print(f"\nmatched-seizure duration (n={d['n']}):")
            print(f"  EEG wrong  median {d['median_eeg_wrong']}s   EEG right  median {d['median_eeg_right']}s")
            print(f"  video wrong median {d['median_vid_wrong']}s   video right median {d['median_vid_right']}s")


if __name__ == "__main__":
    main()
