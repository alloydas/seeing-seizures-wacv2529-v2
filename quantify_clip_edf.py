#!/usr/bin/env python3
"""Quantify the degenerate-clip-EDF defect across the whole cut corpus.

A 120-clip sample found 64% of clip EDFs carrying a near-constant biopotential trace
(median 4 unique values over a whole clip), while the RAW session EDFs those clips were
cut from are healthy (1800-4600 unique values per 60 s). That points at the cutting
pipeline rather than the recordings.

This scans every clip and reports:
  * overall rate of flat / degenerate / normal clips
  * the breakdown per subject and per session, to separate "a few bad sessions" from
    "a systematic cutting bug"
  * THE CRITICAL CROSS-TAB: degeneracy vs. clip LABEL. If seizure and non-seizure clips
    are degenerate at different rates, then data quality is correlated with the target
    and the EEG models can score above chance by detecting the artifact rather than the
    physiology -- which would make every EEG number in the paper uninterpretable.

Reads only the picked biopotential channel of each clip. Parallel over processes;
mne is the bottleneck, not the signal maths.
"""
import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")


def one(path):
    import mne
    mne.set_log_level("ERROR")
    try:
        r = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        names = r.ch_names
        pick = [c for c in names
                if ("EEG" in c.upper() or "ECG" in c.upper()) and "FIR" not in c.upper()]
        if not pick:
            return None
        x = r.get_data(picks=[names.index(pick[0])])[0]
        parts = path.split(os.sep)
        subj = [p for p in parts if p.startswith("Data_RN")][0].split("_")[1]
        sess = parts[-3]
        clip = parts[-2]
        return dict(subj=subj, sess=sess,
                    seizure=clip.startswith("seizure_"),
                    n_uniq=int(len(np.unique(x))),
                    std=float(x.std()),
                    dur=float(len(x) / r.info["sfreq"]),
                    label=pick[0])
    except Exception:
        return None


def main():
    files = sorted(glob.glob("data/Data_RN*_cropped/*/*/eeg.edf"))
    if len(sys.argv) > 1:
        files = files[::max(1, len(files) // int(sys.argv[1]))]
    print(f"scanning {len(files)} clip EDFs ...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, f) for f in files]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if r:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    def kind(r):
        if r["std"] == 0:
            return "flat"
        return "degenerate" if r["n_uniq"] < 100 else "normal"

    for r in rows:
        r["kind"] = kind(r)

    n = len(rows)
    print(f"\n=== overall ({n} clips read) ===")
    for k in ("flat", "degenerate", "normal"):
        c = sum(1 for r in rows if r["kind"] == k)
        print(f"  {k:11s} {c:6d}  {100*c/n:5.1f}%")
    uq = np.array([r["n_uniq"] for r in rows])
    print(f"  unique values: median={int(np.median(uq))}  "
          f"p10={int(np.percentile(uq,10))}  p90={int(np.percentile(uq,90))}")

    print("\n=== CRITICAL: degeneracy vs clip label ===")
    for lab, name in ((True, "seizure"), (False, "non-seizure")):
        sub = [r for r in rows if r["seizure"] == lab]
        if not sub:
            continue
        bad = sum(1 for r in sub if r["kind"] != "normal")
        print(f"  {name:12s} n={len(sub):6d}  degenerate/flat = {bad:6d}  ({100*bad/len(sub):5.1f}%)")
    sz = [r for r in rows if r["seizure"]]
    nz = [r for r in rows if not r["seizure"]]
    if sz and nz:
        a = sum(1 for r in sz if r["kind"] != "normal") / len(sz)
        b = sum(1 for r in nz if r["kind"] != "normal") / len(nz)
        print(f"  -> difference: {100*abs(a-b):.1f} percentage points "
              f"({'CONFOUNDED - data quality tracks the label' if abs(a-b) > 0.05 else 'balanced, no label confound'})")

    print("\n=== per subject ===")
    bysub = defaultdict(list)
    for r in rows:
        bysub[r["subj"]].append(r)
    print(f"  {'subj':8s} {'clips':>6s} {'bad%':>6s} {'median uniq':>12s}")
    for s in sorted(bysub, key=lambda s: -sum(1 for r in bysub[s] if r["kind"] != "normal") / len(bysub[s])):
        v = bysub[s]
        bad = sum(1 for r in v if r["kind"] != "normal")
        print(f"  {s:8s} {len(v):6d} {100*bad/len(v):5.1f}% "
              f"{int(np.median([r['n_uniq'] for r in v])):12d}")

    bysess = defaultdict(list)
    for r in rows:
        bysess[(r["subj"], r["sess"])].append(r)
    allbad = sum(1 for k, v in bysess.items()
                 if all(r["kind"] != "normal" for r in v))
    print(f"\n=== per session ===")
    print(f"  sessions: {len(bysess)}   entirely degenerate: {allbad} "
          f"({100*allbad/len(bysess):.1f}%)")

    json.dump(rows, open("output/clip_edf_quality.json", "w"))
    print("\nwrote output/clip_edf_quality.json")


if __name__ == "__main__":
    main()
