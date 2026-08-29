#!/usr/bin/env python3
"""Re-cut ONLY the eeg.edf of existing clip directories, with the fixed EDF writer.

Why EDF-only: the video, info.txt and directory layout of every clip are correct. The
sole defect is that mne.export.export_raw wrote a shared physical range across channels,
collapsing the biopotential to 2-4 distinct values in 69% of clips (24,214 scanned,
2026-08-12). Re-extracting just the EDF avoids redoing the expensive video cut.

Each clip's info.txt already carries everything needed:
    Clip start : 2023-10-12 16:25:55.328000
    Clip end   : 2023-10-12 16:26:51.873000
    EDF file   : RN197-10-12-2023 (2).edf

The original is preserved as eeg.edf.bad on first rewrite, so this is reversible and
re-runnable; clips whose EDF is already healthy are skipped unless --force.

SOURCE AVAILABILITY: all 601 sessions across all 20 subjects are reachable on
back_up_1 -- RoomD under EEG/Data/, RoomC under EEG/processed/.

Usage:
    python3 recut_clip_edfs.py --dry-run
    python3 recut_clip_edfs.py --workers 6
    python3 recut_clip_edfs.py --subjects RN219 RN210
"""
import argparse
import glob
import os
import re
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

# RoomD sources live under .../EEG/Data/<SUBJ>/<session>/, RoomC under
# .../EEG/processed/ -- partly as <SUBJ>/<session>/ and partly in date-named
# directories holding several subjects' EDFs, so the lookup tries both shapes.
RAW_ROOTS = ["/path/to/archive",
             "/path/to/archive"]
DT_FMTS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def parse_info(path):
    txt = open(path).read()
    def get(k):
        m = re.search(rf"{k}\s*:\s*(.+)", txt)
        return m.group(1).strip() if m else None
    def dt(s):
        for f in DT_FMTS:
            try:
                return datetime.strptime(s, f)
            except (ValueError, TypeError):
                pass
        return None
    return dict(start=dt(get("Clip start")), end=dt(get("Clip end")),
                edf=get("EDF file"))


_SRC_CACHE = {}


def find_source(subj, fname):
    for r in RAW_ROOTS:
        for pat in (f"{r}/{subj}/*/{glob.escape(fname)}",   # RoomD, and RoomC subject dirs
                    f"{r}/*/{glob.escape(fname)}",          # RoomC date-named dirs
                    f"{r}/{glob.escape(fname)}"):
            hits = glob.glob(pat)
            if hits:
                return hits[0]
    return None


def edf_is_healthy(path, min_uniq=100):
    import mne
    mne.set_log_level("ERROR")
    try:
        r = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        pick = [c for c in r.ch_names
                if ("EEG" in c.upper() or "ECG" in c.upper()) and "FIR" not in c.upper()]
        if not pick:
            return False
        x = r.get_data(picks=[r.ch_names.index(pick[0])])[0]
        return len(np.unique(x)) >= min_uniq
    except Exception:
        return False


def one(clip_dir, force):
    import mne
    mne.set_log_level("ERROR")
    from edf_clip_writer import write_edf_clip
    out = os.path.join(clip_dir, "eeg.edf")
    info = os.path.join(clip_dir, "info.txt")
    if not os.path.exists(info):
        return ("no_info", clip_dir)
    if os.path.exists(out) and not force and edf_is_healthy(out):
        return ("already_ok", clip_dir)
    meta = parse_info(info)
    if not (meta["start"] and meta["end"] and meta["edf"]):
        return ("bad_info", clip_dir)
    subj = [p for p in clip_dir.split(os.sep) if p.startswith("Data_RN")][0].split("_")[1]
    src = find_source(subj, meta["edf"])
    if not src:
        return ("no_source", clip_dir)
    try:
        raw = mne.io.read_raw_edf(src, preload=False, verbose="ERROR")
        edf_start = raw.info["meas_date"]
        if edf_start is None:
            return ("no_meas_date", clip_dir)
        edf_start = edf_start.replace(tzinfo=None)
        t0 = max(0.0, (meta["start"] - edf_start).total_seconds())
        t1 = min(raw.n_times / raw.info["sfreq"], (meta["end"] - edf_start).total_seconds())
        if t1 - t0 < 1.0:
            return ("bad_window", clip_dir)
        raw.crop(tmin=t0, tmax=t1)
        raw.load_data(verbose="ERROR")
        # EDF caps labels at 16 chars. Naive truncation collides when two channels
        # share a 16-char prefix (e.g. 'ECG [FIR-HP: 5Hz]' vs 'ECG [FIR-HP: 5.0Hz]'),
        # and mne raises "New channel names are not unique" -- 524 clips failed that
        # way on the first pass. Mirrors the collision handling in cut_seizure_clips.py.
        rename, used = {}, set(raw.ch_names)
        for ch in raw.ch_names:
            if len(ch) > 16:
                base, suffix, cand = ch[:14], 0, ch[:16]
                while cand in used and cand != ch:
                    cand = f"{base}_{suffix}"
                    suffix += 1
                rename[ch] = cand
                used.discard(ch)
                used.add(cand)
        if rename:
            raw.rename_channels(rename)
        if os.path.exists(out) and not os.path.exists(out + ".bad"):
            shutil.move(out, out + ".bad")
        write_edf_clip(raw, out)
        raw.close()
        return ("rewritten", clip_dir)
    except Exception as e:
        return (f"error:{type(e).__name__}", clip_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dirs = sorted(glob.glob("data/Data_RN*_cropped/*/*/"))
    if a.subjects:
        dirs = [d for d in dirs if any(f"Data_{s}_cropped" in d for s in a.subjects)]
    if a.limit:
        dirs = dirs[:a.limit]
    print(f"{len(dirs)} clip directories in scope", flush=True)

    if a.dry_run:
        from collections import Counter
        c = Counter()
        for d in dirs[:400]:
            info = os.path.join(d, "info.txt")
            if not os.path.exists(info):
                c["no_info"] += 1; continue
            m = parse_info(info)
            subj = [p for p in d.split(os.sep) if p.startswith("Data_RN")][0].split("_")[1]
            c["source_found" if (m["edf"] and find_source(subj, m["edf"])) else "no_source"] += 1
        print("dry run over first 400:", dict(c))
        return

    from collections import Counter
    tally = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, d, a.force) for d in dirs]
        for i, f in enumerate(as_completed(futs)):
            st, _ = f.result()
            tally[st] += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(dirs)}  {dict(tally)}", flush=True)
    print("\n=== done ===")
    for k, v in tally.most_common():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
