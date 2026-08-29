#!/usr/bin/env python3
"""Census: which animals were recorded with EEG and which with ECG.

Two independent lines of evidence per animal:

  1. LABEL   -- the DSI channel name in the EDF header ('EEG' vs 'ECG').
  2. SIGNAL  -- explicit R-peak detection, NOT the autocorrelation heuristic used in
                the first three attempts, all of which failed:
                  * a single 60 s window frequently landed on a flat or saturated
                    stretch (RN245 returned nan, RN215/RN229 returned p2p = 0);
                  * kurtosis spanned 0.49 to 6295 because a handful of disconnection
                    artifacts dominate the fourth moment, so it did not separate the
                    modalities at all;
                  * the autocorrelation argmax kept landing on the boundary of the lag
                    window (11 of 21 animals reported exactly 8.00 Hz), i.e. "no
                    periodicity found" was being reported as a measurement.

Method that replaces it:
  - sample up to N_WIN windows spread across the middle 90% of the recording;
  - reject any window that is flat, or whose amplitude distribution is dominated by
    clipping/disconnection artifacts (robust range test);
  - band-pass 5-40 Hz to isolate the QRS band;
  - detect R-peaks with scipy.find_peaks using a robust prominence (4x MAD) and a
    refractory distance set by the maximum plausible rat heart rate (600 bpm);
  - per window record the beat rate and the coefficient of variation of RR intervals;
  - aggregate by MEDIAN over surviving windows.

Decision rule -- a rat ECG trace has a fast, highly regular beat:
    ECG-like  : 3.5 <= median rate <= 10 Hz (210-600 bpm) AND median RR CV < 0.35
    EEG-like  : no such rhythm
    unusable  : fewer than MIN_OK usable windows
"""
import glob
import json
import os
import warnings

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
import mne  # noqa: E402
mne.set_log_level("ERROR")

RAW = "/path/to/archive"
N_WIN, WIN_S, MIN_OK = 12, 20.0, 4


def usable(x):
    """Reject flat / clipped windows only.

    An earlier version rejected any window with ptp/MAD > 100 as "artifact
    dominated". That threw away every ECG window: a sharp QRS spike over a quiet
    baseline is precisely a large peak-to-peak over a small MAD, so the filter
    removed the signature it was meant to preserve (all 12 windows rejected, ok=0,
    for both a known-ECG and a known-EEG file). Only genuinely unusable windows are
    dropped now: flat, non-finite, or clipped at a rail.
    """
    if not np.isfinite(x).all() or x.std() == 0:
        return False
    mad = np.median(np.abs(x - np.median(x)))
    if mad == 0:
        return False
    # clipping: a large share of samples pinned at the extreme value
    top = np.isclose(x, x.max(), rtol=0, atol=1e-12).mean()
    bot = np.isclose(x, x.min(), rtol=0, atol=1e-12).mean()
    return max(top, bot) < 0.05


def beat_stats(x, sf):
    b, a = butter(3, [5 / (sf / 2), min(40, sf / 2 - 1) / (sf / 2)], btype="band")
    y = filtfilt(b, a, x)
    mad = np.median(np.abs(y - np.median(y)))
    pk, _ = find_peaks(np.abs(y), prominence=4 * mad, distance=int(sf * 60 / 600))
    if len(pk) < 5:
        return None
    rr = np.diff(pk) / sf
    rr = rr[(rr > 0.08) & (rr < 0.5)]          # 120-750 bpm physiological window
    if len(rr) < 4:
        return None
    return dict(rate=1.0 / np.median(rr), cv=float(np.std(rr) / np.mean(rr)))


def analyse(path):
    r = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    names = r.ch_names
    eeg = [c for c in names if "EEG" in c.upper() and "FIR" not in c.upper()]
    ecg = [c for c in names if "ECG" in c.upper() and "FIR" not in c.upper()]
    label = "EEG" if eeg else ("ECG" if ecg else "??")
    pick = eeg or ecg
    if not pick:
        return label, None
    sf = r.info["sfreq"]
    n = int(WIN_S * sf)
    lo, hi = int(0.05 * r.n_times), int(0.95 * r.n_times) - n
    if hi <= lo:
        lo, hi = 0, max(1, r.n_times - n)
    rates, cvs, ok = [], [], 0
    for s in np.linspace(lo, hi, N_WIN).astype(int):
        x = r.get_data(picks=[names.index(pick[0])], start=s, stop=s + n)[0]
        if not usable(x):
            continue
        st = beat_stats((x - x.mean()) / x.std(), sf)
        ok += 1
        if st:
            rates.append(st["rate"]); cvs.append(st["cv"])
    if ok < MIN_OK:
        return label, dict(unusable=True, ok=ok)
    if not rates:
        return label, dict(unusable=False, ok=ok, rate=None, cv=None, frac=0.0)
    return label, dict(unusable=False, ok=ok, rate=float(np.median(rates)),
                       cv=float(np.median(cvs)), frac=len(rates) / ok)


def pick_file(subj):
    r = sorted(glob.glob(f"{RAW}/{subj}/*/{subj}-*.edf"))
    if r:
        return r[len(r) // 2], "raw"
    c = sorted(glob.glob(f"data/Data_{subj}_cropped/*/*/eeg.edf"))
    if c:
        return c[len(c) // 2], "clip"
    return None, None


subs = sorted({os.path.basename(d).split("_")[1] for d in glob.glob("data/Data_RN*_cropped")} |
              {os.path.basename(d) for d in glob.glob(f"{RAW}/RN*") if os.path.isdir(d)})
print(f"{'animal':8s} {'src':5s} {'label':6s} {'win':>5s} {'rate Hz':>8s} {'bpm':>6s} "
      f"{'RR cv':>7s} {'beat%':>6s}  verdict")
print("-" * 82)
rows, agree, disagree, unusable = {}, 0, 0, 0
for s in subs:
    f, kind = pick_file(s)
    if not f:
        print(f"{s:8s} {'--':5s} {'--':6s}  no EDF"); continue
    label, st = analyse(f)
    if st is None or st.get("unusable"):
        print(f"{s:8s} {kind:5s} {label:6s} {st.get('ok', 0) if st else 0:5d}"
              f"{'':32s}unusable (too few clean windows)")
        rows[s] = dict(label=label, verdict="unusable"); unusable += 1; continue
    if st["rate"] and 3.5 <= st["rate"] <= 10 and st["cv"] < 0.35 and st["frac"] > 0.5:
        sig = "ECG-like"
    else:
        sig = "EEG-like"
    flag = ""
    if sig[:3] != label:
        flag = "   <-- LABEL/SIGNAL DISAGREE"; disagree += 1
    else:
        agree += 1
    rate = st["rate"] or float("nan")
    print(f"{s:8s} {kind:5s} {label:6s} {st['ok']:5d} {rate:8.2f} {rate*60:6.0f} "
          f"{(st['cv'] if st['cv'] is not None else float('nan')):7.3f} "
          f"{100*st['frac']:5.0f}%  {sig}{flag}")
    rows[s] = dict(label=label, src=kind, verdict=sig, **st)

json.dump(rows, open("output/channel_census.json", "w"), indent=2)
n_eeg = sum(1 for v in rows.values() if v["label"] == "EEG")
print(f"\nlabel: {n_eeg} EEG / {len(rows)-n_eeg} ECG of {len(rows)} animals")
print(f"signal vs label: {agree} agree, {disagree} disagree, {unusable} unusable")
print("wrote output/channel_census.json")
