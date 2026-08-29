"""
Build a 5-class STAGE-labelled EEG window cache from the combined cropped clip dirs.

Mirrors based_on_eeg/build_segments.py preprocessing exactly:
  first EEG channel (ECG fallback), decimate 1000->125 Hz, 6 s windows, 3 s stride,
  per-window z-score is applied later at train time.

Window label: the clip's stage class if the window overlaps the seizure interval by
>= 0.8*6s, else 0 (non-seizure). Non-seizure clips -> all windows 0.

Outputs stage_segments_pooled.npz: segs, wlab (window labels), clip_id, clip_lab, clip_sess
"""
import glob, os, re, sys, warnings
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, mne
from scipy.signal import resample_poly

warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
ROOTS = {
    # RoomC
    "RN197": "data/Data_RN197_cropped", "RN199": "data/Data_RN199_cropped",
    "RN204": "data/Data_RN204_cropped", "RN213": "data/Data_RN213_cropped",
    "RN216": "data/Data_RN216_cropped", "RN222": "data/Data_RN222_cropped",
    "RN235": "data/Data_RN235_cropped", "RN237": "data/Data_RN237_cropped",
    "RN238": "data/Data_RN238_cropped", "RN245": "data/Data_RN245_cropped",
    # RoomD (added 2026-07-22)
    "RN208": "data/Data_RN208_cropped", "RN210": "data/Data_RN210_cropped",
    "RN215": "data/Data_RN215_cropped", "RN219": "data/Data_RN219_cropped",
    "RN223": "data/Data_RN223_cropped", "RN224": "data/Data_RN224_cropped",
    "RN227": "data/Data_RN227_cropped", "RN229": "data/Data_RN229_cropped",
    "RN242": "data/Data_RN242_cropped", "RN244": "data/Data_RN244_cropped",
}
STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}
CHUNK_S, STRIDE_S, POS_OVERLAP, DECIM = 6.0, 3.0, 0.8, 8   # defaults; overridden by CLI
OUT_PATH = "stage_segments_pooled.npz"
N_WORKERS = 12
DT = "%Y-%m-%d %H:%M:%S.%f"


def to_dt(s):
    for f in (DT, "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s, f)
        except ValueError: pass
    return None


def parse_info(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    return d


def seizure_rel(info):
    cs, ss, se = info.get("Clip start"), info.get("Seizure start"), info.get("Seizure end")
    if cs and ss and se:
        c, s, e = to_dt(cs), to_dt(ss), to_dt(se)
        if c and s and e:
            return (s - c).total_seconds(), (e - c).total_seconds()
    try:
        pre = float(info.get("Pre-buffer (s)", "nan")); dur = float(info.get("Duration (s)", "nan"))
        if pre == pre and dur == dur: return pre, pre + dur
    except ValueError: pass
    return None


def one(job):
    cdir, ycls, sess = job
    edf = Path(cdir) / "eeg.edf"
    if not edf.exists(): return None
    try:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    except Exception:
        return None
    eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
    ecg = [c for c in raw.ch_names if "ECG" in c.upper()]
    chans = eeg or ecg
    if not chans: return None
    sig = raw.get_data(picks=[raw.ch_names.index(chans[0])])[0]
    sf = raw.info["sfreq"]
    if DECIM > 1:
        sig = resample_poly(sig, 1, DECIM).astype(np.float32); sf = sf / DECIM
    cs, ss = int(CHUNK_S * sf), int(STRIDE_S * sf)
    sz = None
    if ycls > 0 and (Path(cdir) / "info.txt").exists():
        sz = seizure_rel(parse_info(Path(cdir) / "info.txt"))
    segs, labs = [], []
    i = 0
    while i + cs <= len(sig):
        w0, w1 = i / sf, (i + cs) / sf
        lab = 0
        if sz is not None:
            ov = max(0.0, min(w1, sz[1]) - max(w0, sz[0]))
            if ov >= POS_OVERLAP * CHUNK_S:
                lab = ycls
        segs.append(sig[i:i + cs].astype(np.float32)); labs.append(lab)
        i += ss
    if not segs: return None
    return np.stack(segs), np.array(labs, np.int64), ycls, sess, str(cdir)


def main():
    jobs = []
    for subj, root in ROOTS.items():
        for d in glob.glob(f"{root}/*/seizure_*") + glob.glob(f"{root}/*/clip_*_vs_seizure_*"):
            if not os.path.exists(os.path.join(d, "video.mp4")): continue
            b, sess = os.path.basename(d), os.path.basename(os.path.dirname(d))
            if b.startswith("seizure_"):
                m = re.search(r"Stage_[0-9]+", b)
                if not m or m.group() not in STAGE: continue
                y = STAGE[m.group()]
            else:
                y = 0
            jobs.append((d, y, f"{subj}/{sess}"))
    print(f"{len(jobs)} clips -> extracting windows", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is not None: results.append(r)
            if (k + 1) % 500 == 0: print(f"  {k+1}/{len(jobs)}", flush=True)

    # Sort by clip path so the cache is byte-stable across builds. Previously clips
    # were numbered in as_completed() order, so two builds of the SAME config produced
    # different orderings -- which is why predictions from different runs could not be
    # aligned per-clip, blocking paired video-vs-EEG McNemar tests.
    results.sort(key=lambda r: r[4])

    S, W, CID, CLAB, CSESS, CPATH = [], [], [], [], [], []
    for n, (segs, labs, ycls, sess, cpath) in enumerate(results):
        S.append(segs); W.append(labs)
        CID.append(np.full(len(segs), n, np.int64))
        CLAB.append(ycls); CSESS.append(sess); CPATH.append(cpath)
    n = len(results)

    segs = np.concatenate(S); wlab = np.concatenate(W); cid = np.concatenate(CID)
    clab = np.array(CLAB, np.int64); csess = np.array(CSESS)
    cpath = np.array(CPATH)
    print(f"clips={n}  windows={len(segs)}  shape={segs.shape}")
    print("window label counts:", np.bincount(wlab, minlength=5).tolist())
    print("clip   label counts:", np.bincount(clab, minlength=5).tolist())
    np.savez_compressed(OUT_PATH, segs=segs, wlab=wlab, clip_id=cid,
                        clip_lab=clab, clip_sess=csess, clip_path=cpath)
    print(f"wrote {OUT_PATH}")


def _cli():
    """Expose the preprocessing knobs the EEG window/rate sweep (plan item #8) needs.

    Defaults reproduce the original constants exactly, so an argument-free run is
    bit-identical to the pre-sweep behaviour.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--win", type=float, default=6.0, help="window length, seconds")
    ap.add_argument("--stride", type=float, default=3.0, help="hop between windows, seconds")
    ap.add_argument("--decim", type=int, default=8,
                    help="decimation from 1000 Hz (8 -> 125 Hz, 4 -> 250 Hz, 16 -> 62.5 Hz)")
    ap.add_argument("--overlap", type=float, default=0.8,
                    help="fraction of a window that must lie inside the seizure to inherit its stage")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="stage_segments_pooled.npz")
    a = ap.parse_args()

    global CHUNK_S, STRIDE_S, POS_OVERLAP, DECIM, OUT_PATH, N_WORKERS
    CHUNK_S, STRIDE_S, POS_OVERLAP, DECIM = a.win, a.stride, a.overlap, a.decim
    OUT_PATH, N_WORKERS = a.out, a.workers
    print(f"win={CHUNK_S}s stride={STRIDE_S}s decim={DECIM} "
          f"({1000/DECIM:.4g} Hz, {int(CHUNK_S*1000/DECIM)} samples/window) -> {OUT_PATH}",
          flush=True)
    main()


if __name__ == "__main__":
    _cli()
