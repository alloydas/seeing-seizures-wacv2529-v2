#!/usr/bin/env python3
"""Evaluate an UNCUT session -- full-length video and its EDF -- on one wall clock.

Everything else in this repo scores pre-cut fixed-length clips. This scores the
recording as it actually arrives: an .mp4 (or several) plus an .edf plus the .xlsx
annotations, swept end to end, both modalities placed on one absolute time axis and
reported through one schema.

WHY IT EXISTS

  * `sweep_session_eeg.py` (now attic/py/) produced arithmetically WRONG ground truth.
    Its annotation reader keyed on a "clock" column that exists in none of 60 checked
    spreadsheets; absent it, `base = s` and every seizure collapsed to (0, duration).
    Verified on RN243: four events all beginning at second 0 of a 24-hour recording.
    Here, annotation times are converted through the EDF header's `meas_date` and a
    missing one is an error, never a silent fallback.

  * `sweep_session.py` is video-only, takes `edfs[0]` nowhere, and buries its output
    under `output/`. It is otherwise good and this script reuses its decoder verbatim.

  * Schema drift: `sweep`/`rows`, `thr`/`threshold`, `hours`/`duration_h`,
    `fp_per_hour`/`fp_per_h`. That last pair is not hypothetical --
    `sh/_rn243_swin_gpu0.sh:62` reads `b.get('fp_per_h', 0)` from a JSON that writes
    `fp_per_hour`, so its summary line has always printed 0.0. One schema here.

Outputs land in `<session>/session_eval/` by default, mirroring how the cutters write
`seizure_clips/` in place, rather than under `output/`.
"""
import argparse, csv, glob, json, os, sys, traceback
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse, never reimplement. sweep_video() is the expensive part and it is already
# right: one sequential pass, cap.grab() to skip and cap.retrieve() only for sampled
# frames (seeking is ruinous on copy-mode mp4s), and it DROPS windows hit by a decode
# gap rather than zero-filling them.
from sweep_session import (crop_box, sweep_video, extract_intervals, match,
                           CROPS as _CROPS_ROOMC)
# cut_seizure_clips.py runs `args = parse_args()` at MODULE level (line 56), so merely
# importing it parses sys.argv -- it hijacks our --help and would abort on any argv it
# does not recognise. Stub argv across the import and restore it. The proper fix is to
# guard that line in cut_seizure_clips.py, but that file is on the live cutting path
# (sh/_cut_worker.sh) and its module-level `args` is read throughout, so it is not
# something to change from here.
_argv = sys.argv
try:
    sys.argv = [_argv[0], "--session", "."]
    from cut_seizure_clips import (parse_annotations as _parse_ann_full,
                                   parse_edf_files, find_edf, _xml_start_time)
finally:
    sys.argv = _argv


# ── crop table ───────────────────────────────────────────────────────────────
# Merged from two shell tables that store their fields in DIFFERENT orders -- a live
# trap. sh/_crop_all.sh:23 reads `s wf xf hf yf`; sh/_crop_roomd.sh:36 reads
# `s xf wf yf hf`. crop_box() wants (left=width_frac, x_frac, top=height_frac, y_frac),
# so RoomC entries transfer directly and RoomD entries must be transposed.
# Cross-check: RoomD's "RN243 0.50 0.40 0.45 0.37" transposes to
# (0.40, 0.50, 0.37, 0.45), which is exactly the hand-verified RN243 entry already in
# sweep_session.CROPS. sweep_session covered 11 subjects; this covers 21.
_ROOMD_RAW = {                      # subject: (xf, wf, yf, hf)  <- source order
    "RN208": (0.00, 0.43, 0.33, 0.39), "RN223": (0.55, 0.42, 0.35, 0.38),
    "RN210": (0.10, 0.43, 0.42, 0.38), "RN224": (0.53, 0.40, 0.44, 0.38),
    "RN215": (0.00, 0.42, 0.20, 0.35), "RN219": (0.50, 0.43, 0.22, 0.35),
    "RN227": (0.00, 0.40, 0.42, 0.33), "RN244": (0.52, 0.46, 0.42, 0.36),
    "RN242": (0.06, 0.44, 0.44, 0.36), "RN243": (0.50, 0.40, 0.45, 0.37),
    "RN229": (0.00, 0.45, 0.22, 0.35),
}
CROPS = dict(_CROPS_ROOMC)
for _s, (_xf, _wf, _yf, _hf) in _ROOMD_RAW.items():
    CROPS.setdefault(_s, (_wf, _xf, _hf, _yf))     # -> (left, xf, top, yf)

SEIZURE_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]


def subject_of(path):
    import re
    m = re.search(r"(RN\d{3})", os.path.abspath(path))
    return m.group(1) if m else None


# ── time alignment ───────────────────────────────────────────────────────────

def video_start(fpath, strict):
    """Absolute start of one mp4, from its .XML sidecar.

    The 14-digit filename timestamp is NOT trusted as a fallback. Both
    sweep_session.parse_video_files() and cut_seizure_clips.parse_video_files() fall
    back to it, and that fallback has never fired only because every mp4 in this
    dataset has a sidecar -- which is exactly why a mistake there would go unnoticed.
    Under --strict a missing sidecar is an error; otherwise the video is skipped with
    a warning, which is still safer than placing it at a guessed hour.
    """
    dt = _xml_start_time(fpath)
    if dt is None:
        msg = f"no .XML sidecar for {os.path.basename(fpath)}; refusing to guess from filename"
        if strict:
            raise SystemExit(f"[strict] {msg}")
        print(f"  [warn] {msg} -- skipped")
    return dt


def load_videos(folder, strict):
    import cv2
    out, missing = [], 0
    for fpath in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
        dt = video_start(fpath, strict)
        if dt is None:
            missing += 1
            continue
        cap = cv2.VideoCapture(fpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            print(f"  [warn] unreadable {os.path.basename(fpath)} -- skipped")
            missing += 1
            continue
        out.append(dict(path=fpath, start=dt, fps=fps, n=n,
                        end=dt + timedelta(seconds=n / fps)))
    return out, missing


def annotations(xlsx, keep_labels, include_spikes):
    """Annotation intervals as naive local datetimes, filtered by label.

    Label filtering is the point. Neither existing sweep filters at all, so `Spikes`
    rows are scored as seizures -- and 3 of RN243's 4 annotated events are Spikes,
    which by itself reframes every published RN243 recall number.
    """
    rows = _parse_ann_full(xlsx)
    kept, excluded = [], 0
    for r in rows:
        lab = str(r.get("label") or r.get("stage") or "").strip()
        if not include_spikes and "spike" in lab.lower():
            excluded += 1
            continue
        if keep_labels and not any(k.lower() in lab.lower() for k in keep_labels):
            excluded += 1
            continue
        st, en = r.get("start"), r.get("end")
        if st is None:
            excluded += 1
            continue
        if en is None and r.get("duration") is not None:
            en = st + timedelta(seconds=float(r["duration"]))
        if en is None:
            excluded += 1
            continue
        kept.append(dict(label=lab, start=st, end=en))
    return kept, excluded, len(rows)


# ── EEG ──────────────────────────────────────────────────────────────────────

def select_channel(raw):
    """First channel named EEG, else first named ECG, else channel 0.

    Note for interpretation, not code: across this cohort 19 of 20 subjects carry ECG
    and only RN219 carries true EEG, so a run labelled "EEG" is usually single-lead
    ECG. The channel kind is reported in qc so the distinction survives into results.
    """
    names = raw.ch_names
    for kind in ("EEG", "ECG"):
        for i, n in enumerate(names):
            if kind.lower() in n.lower():
                return i, n, kind
    return 0, names[0], "UNKNOWN"


def sweep_eeg(edf_path, model, dev, args, t_window):
    """Yield (absolute_epoch_seconds, prob) for every EEG window.

    Reads the EDF header only (preload=False), crops, then load_data() -- roughly two
    orders of magnitude less I/O than preloading a multi-hour recording, which is the
    bug that made the cut scripts take >5 min per clip before 2026-08-11.
    """
    import mne
    from scipy.signal import resample_poly
    import torch

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    meas = raw.info.get("meas_date")
    if meas is None:
        raise SystemExit(f"[fatal] {os.path.basename(edf_path)} has no meas_date; "
                         "absolute alignment is impossible (this is the failure "
                         "sweep_session_eeg.py papered over)")
    t0_abs = meas.replace(tzinfo=None).timestamp()
    sfreq = float(raw.info["sfreq"])

    # Decimation from the file, never hardcoded. Every EEG path in this repo carries a
    # literal DECIM = 8, which silently produces the wrong window length on any
    # recording that is not 1000 Hz. Checkpoints here were trained at 125 Hz.
    decim = max(1, int(round(sfreq / 125.0))) if args.eeg_decim == "auto" else int(args.eeg_decim)
    target_sf = sfreq / decim
    ch_i, ch_name, ch_kind = select_channel(raw)

    dur = raw.n_times / sfreq
    if args.max_seconds:
        dur = min(dur, float(args.max_seconds))
    win, stride = args.eeg_win, args.eeg_stride
    starts = np.arange(0.0, max(0.0, dur - win), stride)

    out, buf_t, buf_x = [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).float().unsqueeze(-1).to(dev)
        with torch.no_grad():
            logits = model(x)
            p = torch.softmax(logits.float(), 1)[:, 1] if logits.shape[-1] > 1 \
                else torch.sigmoid(logits.float()).squeeze(-1)
        out.extend(zip(buf_t, p.cpu().numpy().tolist()))
        buf_t.clear(); buf_x.clear()

    for s in starts:
        seg = raw.copy().crop(tmin=float(s), tmax=float(min(s + win, dur)), include_tmax=False)
        seg.load_data(verbose="ERROR")
        sig = seg.get_data(picks=[ch_i])[0]
        if decim > 1:
            sig = resample_poly(sig, 1, decim)
        need = int(round(win * target_sf))
        if len(sig) < need:
            continue
        sig = sig[:need]
        sd = sig.std()
        sig = (sig - sig.mean()) / (sd if sd > 1e-12 else 1.0)   # per-window z-score
        buf_t.append(t0_abs + float(s))
        buf_x.append(sig.astype(np.float32))
        if len(buf_x) >= args.batch_size:
            flush()
    flush()
    return out, dict(sfreq=sfreq, decim=decim, channel_name=ch_name,
                     channel_kind=ch_kind, edf_start_abs=t0_abs, edf_dur_s=dur)


# ── scoring ──────────────────────────────────────────────────────────────────

def score(trace_t, trace_p, gt_abs, args, hours, modality):
    """One scoring rule for every modality: greedy one-to-one temporal IoU.

    The two existing sweeps disagree -- video does greedy IoU>=0.3, the EEG path
    counted any overlap and mixed units (precision = tp/len(det) where tp counted
    ground-truth events, so precision could exceed 1). This uses match() for both.
    """
    rows, events = [], []
    for thr in args.thresholds:
        det = extract_intervals(trace_t, trace_p, thr, args.video_win,
                                min_dur=args.min_dur, merge_gap=args.merge_gap)
        pairs, used = match(det, gt_abs, min_iou=args.min_iou)
        tp, fp, fn = len(pairs), len(det) - len(used), len(gt_abs) - len(pairs)
        onset = [abs(det[di][0] - gt_abs[gi][0]) for gi, di, _ in pairs]
        offset = [abs(det[di][1] - gt_abs[gi][1]) for gi, di, _ in pairs]
        rows.append(dict(
            modality=modality, threshold=float(thr), n_det=len(det),
            tp=tp, fp=fp, fn=fn,
            recall=tp / len(gt_abs) if gt_abs else 0.0,
            precision=tp / len(det) if det else 0.0,
            fp_per_hour=fp / hours if hours else 0.0,
            onset_mae=float(np.mean(onset)) if onset else None,
            offset_mae=float(np.mean(offset)) if offset else None,
            mean_iou=float(np.mean([p[2] for p in pairs])) if pairs else 0.0))
        m = {di: (gi, iou) for gi, di, iou in pairs}
        for di, (ds, de) in enumerate(det):
            gi, iou = m.get(di, (None, 0.0))
            seg = [p for t, p in zip(trace_t, trace_p) if ds <= t <= de]
            events.append(dict(modality=modality, threshold=float(thr),
                               t_start=float(ds), t_end=float(de),
                               peak_prob=float(max(seg)) if seg else None,
                               matched_gt_idx=gi, iou=float(iou)))
    return rows, events


def curves(trace_t, trace_p, gt_abs, args, hours):
    """AUROC/AUPRC on the continuous trace plus sensitivity vs FP/h.

    The deliverable for a clinician is "what threshold gives at most N false alarms
    per hour", which five fixed thresholds cannot answer.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([any(gs <= t + args.video_win / 2 <= ge for gs, ge in gt_abs)
                  for t in trace_t], dtype=int)
    p = np.asarray(trace_p, dtype=float)
    out = {"auroc": None, "auprc": None, "sens_vs_fp_per_hour": []}
    if y.any() and not y.all():
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    for thr in np.linspace(0.05, 0.99, 40):
        det = extract_intervals(trace_t, trace_p, float(thr), args.video_win,
                                min_dur=args.min_dur, merge_gap=args.merge_gap)
        pairs, used = match(det, gt_abs, min_iou=args.min_iou)
        fp = len(det) - len(used)
        out["sens_vs_fp_per_hour"].append(
            [fp / hours if hours else 0.0,
             len(pairs) / len(gt_abs) if gt_abs else 0.0, float(thr)])
    return out


def fuse(vt, vp, et, ep, how, w):
    """Session-level cross-modal fusion on the shared absolute axis.

    All fusion in this repo is clip-level (attic/based_on_eeg/fusion_train.py and the
    archived tex_compare/make_fusion_figs.py). Nothing fuses a session. EEG probability
    is sampled onto the video axis by nearest neighbour.
    """
    if how == "none" or not vt or not et:
        return [], []
    e_t, e_p = np.asarray(et), np.asarray(ep)
    idx = np.clip(np.searchsorted(e_t, vt), 0, len(e_t) - 1)
    ep_on_v = e_p[idx]
    v = np.asarray(vp)
    if how == "mean":  f = w * v + (1 - w) * ep_on_v
    elif how == "max": f = np.maximum(v, ep_on_v)
    elif how == "or":  f = np.maximum(v, ep_on_v)
    elif how == "and": f = np.minimum(v, ep_on_v)
    else:              return [], []
    return list(vt), f.tolist()


# ── per-session driver ───────────────────────────────────────────────────────

def find_inputs(folder):
    x = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
    return (x[0] if x else None), sorted(glob.glob(os.path.join(folder, "*.edf")))


def run_session(folder, args, video_model, eeg_model, dev):
    import torch
    subj = args.crop if args.crop not in (None, "auto", "none") else subject_of(folder)
    fr = CROPS.get(subj) if args.crop != "none" else None
    if args.crop_box:
        fr = tuple(float(v) for v in args.crop_box.split(","))
    if video_model is not None and fr is None:
        print(f"  [warn] no crop entry for {subj}; sweeping the full frame")
        fr = (1.0, 0.0, 1.0, 0.0)

    xlsx, edfs = find_inputs(folder)
    if xlsx is None:
        raise SystemExit(f"[fatal] no .xlsx in {folder}")
    videos, n_missing_xml = load_videos(folder, args.strict)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos and video_model is not None:
        raise SystemExit(f"[fatal] no usable mp4 in {folder}")

    ann, excluded, total = annotations(xlsx, args.labels, args.include_spikes)
    qc = dict(n_mp4=len(videos), n_mp4_missing_xml=n_missing_xml, n_edf=len(edfs),
              annotations_total=total, annotations_kept=len(ann),
              labels_excluded=excluded, video_gap_windows_dropped=None,
              edf_selected=None, edf_xml_delta_s=None)

    # ---- video sweep, placed on the absolute axis ----
    vt, vp = [], []
    if video_model is not None:
        class _A:  # sweep_video reads these off an argparse-ish object
            win, stride = args.video_win, args.video_stride
            frames, size = args.frames, args.size
            batch_size, max_seconds = args.batch_size, args.max_seconds
        for vid in videos:
            rel = sweep_video(vid, video_model, dev, fr, _A)
            base = vid["start"].timestamp()
            vt.extend(base + t + args.video_offset_s for t, _ in rel)
            vp.extend(p for _, p in rel)
        order = np.argsort(vt)
        vt = list(np.asarray(vt)[order]); vp = list(np.asarray(vp)[order])

    # ---- EEG sweep ----
    et, ep, eeg_qc = [], [], {}
    if eeg_model is not None and edfs:
        recs = parse_edf_files(folder)
        chosen = None
        if recs and ann:
            chosen = find_edf(ann[0]["start"], ann[-1]["end"], recs)
        if chosen is None:
            # parse_edf_files() keys its records by "fpath"/"fname", not "path" --
            # match that schema so the coverage hit and the fallback are interchangeable.
            chosen = dict(fpath=edfs[0])
            if len(edfs) > 1:
                print(f"  [warn] {len(edfs)} EDFs and no coverage match; using the first")
        qc["edf_selected"] = os.path.basename(chosen["fpath"])
        pairs_, eeg_qc = sweep_eeg(chosen["fpath"], eeg_model, dev, args, args.eeg_win)
        et = [t for t, _ in pairs_]
        ep = [p for _, p in pairs_]
        qc.update(eeg_qc)
        if videos:
            delta = videos[0]["start"].timestamp() - eeg_qc["edf_start_abs"]
            qc["edf_xml_delta_s"] = float(delta)
            # Constant +17..+18 s across sessions: the headstage starts before the
            # camera. Absolute alignment absorbs it, so do NOT correct for it -- but
            # a value outside this band means one of the two clocks is wrong.
            if not (10.0 <= delta <= 25.0):
                print(f"  [warn] video-minus-EDF delta {delta:.1f}s outside the "
                      f"expected 10-25s device stagger -- check the clocks")
    elif eeg_model is not None:
        print("  [warn] no EDF in session; degrading to video-only")

    # ---- ground truth on the absolute axis ----
    ref = eeg_qc.get("edf_start_abs")
    gt_abs = [(a["start"].timestamp(), a["end"].timestamp()) for a in ann]
    hours = 0.0
    if vt: hours = (max(vt) - min(vt)) / 3600.0
    elif et: hours = (max(et) - min(et)) / 3600.0

    rows, events = [], []
    curv = {}
    if vt:
        r, e = score(vt, vp, gt_abs, args, hours, "video"); rows += r; events += e
        curv["video"] = curves(vt, vp, gt_abs, args, hours)
    if et:
        r, e = score(et, ep, gt_abs, args, hours, "eeg"); rows += r; events += e
        curv["eeg"] = curves(et, ep, gt_abs, args, hours)
    ft, fp_ = fuse(vt, vp, et, ep, args.fusion, args.fusion_weight)
    if ft:
        r, e = score(ft, fp_, gt_abs, args, hours, f"fusion:{args.fusion}")
        rows += r; events += e
        curv[f"fusion:{args.fusion}"] = curves(ft, fp_, gt_abs, args, hours)

    out_dir = args.out or os.path.join(folder, "session_eval")
    os.makedirs(out_dir, exist_ok=True)
    res = dict(session=os.path.abspath(folder), subject=subj, hours=hours,
               video_ckpt=args.video_ckpt, eeg_ckpt=args.eeg_ckpt,
               video_arch=args.video_arch, eeg_arch=args.eeg_arch, qc=qc,
               annotations=[dict(idx=i, label=a["label"],
                                 t_start=a["start"].timestamp(),
                                 t_end=a["end"].timestamp(),
                                 duration=(a["end"] - a["start"]).total_seconds())
                            for i, a in enumerate(ann)],
               sweep=rows, detected_events=events, curves=curv)
    with open(os.path.join(out_dir, "session_eval.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez(os.path.join(out_dir, "trace.npz"),
             t_video=np.asarray(vt), p_video=np.asarray(vp),
             t_eeg=np.asarray(et), p_eeg=np.asarray(ep),
             ann=np.asarray(gt_abs, dtype=float).reshape(-1, 2),
             win=args.video_win, stride=args.video_stride)
    return res, out_dir


# ── models ───────────────────────────────────────────────────────────────────

def load_video_model(ckpt, arch, dev):
    import torch
    from train_pooled import build_video_model
    model, frames, size = build_video_model(arch, 2, pretrained=False)
    sd = torch.load(ckpt, map_location="cpu")          # torch<=2.5; see requirements.txt
    model.load_state_dict(sd.get("model", sd))
    return model.to(dev).eval(), frames, size


def load_eeg_model(ckpt, arch, dev, hidden=128):
    import torch
    from train_pooled_eeg import build_model as build_eeg
    model = build_eeg(arch, 2, hidden)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    return model.to(dev).eval()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", nargs="+")
    g.add_argument("--batch")
    p.add_argument("--video_ckpt"); p.add_argument("--video_arch", default="slowfast")
    p.add_argument("--eeg_ckpt");   p.add_argument("--eeg_arch", default="gru")
    p.add_argument("--video_win", type=float, default=4.0)
    p.add_argument("--video_stride", type=float, default=2.0)
    # 0 = use the backbone's native geometry (train_pooled.py uses the same convention).
    # A non-zero default here would silently override it: slowfast needs 32x224 and would
    # have been fed 16x112.
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--eeg_win", type=float, default=6.0)
    p.add_argument("--eeg_stride", type=float, default=3.0)
    p.add_argument("--eeg_decim", default="auto")
    p.add_argument("--video_offset_s", type=float, default=0.0)
    p.add_argument("--labels", nargs="*", default=SEIZURE_LABELS)
    p.add_argument("--include_spikes", action="store_true")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.5, 0.7, 0.9, 0.95, 0.99])
    p.add_argument("--min_iou", type=float, default=0.3)
    p.add_argument("--min_dur", type=float, default=4.0)
    p.add_argument("--merge_gap", type=float, default=4.0)
    p.add_argument("--fusion", choices=["none", "or", "and", "mean", "max"], default="none")
    p.add_argument("--fusion_weight", type=float, default=0.5)
    p.add_argument("--crop", default="auto"); p.add_argument("--crop_box")
    p.add_argument("--out"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_videos", type=int); p.add_argument("--max_seconds", type=float)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    if not a.video_ckpt and not a.eeg_ckpt:
        p.error("at least one of --video_ckpt / --eeg_ckpt is required")
    return a


def main():
    import torch
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vm = em = None
    if args.video_ckpt:
        vm, fr_, sz_ = load_video_model(args.video_ckpt, args.video_arch, dev)
        args.frames = args.frames or fr_
        args.size = args.size or sz_
        print(f"video: arch={args.video_arch} frames={args.frames} size={args.size}")
    if args.eeg_ckpt:
        em = load_eeg_model(args.eeg_ckpt, args.eeg_arch, dev)

    sessions = args.session
    if args.batch:
        sessions = [d for d in sorted(glob.glob(os.path.join(args.batch, "*")))
                    if os.path.isdir(d) and glob.glob(os.path.join(d, "*.xlsx"))]
        print(f"batch: {len(sessions)} candidate session folders under {args.batch}")

    log = os.path.join(args.batch or ".", "_session_eval_log.csv") if args.batch else None
    done = set()
    if log and args.resume and os.path.exists(log):
        with open(log) as fh:
            done = {r["session"] for r in csv.DictReader(fh)}
        print(f"resume: skipping {len(done)} already-scored sessions")

    for folder in sessions:
        if os.path.abspath(folder) in done:
            continue
        print(f"\n=== {folder} ===")
        try:
            res, out_dir = run_session(folder, args, vm, em, dev)
        except SystemExit as e:
            print(f"  {e}")
            continue
        except Exception:
            traceback.print_exc()
            continue
        for r in res["sweep"]:
            if r["threshold"] == 0.5:
                print(f"  {r['modality']:14s} thr0.50  recall={r['recall']:.3f}  "
                      f"prec={r['precision']:.3f}  FP/h={r['fp_per_hour']:.1f}  "
                      f"n_det={r['n_det']}")
        print(f"  -> {out_dir}/session_eval.json")
        if log:
            new = not os.path.exists(log)
            with open(log, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["session", "subject", "hours", "n_ann", "modalities"])
                w.writerow([res["session"], res["subject"], f"{res['hours']:.2f}",
                            len(res["annotations"]),
                            "|".join(sorted({r["modality"] for r in res["sweep"]}))])


if __name__ == "__main__":
    main()
