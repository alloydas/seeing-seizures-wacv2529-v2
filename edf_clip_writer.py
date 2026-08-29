"""Write an MNE Raw window to EDF with PER-CHANNEL physical ranges.

Replaces `mne.export.export_raw(..., fmt="edf")` in the clip cutters.

THE DEFECT THIS FIXES
---------------------
mne.export.export_raw writes a SHARED physical range across all channels. These
recordings mix a ~0.0013-amplitude biopotential channel with SignalStr (~31) and
Temp (~37) in the same file, so the shared range spans roughly [0, 37] and one
16-bit step is ~5.6e-4 -- larger than the entire ECG/EEG waveform. Every sample
lands in 2-3 quantization bins.

Measured on 24,214 cut clips (2026-08-12): 69.2% degenerate (<100 unique values,
median 4), 1.1% completely flat. Thirteen of twenty animals had NO usable clip EDF,
while the raw sessions they were cut from are healthy (1800-4600 unique values per
60 s). Verified on one window: source 4038 unique values -> mne.export 3 unique.

Writing per-channel ranges with edfio reproduces the source to correlation 1.000000.

NOTES
-----
* EDF stores whole data records (1 s here), so the signal is trimmed to an integer
  number of seconds. Callers that care about the exact end time should account for
  up to 1 s of truncation.
* Channel labels are truncated to the EDF limit of 16 characters by the caller.
"""
import numpy as np


def write_edf_clip(raw, out_path, record_s=1.0):
    """Write `raw` (an MNE Raw, already cropped and loaded) to `out_path` as EDF.

    Each channel gets its own physical range taken from its own data, so a
    large-amplitude auxiliary channel cannot crush a small-amplitude biopotential.
    """
    from edfio import Edf, EdfSignal

    sf = float(raw.info["sfreq"])
    data = raw.get_data()
    # EDF requires an integer number of data records
    n = int(data.shape[1] // (sf * record_s) * (sf * record_s))
    if n <= 0:
        raise ValueError(f"window shorter than one {record_s}s data record")
    data = data[:, :n]

    signals = []
    for k, ch in enumerate(raw.ch_names):
        v = np.asarray(data[k], dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            lo, hi = -1.0, 1.0
            v = np.zeros_like(v)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            v = np.nan_to_num(v, nan=lo, posinf=hi, neginf=lo)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # constant channel: EDF needs a non-degenerate physical range
            lo, hi = lo - 1e-6, lo + 1e-6
        signals.append(EdfSignal(v, sampling_frequency=sf, label=ch[:16],
                                 physical_range=(lo, hi)))
    Edf(signals, data_record_duration=record_s).write(out_path)
    return n / sf
