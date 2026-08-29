# Video–EEG Seizure Analysis

**📊 Project page:** [https://anonymous.4open.science/w/seeing-seizures-wacv2529-v2/index.html](https://anonymous.4open.science/w/seeing-seizures-wacv2529-v2/index.html) — interactive results, seizure clips with paired ECG, Grad-CAM videos, and the full backbone leaderboards.

**🧠 Model weights:** [https://drive.google.com/drive/folders/1mAclhqRjE9wAIW_B6Jh2yF-nfW54Df5L?usp=drive_link](https://drive.google.com/drive/folders/1mAclhqRjE9wAIW_B6Jh2yF-nfW54Df5L?usp=drive_link) — best checkpoint per task; checksums and loading notes in `models/`.

A toolkit for seizure detection and severity grading in rodents from
**synchronized video and EDF biopotential recordings**. It takes raw `.mp4`
video, `.edf` recordings and `.xlsx` seizure annotations, cuts labelled clips,
trains and compares ten video backbones and five EEG models, and evaluates
whole uncut sessions.

**If you have never run this before, read [Quick start](#quick-start) then
[The five stages](#the-five-stages) in order.** Every command below is
copy-pasteable. Setup details live in [`SETUP.md`](SETUP.md).

---

## Contents

1. [What you need](#what-you-need)
2. [Quick start](#quick-start)
3. [How the data must be laid out](#how-the-data-must-be-laid-out)
4. [The five stages](#the-five-stages)
5. [Script reference](#script-reference)
6. [Things that will bite you](#things-that-will-bite-you)
7. [Repository layout](#repository-layout)
8. [Troubleshooting](#troubleshooting)

---

## What you need

| | |
|---|---|
| OS | Linux (developed on Ubuntu) |
| Python | 3.12 |
| GPU | NVIDIA with ≥16 GB for the large backbones; CUDA 12.1-capable driver |
| Disk | **150 GB+** free — the frame caches alone are 110 GB and 55 GB |
| RAM | 32 GB is enough; the trainers stream from a memory-mapped cache |
| ffmpeg | **not needed separately** — `imageio-ffmpeg` ships a static build |

No GPU? Everything except training and sweeping still runs: clip cutting,
cache building, annotation parsing and all the table/figure generators.

---

## Quick start

```bash
git clone https://anonymous.4open.science/r/seeing-seizures-wacv2529-v2/
cd EEG-seizure-classification

python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.__version__)"
```

Expected: `CUDA: True 2.4.0+cu121`. If it prints `False`, training will fall
back to CPU and be unusably slow — fix the driver first.

**Smoke-test without any data** (verifies the install, ~10 s):

```bash
python -c "
from train_pooled import build_video_model
m, frames, size = build_video_model('r2plus1d', 2, pretrained=False)
print('model built OK:', frames, 'frames at', size, 'px')"
# -> model built OK: 16 frames at 112 px
# (r2plus1d honours pretrained=False, so this needs no network. Note x3d and
#  slowfast always fetch pretrained weights regardless of that flag.)
```

---

## How the data must be laid out

A **session folder** is the unit everything iterates over. It holds one or more
videos, one or more EDFs, and exactly one annotation spreadsheet:

```
Data/RN208/11-20-2023/
├── RN208-RN223.20231120220200.mp4      # video
├── RN208-RN223.20231120220200.XML      # sidecar — REQUIRED, see below
├── RN208-RN223.20231121055400.mp4      # sessions often span several files
├── RN208-RN223.20231121055400.XML
├── RN208-11-20-2023.edf                # biopotential recording
└── RN208-11-20-2023.xlsx               # annotations
```

**The `.XML` sidecar is not optional.** It carries `DSI_utc_start_time`, the
only trustworthy start time for a video. The 14-digit number in the filename is
the recording **end** time in UTC — measured on real files as
`filename − XML_end = +5.00 h` exactly. Treating it as a start time misplaces a
video by 11–12 hours.

**The annotation spreadsheet** needs columns like `Label`, `Duration`,
`Start Time`, `End Time`. Header matching is fuzzy and case-insensitive, so
`Stage`/`Length`/`Start` also work. `Label` values look like `Stage 1` … `Stage 5`
and `Spikes`. **`Spikes` are not seizures** and are excluded by default.

---

## The five stages

### Stage 1 — Cut clips from raw sessions

```bash
# one session
python cut_seizure_clips.py --session Data/RN208/11-20-2023 --pre 10 --post 10

# every session under a root, matched non-seizure clips too
python cut_seizure_clips.py     --batch Data/RN208 --pre 10 --post 10 --mode copy
python cut_non_seizure_clips.py --batch Data/RN208 --pre 10 --post 10 --safety 30 --seed 42
```

Produces `seizure_clips/seizure_N/{video.mp4, eeg.edf, info.txt}` and a matched
`non_seizure_clips/` beside it. `--mode copy` is fast but GOP-aligned;
`--mode reencode` is frame-accurate and much slower.

`cut_non_seizure_clips.py` samples one negative per seizure, of identical
duration, from a window that does not overlap any seizure ± `--safety` seconds.
That matching is what stops a model from winning on appearance alone.

**Optional — crop to one animal's cage** when a camera films several:

```bash
python crop_clips.py --help     # crop fractions per subject
```

### Stage 2 — Build the caches

Decoding H.264 during training is 8× slower than reading pre-decoded frames, so
decode once:

```bash
# video frame cache (uint8 memmap). 32 frames @224 px = 110 GB for 24.5k clips
python build_frame_cache.py --frames 32 --size 224 --out cache_frames/f32s224 --workers 12

# smaller/faster geometry used by most backbones
python build_frame_cache.py --frames 16 --size 224 --out cache_frames/f16s224 --workers 12

# EEG windows
python build_stage_segments_pooled.py --win 6 --stride 3 --decim 8 --out stage_segments_pooled_v3.npz
```

`--decim 8` takes 1000 Hz to 125 Hz, which is what every EEG checkpoint expects.

### Stage 3 — Train

```bash
# video: detection (2-class)
python train_pooled.py --arch slowfast --group2 --epochs 12 --batch_size 8 --lr 1e-4 \
    --workers 5 --seed 42 --split_seed 49 --cache_dir cache_frames/f32s224 \
    --output output/my_run

# 3-class severity: --group3 ;  5-class: pass neither flag
# architectures: r2plus1d swin swin_s mvit mvit_v1 s3d x3d slowfast

# EEG
python train_pooled_eeg.py --arch gru --group2 --epochs 30 --hidden 128 \
    --cache stage_segments_pooled_v3.npz --output output/my_eeg_run
```

Each run writes `results.json` (metrics), `history.json` (per-epoch),
`val_preds.npz` (per-clip predictions) and `best.pt` (weights).

Add `--deterministic` for bit-reproducible training. It is **off by default**
because every published number here was produced without it — see
[Things that will bite you](#things-that-will-bite-you).

### Stage 4 — Evaluate a whole uncut session

Stages 1–3 work on pre-cut clips. To score a full recording end to end:

```bash
python session_eval.py --session Data/RN208/11-20-2023 \
    --video_ckpt output/my_run/best.pt --video_arch slowfast \
    --eeg_ckpt   output/my_eeg_run/best.pt --eeg_arch gru \
    --fusion mean
```

Writes `session_eval/session_eval.json`, `trace.npz` and a per-session log into
the session folder. Both modalities are placed on **one absolute wall clock**
from the video XML and the EDF `meas_date`, so they are directly comparable.

Smoke-test it first with `--max_videos 1 --max_seconds 240`.

Plot the result:

```bash
python make_session_timeline.py \
    --trace Data/RN208/11-20-2023/session_eval/trace.npz \
    --out timeline.pdf --smooth 5
```

### Stage 5 — Tables and figures

```bash
python make_tab_vidarch_meansd.py   # video backbone comparison, mean ± sd over seeds
python make_tab_eegarch.py          # EEG backbone comparison
python make_tab1_meansd.py          # headline results table
python make_degrade_fig.py          # granularity degradation figure
```

The paper lives in `tex_wacv/`. Build it with
[tectonic](https://tectonic-typesetting.github.io/):

```bash
cd tex_wacv && tectonic -X compile main.tex
```

---

## Script reference

| Script | Does |
|---|---|
| `cut_seizure_clips.py` | Cut a clip per annotated seizure |
| `cut_non_seizure_clips.py` | Cut a duration-matched negative per seizure |
| `crop_clips.py` | Crop clips to one animal's cage |
| `build_frame_cache.py` | Pre-decode video into a uint8 memmap |
| `build_stage_segments_pooled.py` | Build EEG windows from clip EDFs |
| `train_pooled.py` | Train any of 8 video backbones |
| `train_pooled_eeg.py` | Train any of 5 EEG models |
| `train_pooled_eeg_classical.py` | RF / XGBoost baselines on EEG features |
| `train_pooled_videomae.py`, `train_pooled_timesformer.py` | Transformer backbones |
| `session_eval.py` | Score a whole uncut session, both modalities |
| `make_session_timeline.py` | Plot video vs EEG confidence over a session |
| `sweep_session.py` | Older video-only session sweep |
| `edf_clip_writer.py` | Write a clip EDF without destroying the waveform |
| `recut_clip_edfs.py` | Repair clip EDFs written by the old broken path |
| `quantify_clip_edf.py` | Audit clip EDFs for quantization damage |
| `analyze_failures.py`, `motion_vs_error.py`, `compute_mcc.py` | Analysis |
| `make_tab*.py`, `make_*_fig.py` | Paper tables and figures |

Orchestration drivers live in `sh/`. Superseded code is in `attic/` — see
[`attic/ATTIC.md`](attic/ATTIC.md) for what each file was and why it was retired.

---

## Things that will bite you

**1. Never rename `data/`.** All 24,497 rows of `cache_frames/*/index.json` are
keyed by the repo-root-relative string `data/Data_RN*_cropped/.../video.mp4`, and
the same strings are frozen into every `val_preds.npz`. A rename **fails
silently**: the cache misses, the loader gets a missing path, and the fill branch
substitutes a constant grey block — so a 12-epoch run completes and writes a
perfectly normal-looking `results.json` trained on grey. Use a symlink named
exactly `data` if it must live elsewhere.

**2. Training is not reproducible by default.** Setting `--seed` is not enough.
MViT in particular diverges from itself: two byte-identical runs at the same seed
differed by **0.0138 macro-F1 after one epoch**. Use `--deterministic` if you need
reproducibility, and report a distribution rather than a point value. Other
backbones (X3D, SlowFast) do reproduce in practice.

**3. Two large backbones do not fit on one 24 GB card.** Video Swin at 32×224
batch 4 measures 15.3 GB. The drivers cap concurrency at one job per GPU.

**4. `systemd-oomd` kills long jobs started from a terminal.** Ubuntu ships
`ManagedOOMMemoryPressure=kill` at 50% PSI for `user@1000.service`, so anything a
login shell starts is a candidate. `setsid` does **not** help — it changes the
session, not the cgroup. Launch long runs from **cron**, which lives in
`system.slice` and is exempt.

**5. `edfio` is hard-pinned.** `mne.export.export_raw` writes one physical range
shared across all channels; on these files a single 16-bit step exceeded the whole
biopotential amplitude and crushed **69% of clips** to ~4 unique values. The files
still open — the waveform is just gone. `edf_clip_writer.py` fixes this and
depends on the edfio 0.4.x API exactly.

**6. `cut_seizure_clips.py` parses `sys.argv` at import time.** Importing it from
another script hijacks that script's `--help`. Stub `sys.argv` around the import.

---

## Repository layout

```
.
├── *.py                    live pipeline, training, eval and analysis code
├── sh/                     orchestration drivers (queues, sweeps, cron jobs)
├── tex_wacv/               the paper: sources, figures, main.pdf
├── attic/                  superseded code, with ATTIC.md explaining each file
├── requirements.txt        pinned, verified against a working environment
├── SETUP.md                environment setup and prerequisites
├── CLAUDE.md               deeper notes on internals and conventions
│
├── data/                   raw + cropped recordings      (gitignored)
├── cache_frames/           pre-decoded frame caches      (gitignored, 110 GB)
├── output/                 run outputs                   (gitignored)
└── logs/                   run logs                      (gitignored)
```

Code stays flat at the repository root on purpose: 58 shell scripts gate GPU
concurrency by `pgrep`-ing the literal string `python3 train_pooled.py`, and those
gates **fail open**. Moving the trainers into a package would silently uncap
concurrency and OOM the cards.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA out of memory` | Two big backbones on one card, or `--batch_size` too high. One job per GPU; Swin needs batch 4. |
| Training dies with no traceback, no error | `systemd-oomd`. Check `journalctl \| grep systemd-oomd`. Relaunch from cron. |
| `results.json` looks normal but accuracy is near chance | Probably training on the grey fill — check that `data/` resolves and the cache index matches. |
| `assert meta["frames"] == frames` | The cache geometry does not match `--frames`/`--size`. Build the right cache. |
| `KeyError: 'clock'` or every seizure at second 0 | You are running the archived `sweep_session_eeg.py`. Use `session_eval.py`. |
| `--help` prints another script's options | Import-time `parse_args()` in `cut_seizure_clips.py`; stub `sys.argv`. |
| Model reproduces a different number than the paper | Expected for MViT — see bite #2. Compare against the seed distribution. |
| `UnpicklingError` loading a checkpoint | torch ≥ 2.6 flipped `weights_only` to `True`. The pin is `torch==2.4.0`. |

---

## Citation

The accompanying paper is in `tex_wacv/`. Model checkpoints are distributed as
GitHub Release assets rather than in the repository, since they exceed GitHub's
100 MB per-file limit.
