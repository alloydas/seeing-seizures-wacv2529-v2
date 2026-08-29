# Environment setup

Verified 2026-08-25 on Ubuntu, CPython 3.12.2, 2x NVIDIA TITAN RTX (24.5 GB),
driver 595.84, torch CUDA 12.1.

## Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt
```

or `conda env create -f environment.yml`.

## Non-pip prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA driver | 595.84 here; needs to support CUDA 12.1 for the pinned torch |
| ffmpeg | **not** required separately — `imageio-ffmpeg` ships ffmpeg 7.0.2-static and the cutters resolve it via `imageio_ffmpeg.get_ffmpeg_exe()`. Install a system ffmpeg only if you want to call it directly. |
| Disk | the frame caches are large: `cache_frames/f32s224` is 110 GB, `f16s224` is 55 GB. Both are regenerable with `build_frame_cache.py`. |
| GPU memory | Video Swin at 32x224 batch 4 measures ~15.3 GB, so two Swin jobs do not fit on one 24.5 GB card. The drivers cap concurrency at 2 (one per card) for this reason. |

## Data layout

`data/` **must keep that exact name.** All 24,497 rows of
`cache_frames/*/index.json` are keyed by the repo-root-relative string
`data/Data_RN*_cropped/.../video.mp4`, and the same strings are frozen into every
`output/*/val_preds.npz`. A rename fails *silently*: the cache misses, `load_clip`
gets a missing path, returns `None`, and the fill branch substitutes a constant
grey block — a 12-epoch run then completes and writes a normal-looking
`results.json` trained on grey. If it must move, use a symlink named exactly `data`.
