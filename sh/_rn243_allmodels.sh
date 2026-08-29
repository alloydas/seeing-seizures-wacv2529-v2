#!/bin/bash
# RN243 held-out long-timeframe sweep, ALL VIDEO BACKBONES.
#
# Extends the 2026-08-11 R(2+1)D result (79 FP/h @thr 0.5 over 24 h, recall 0.50) to
# every video backbone that has a detection checkpoint, so the deployment question --
# "which backbone gives a usable false-alarm rate on continuous unseen-animal video?"
# -- is answered across the field rather than for one model.
#
# CONFOUND, stated up front: only video_win_out_W4 (r2plus1d_win) was TRAINED on 4 s
# windows. The other eight are whole-clip models -- 16 frames spread across a whole
# ~40-80 s clip -- applied here to 4 s windows, i.e. the same 16 frames over a 10-20x
# shorter span. They are therefore out of distribution and will likely look worse than
# they are. The window is held identical across all nine so they remain comparable to
# each other and to the existing R(2+1)D number; a clean cross-backbone comparison would
# need windowed retraining of each (8 trainings, ~4 h apiece).
#
# EEG IS DELIBERATELY NOT INCLUDED. RN243's EDF carries a real EEG channel while every
# EEG model was trained on ~96.5% ECG (19/20 cohort subjects are ECG; only RN219 is EEG).
# The GRU sweep already returned recall 0.000 at every threshold for exactly this reason.
# Running the other six EEG backbones would reproduce the same modality mismatch six more
# times and produce six more uninterpretable numbers. See memory: no-eeg-channel-only-ecg.
#
# ~35 min per backbone on the 24 h session (decode-bound), so ~5 h for the eight.
# GPU0: the Table 2 sweep holds GPU1.
set -u
cd /path/to/repo
L=logs/rn243; mkdir -p "$L" output/rn243
log(){ echo "[$(date +%F_%T)] $*"; }
SESS="/path/to/archive"

# arch : checkpoint
run_one(){
  local arch=$1 ckpt=$2
  local out="output/rn243/video_${arch}_10-16-2023"   # separate stmt: under set -u,
                                                      # ${arch} is expanded before
                                                      # the first `local` assigns it
  if [ ! -f "$ckpt" ]; then log "SKIP $arch (no checkpoint $ckpt)"; return 0; fi
  if [ -f "$out/results.json" ]; then log "SKIP $arch (done)"; return 0; fi
  log "START $arch"
  python3 sweep_session.py --session "$SESS" --ckpt "$ckpt" --arch "$arch" \
      --subject RN243 --out "$out" --batch_size 16 --device cuda:0 \
      > "$L/video_${arch}.log" 2>&1
  if [ -f "$out/results.json" ]; then log "DONE $arch"; else
    log "FAILED $arch -- tail:"; tail -4 "$L/video_${arch}.log"; fi
}

run_one mvit     output/vid_mvit_bin/best.pt
run_one mvit_v1  output/vid_mvit_v1_bin/best.pt
run_one x3d      output/vid_x3d_bin/best.pt
run_one slowfast output/vid_slowfast_bin/best.pt
run_one s3d      output/vid_s3d_bin/best.pt
run_one swin     output/vid_swin_bin/best.pt
run_one swin_s   output/vid_swin_s_bin/best.pt

log "===== all-backbone RN243 sweep complete ====="
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob('output/rn243/video_*_10-16-2023/results.json')) + \
         ['output/rn243/video_10-16-2023/results.json']:
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    name = os.path.basename(os.path.dirname(f)).replace('video_', '').replace('_10-16-2023', '')
    if name == '10-16-2023':
        name = 'r2plus1d_win'
    for x in r['sweep']:
        if abs(x['thr'] - 0.5) < 1e-9:
            rows.append((name, x['recall'], x['fp_per_hour'], x['n_det']))
print(f"{'backbone':14s} {'recall@0.5':>11s} {'FP/h@0.5':>9s} {'detections':>11s}")
for n, rec, fph, nd in sorted(rows, key=lambda r: r[2]):
    print(f'{n:14s} {rec:11.2f} {fph:9.1f} {nd:11d}')
print('\n(only r2plus1d_win was trained at this 4 s window; the rest are whole-clip '
      'models applied out of distribution)')
PY
