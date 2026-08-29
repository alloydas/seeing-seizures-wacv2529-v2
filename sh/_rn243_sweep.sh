#!/bin/bash
# RN243 long-timeframe evaluation -- the held-out 21st animal.
#
# RN243 was never cut, trained on, or evaluated: it is one of three animals outside the
# 20-subject cohort (the other two, RN201/RN203, have empty annotation files). It has
# 6 annotated seizures across 2 of its 48 sessions, and ~31 h of continuous video in
# those two sessions -- so it is a genuine unseen-animal test at a realistic duty cycle
# rather than a curated clip set.
#
#   10-11-2023(2)  2 seizures, 3 videos
#   10-16-2023     4 seizures, 4 videos (~24 h)
#
# Sweeps the windowed R(2+1)D (output/video_win_out_W4/last.pt) across the UNCUT
# session with 4 s windows / 2 s stride and scores detected intervals against the xlsx.
# Uncut on purpose: every clip in Data_* was cut with Pre-buffer=10 s exactly, so onset
# always sits at t=10 s and any within-clip onset metric would measure the cutter.
#
# GPU1 with batch 16: the default batch 64 on GPU0 hit the NVML assert at 11:xx because
# the Table 2 EEG sweep holds 3 contexts per GPU.
#
# Crop: RN243 is RoomD (RN242-RN243 camera, lower-right cage). sh/_crop_roomd.sh stores
# its geometry in a different field order than sh/_crop_all.sh, so the CROPS entry added
# to sweep_session.py is transposed and was verified against a decoded frame.
set -u
cd /path/to/repo
L=logs/rn243; mkdir -p "$L" output/rn243
log(){ echo "[$(date +%F_%T)] $*"; }
BASE="/path/to/archive"

for sess in "10-16-2023" "10-11-2023(2)"; do
  tag=$(echo "$sess" | tr -d '()' | tr ' ' '_')
  out="output/rn243/video_${tag}"
  if [ -f "$out/results.json" ]; then log "SKIP video $sess (done)"; continue; fi
  log "START video sweep: $sess"
  python3 sweep_session.py --session "$BASE/$sess" \
      --ckpt output/video_win_out_W4/last.pt --subject RN243 \
      --out "$out" --batch_size 16 --device cuda:1 \
      > "$L/video_${tag}.log" 2>&1
  if [ -f "$out/results.json" ]; then log "DONE video $sess"; else
    log "FAILED video $sess -- tail:"; tail -5 "$L/video_${tag}.log"; fi
done

log "===== RN243 video sweeps complete ====="
