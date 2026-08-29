#!/bin/bash
# Crop every remaining subject to ITS OWN cage. Geometry is per-camera and was
# verified frame-by-frame (see crop_verify/):
#   side-by-side cameras -> target occupies top 80%
#   2x2 grid cameras     -> target is on the LOWER shelf: y offset 0.30, height 0.60
# Getting the side/shelf wrong silently poisons the labels, so do not "simplify" this table.
cd /path/to/repo
W=12   # ffmpeg workers

# subj  wf    xf    hf    yf
JOBS=(
  "RN199 0.50 0.00 0.60 0.30"   # 2x2 lower-left  (above: RN222, right: RN201)
  "RN245 0.50 0.00 0.60 0.30"   # 2x2 lower-left  (above: RN197, right: RN216)
  "RN216 0.48 0.52 0.60 0.30"   # 2x2 lower-right (above: RN213, left: RN245)
  "RN222 0.60 0.00 0.80 0.00"   # side-by-side left
  "RN237 0.60 0.00 0.80 0.00"   # side-by-side left
  "RN204 0.40 0.60 0.80 0.00"   # side-by-side right
  "RN235 0.40 0.60 0.80 0.00"   # side-by-side right
  "RN238 0.40 0.60 0.80 0.00"   # side-by-side right
)

for j in "${JOBS[@]}"; do
  read -r s wf xf hf yf <<< "$j"
  in="Data_$s"; out="Data_${s}_cropped"
  [ -d "$in" ] || { echo "SKIP $s (no $in)"; continue; }
  n=$(find "$in" -name '*.mp4' | wc -l)
  echo "======== $s : $n clips  w=$wf x=$xf h=$hf y=$yf -> $out  $(date +%T) ========"
  python3 crop_clips.py --input "$in" --output "$out" \
      --left_frac "$wf" --x_frac "$xf" --top_frac "$hf" --y_frac "$yf" \
      --workers $W --threads 2 --crf 18 --preset fast --overwrite \
      2>&1 | grep -vE '^\s*[0-9]+%|vid/s'
  echo "  $s done: $(find "$out" -name '*.mp4' | wc -l) cropped  $(date +%T)"
done
echo "==== ALL CROPS COMPLETE $(date +%T) ===="
