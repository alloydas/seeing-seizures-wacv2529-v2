#!/bin/bash
# Crop RoomD subjects to their OWN cage. Geometry is per-camera, read off a
# populated frame per camera and visually verified (scratchpad/boxes/*.jpg,
# 2026-07-18). Every RoomD camera films a 2x2 cage grid; the filename RNa-RNb
# names the two PRIMARY cages but their position in-frame differs per camera, so
# each animal has its own box. Wrong geometry silently poisons labels -- do not
# "simplify" this table.
#
# Fractions are crop_clips.py convention: x_frac/y_frac = top-left offset,
# left_frac = width kept, top_frac = height kept (all fractions of the source).
#
# Usage: sh/_crop_roomd.sh [SUBJECT ...]   (default: all croppable subjects)
cd /path/to/repo
W=8   # ffmpeg re-encode workers (coexist with the cutting job)

# subj   x_frac left_frac y_frac top_frac   camera / position
JOBS=(
  "RN208 0.00 0.43 0.33 0.39"   # RN208-RN223  lower-left
  "RN223 0.55 0.42 0.35 0.38"   # RN208-RN223  lower-right
  "RN210 0.10 0.43 0.42 0.38"   # RN210-RN224  lower-left
  "RN224 0.53 0.40 0.44 0.38"   # RN210-RN224  lower-right
  "RN215 0.00 0.42 0.20 0.35"   # RN215-RN219  mid-left
  "RN219 0.50 0.43 0.22 0.35"   # RN215-RN219  mid-right
  "RN227 0.00 0.40 0.42 0.33"   # RN227-RN244  mid-left
  "RN244 0.52 0.46 0.42 0.36"   # RN227-RN244  mid-right
  "RN242 0.06 0.44 0.44 0.36"   # RN242-RN243  lower-left
  "RN243 0.50 0.40 0.45 0.37"   # RN242-RN243  lower-right
  "RN229 0.00 0.45 0.22 0.35"   # RN229-RN245  mid-left
  # RN245: no cage in RoomD (empty platform, already scored in RoomC) -> skip
)

want=("$@")
in_want() { [ ${#want[@]} -eq 0 ] && return 0; for x in "${want[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }

for j in "${JOBS[@]}"; do
  read -r s xf wf yf hf <<< "$j"
  in_want "$s" || continue
  in="Data_$s"; out="data/Data_${s}_cropped"
  [ -d "$in" ] || { echo "SKIP $s (no $in)"; continue; }
  n=$(find "$in" -name '*.mp4' | wc -l)
  echo "======== $s : $n clips  x=$xf w=$wf y=$yf h=$hf -> $out  $(date +%T) ========"
  python3 crop_clips.py --input "$in" --output "$out" \
      --x_frac "$xf" --left_frac "$wf" --y_frac "$yf" --top_frac "$hf" \
      --workers $W --threads 2 --crf 18 --preset fast --overwrite \
      2>&1 | grep -vE '^\s*[0-9]+%|vid/s'
  echo "  $s done: $(find "$out" -name '*.mp4' | wc -l) cropped  $(date +%T)"
done
echo "==== crop run complete $(date +%T) ===="
