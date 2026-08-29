#!/bin/bash
# Fill in straggler crops: clips cut AFTER _crop_all.sh ran are not yet cropped.
# crop_clips.py skips files that already exist (no --overwrite here), so this only
# processes the new clips -- it will NOT re-encode the ~10k already done.
# Geometry is the SAME per-subject table verified in crop_verify/ (do not change).
cd /path/to/repo
W=12

#  subj   input          output               wf    xf    hf    yf
JOBS=(
  "RN197 Data           Data_cropped          0.60 0.00 0.80 0.00"  # side-by-side left
  "RN213 Data_RN213     Data_RN213_cropped    0.40 0.60 0.80 0.00"  # side-by-side right
  "RN222 Data_RN222     Data_RN222_cropped    0.60 0.00 0.80 0.00"  # side-by-side left
  "RN237 Data_RN237     Data_RN237_cropped    0.60 0.00 0.80 0.00"  # side-by-side left
  "RN204 Data_RN204     Data_RN204_cropped    0.40 0.60 0.80 0.00"  # side-by-side right
  "RN235 Data_RN235     Data_RN235_cropped    0.40 0.60 0.80 0.00"  # side-by-side right
  "RN238 Data_RN238     Data_RN238_cropped    0.40 0.60 0.80 0.00"  # side-by-side right
  "RN199 Data_RN199     Data_RN199_cropped    0.50 0.00 0.60 0.30"  # 2x2 lower-left
  "RN245 Data_RN245     Data_RN245_cropped    0.50 0.00 0.60 0.30"  # 2x2 lower-left
  "RN216 Data_RN216     Data_RN216_cropped    0.48 0.52 0.60 0.30"  # 2x2 lower-right
)

for j in "${JOBS[@]}"; do
  read -r s in out wf xf hf yf <<< "$j"
  [ -d "$in" ] || { echo "SKIP $s (no $in)"; continue; }
  src=$(find "$in" -name '*.mp4' | wc -l)
  dst=$(find "$out" -name '*.mp4' 2>/dev/null | wc -l)
  miss=$((src - dst))
  if [ "$miss" -le 0 ]; then echo "$s: up to date ($dst/$src)"; continue; fi
  echo "======== $s : $miss new clips -> $out  $(date +%T) ========"
  python3 crop_clips.py --input "$in" --output "$out" \
      --left_frac "$wf" --x_frac "$xf" --top_frac "$hf" --y_frac "$yf" \
      --workers $W --threads 2 --crf 18 --preset fast \
      2>&1 | grep -vE '^\s*[0-9]+%|vid/s'
  echo "  $s now: $(find "$out" -name '*.mp4' | wc -l)/$src  $(date +%T)"
done
echo "==== FILL COMPLETE $(date +%T) ===="
