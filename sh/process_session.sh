#!/bin/bash
# Process a RAW session folder (.mp4 + .edf + .xlsx [+ .XML]) into the flat
# Data/<name>/ layout the classifiers expect: seizure_*/ and clip_*_vs_seizure_*/
# clip dirs side-by-side, each with video.mp4 + eeg.edf + info.txt.
#
# Usage:
#   bash process_session.sh /path/to/raw_session [output_name]
#   bash process_session.sh --batch /path/to/root      # each subdir -> Data/<subdir>
#
# Buffers/seed match the existing dataset (pre=10 post=10 safety=30 seed=42, copy mode).

set -euo pipefail
cd /path/to/repo
PRE=10; POST=10; SAFETY=30; SEED=42; MODE=copy
DATA_ROOT=/path/to/repo/Data

process_one() {
  local raw="$1" name="$2"
  local out="$DATA_ROOT/$name"
  echo "======== $name ========"
  echo "  raw:    $raw"
  echo "  output: $out"
  python3 cut_seizure_clips.py     --session "$raw" --output "$out" \
      --pre $PRE --post $POST --mode $MODE
  python3 cut_non_seizure_clips.py --session "$raw" --output "$out" \
      --pre $PRE --post $POST --safety $SAFETY --seed $SEED --mode $MODE
  local sz cl
  sz=$(find "$out" -maxdepth 1 -type d -name 'seizure_*' | wc -l)
  cl=$(find "$out" -maxdepth 1 -type d -name 'clip_*_vs_seizure_*' | wc -l)
  echo "  -> $sz seizure clips, $cl non-seizure clips in $out"
}

if [ "${1:-}" = "--batch" ]; then
  root="${2:?--batch needs a root dir}"
  for d in "$root"/*/; do
    [ -d "$d" ] || continue
    process_one "${d%/}" "$(basename "${d%/}")"
  done
else
  raw="${1:?usage: process_session.sh RAW_SESSION_DIR [output_name]}"
  name="${2:-$(basename "$raw")}"
  process_one "$raw" "$name"
fi
echo "DONE. Verify with:  python3 -c \"import sys;sys.path.insert(0,'.');from train_classifier import discover_clips,split_by_session as s;i=discover_clips('Data');print(len(i),'clips',len({x[2] for x in i}),'sessions')\""
