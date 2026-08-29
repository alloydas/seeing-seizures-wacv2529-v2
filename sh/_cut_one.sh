#!/bin/bash
# Cut one raw session into <out_root>/<name>/ (seizure + non-seizure), logging to logs/<name>.log
raw="$1"; out_root="$2"
cd /path/to/repo
name="$(basename "$raw")"
out="$out_root/$name"
log="logs/${name}.log"
{
  echo "[$(date +%T)] START $name"
  python3 cut_seizure_clips.py     --session "$raw" --output "$out" --pre 10 --post 10 --mode copy
  python3 cut_non_seizure_clips.py --session "$raw" --output "$out" --pre 10 --post 10 --safety 30 --seed 42 --mode copy
  sz=$(find "$out" -maxdepth 1 -type d -name 'seizure_*' 2>/dev/null | wc -l)
  cl=$(find "$out" -maxdepth 1 -type d -name 'clip_*_vs_seizure_*' 2>/dev/null | wc -l)
  echo "[$(date +%T)] DONE $name : $sz seizure, $cl non-seizure clips"
} > "$log" 2>&1
