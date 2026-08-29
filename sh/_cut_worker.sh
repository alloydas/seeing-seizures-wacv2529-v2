#!/bin/bash
# Cut one raw session into <out_root>/<name>/, logging to <logdir>/<name>.log
# Args: <raw_session_dir> <out_root> <logdir>
raw="$1"; out_root="$2"; logdir="$3"
cd /path/to/repo
mkdir -p "$logdir"
name="$(basename "$raw")"
out="$out_root/$name"
log="$logdir/${name}.log"
{
  echo "[$(date +%T)] START $name"
  # 90-min guard per step: a multi-day EDF can hang MNE indefinitely on a single
  # window (seen on RN210 11-30..12-04, ~35 h stuck), which blocks the whole
  # xargs pool. timeout aborts the hang; the session just gets no DONE marker and
  # can be resumed later. Legit large sessions finish well under this.
  timeout 5400 python3 cut_seizure_clips.py     --session "$raw" --output "$out" --pre 10 --post 10 --mode copy \
    || echo "[$(date +%T)] cut_seizure TIMEOUT/err ($?) $name"
  timeout 5400 python3 cut_non_seizure_clips.py --session "$raw" --output "$out" --pre 10 --post 10 --safety 30 --seed 42 --mode copy \
    || echo "[$(date +%T)] cut_non_seizure TIMEOUT/err ($?) $name"
  sz=$(find "$out" -maxdepth 1 -type d -name 'seizure_*' 2>/dev/null | wc -l)
  cl=$(find "$out" -maxdepth 1 -type d -name 'clip_*_vs_seizure_*' 2>/dev/null | wc -l)
  echo "[$(date +%T)] DONE $name : $sz seizure, $cl non-seizure clips"
} > "$log" 2>&1
