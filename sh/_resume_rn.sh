#!/bin/bash
# Resume clip-cutting for RN199/RN204/RN213 from back_up_1.
# For each subject, process only sessions whose log lacks a "DONE <name>" marker
# (missing or interrupted mid-run). Reuses _cut_worker.sh + xargs -P 16.
cd /path/to/repo
BK=/path/to/archive
JOBS=16

process_subject() {
  local r="$1" logdir="$2"
  local bk="$BK/RN$r" out="Data_RN$r"
  mkdir -p "$logdir" "$out"
  local pending=()
  while IFS= read -r d; do
    local name; name="$(basename "$d")"
    local log="$logdir/$name.log"
    if [ -f "$log" ] && grep -q "DONE $name" "$log"; then continue; fi
    pending+=("$d")
  done < <(find "$bk" -mindepth 1 -maxdepth 1 -type d | sort)

  echo "======== RN$r : ${#pending[@]} pending session(s) -> $out (logs: $logdir) ========"
  [ ${#pending[@]} -eq 0 ] && { echo "  nothing to do"; return; }
  printf '%s\0' "${pending[@]}" \
    | xargs -0 -P "$JOBS" -I{} bash _cut_worker.sh {} "$out" "$logdir"

  local sz cl
  sz=$(find "$out" -maxdepth 2 -type d -name 'seizure_*' | wc -l)
  cl=$(find "$out" -maxdepth 2 -type d -name 'clip_*_vs_seizure_*' | wc -l)
  echo "  RN$r done: $sz seizure, $cl non-seizure clips total in $out"
}

process_subject 199 logs
process_subject 204 logs_RN204
process_subject 213 logs_RN213
echo "==== ALL SUBJECTS COMPLETE $(date +%T) ===="
