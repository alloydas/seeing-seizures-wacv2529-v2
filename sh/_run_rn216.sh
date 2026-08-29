#!/bin/bash
# Clip-cutting for RN216 from back_up_1, same pattern as the other subjects.
# Processes sessions whose log lacks a "DONE <name>" marker. xargs -P 16.
cd /path/to/repo
BK=/path/to/archive
JOBS=16
r=216; logdir=logs_RN216; bk="$BK/RN$r"; out="Data_RN$r"
mkdir -p "$logdir" "$out"
pending=()
while IFS= read -r d; do
  name="$(basename "$d")"
  log="$logdir/$name.log"
  if [ -f "$log" ] && grep -q "DONE $name" "$log"; then continue; fi
  pending+=("$d")
done < <(find "$bk" -mindepth 1 -maxdepth 1 -type d | sort)
echo "======== RN$r : ${#pending[@]} pending session(s) -> $out (logs: $logdir) $(date +%T) ========"
if [ ${#pending[@]} -gt 0 ]; then
  printf '%s\0' "${pending[@]}" \
    | xargs -0 -P "$JOBS" -I{} bash _cut_worker.sh {} "$out" "$logdir"
fi
sz=$(find "$out" -maxdepth 2 -type d -name 'seizure_*' | wc -l)
cl=$(find "$out" -maxdepth 2 -type d -name 'clip_*_vs_seizure_*' | wc -l)
echo "  RN$r done: $sz seizure, $cl non-seizure clips total in $out"
echo "==== RN216 COMPLETE $(date +%T) ===="
