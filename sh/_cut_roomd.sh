#!/bin/bash
# Cut clips for every RoomD subject from the back_up_1 staging area.
# Annotation-driven: sessions with 0 annotated seizures produce 0 clips.
# Per subject: each session -> Data_<SUBJECT>/<session>/ via _cut_worker.sh,
# fanned out at -P 16. Resumable: sessions whose log already has "DONE <name>"
# are skipped, so re-running finishes an interrupted subject.
#
# Usage: sh/_cut_roomd.sh [SUBJECT ...]   (default: all 12 RoomD subjects)
cd /path/to/repo
BK=/path/to/archive
JOBS=${JOBS:-16}
SUBJECTS=("$@")
[ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(RN208 RN210 RN215 RN219 RN223 RN224 RN227 RN229 RN242 RN243 RN244 RN245)

MASTER=logs/cut_roomd/_master.log
mkdir -p logs/cut_roomd
echo "=== RoomD cut start $(date +%F_%T)  subjects=${SUBJECTS[*]}  jobs=$JOBS ===" | tee -a "$MASTER"

for s in "${SUBJECTS[@]}"; do
  bk="$BK/$s"; out="Data_$s"; logdir="logs/cut_roomd/$s"
  [ -d "$bk" ] || { echo "SKIP $s (no $bk)" | tee -a "$MASTER"; continue; }
  mkdir -p "$out" "$logdir"
  pending=()
  while IFS= read -r d; do
    name="$(basename "$d")"
    log="$logdir/$name.log"
    [ -f "$log" ] && grep -q "DONE $name" "$log" && continue
    pending+=("$d")
  done < <(find "$bk" -mindepth 1 -maxdepth 1 -type d | sort)

  echo "--- $s : ${#pending[@]} pending session(s) -> $out ---" | tee -a "$MASTER"
  [ ${#pending[@]} -eq 0 ] && { echo "  $s nothing to do" | tee -a "$MASTER"; continue; }
  printf '%s\0' "${pending[@]}" \
    | xargs -0 -P "$JOBS" -I{} bash sh/_cut_worker.sh {} "$out" "$logdir"

  sz=$(find "$out" -maxdepth 2 -type d -name 'seizure_*' | wc -l)
  cl=$(find "$out" -maxdepth 2 -type d -name 'clip_*_vs_seizure_*' | wc -l)
  echo "--- $s DONE : $sz seizure, $cl non-seizure clips in $out ---" | tee -a "$MASTER"
done
echo "=== RoomD cut finished $(date +%F_%T) ===" | tee -a "$MASTER"
