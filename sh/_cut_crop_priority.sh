#!/bin/bash
# Priority processing for specific RoomD subjects, capped at ONE shared pool of
# 16 workers total (not 16 per subject). Sessions are queued in subject order,
# so the first subject's sessions start immediately and later subjects fill idle
# slots as workers free up -- keeping utilisation at ~16 without oversubscribing.
# Resumable (skips sessions already marked DONE). After all cuts finish, crops
# each subject. Runs independently of the main _cut_roomd.sh batch; the main
# batch will SKIP these subjects when it reaches them (their sessions carry DONE
# markers by then).
#
# Usage: sh/_cut_crop_priority.sh RN242 RN229 [...]
cd /path/to/repo
BK=/path/to/archive
JOBS=${JOBS:-16}
SUBJECTS=("$@")
MASTER=logs/cut_roomd/_priority.log
mkdir -p logs/cut_roomd
echo "=== PRIORITY start $(date +%F_%T)  subjects=${SUBJECTS[*]}  shared pool=$JOBS ===" | tee -a "$MASTER"

# build one combined pending list, in subject (priority) order
pending=()
for s in "${SUBJECTS[@]}"; do
  mkdir -p "logs/cut_roomd/$s" "Data_$s"
  cnt=0
  while IFS= read -r d; do
    name="$(basename "$d")"; log="logs/cut_roomd/$s/$name.log"
    [ -f "$log" ] && grep -q "DONE $name" "$log" && continue
    pending+=("$s|$d"); cnt=$((cnt+1))
  done < <(find "$BK/$s" -mindepth 1 -maxdepth 1 -type d | sort)
  echo "  $s: $cnt pending session(s)" | tee -a "$MASTER"
done

echo "--- cutting ${#pending[@]} sessions across a $JOBS-worker pool $(date +%T) ---" | tee -a "$MASTER"
printf '%s\0' "${pending[@]}" \
  | xargs -0 -P "$JOBS" -I{} bash -c '
      IFS="|" read -r s d <<< "$1"
      bash sh/_cut_worker.sh "$d" "Data_$s" "logs/cut_roomd/$s"' _ {}

# per-subject cut totals, then crop
for s in "${SUBJECTS[@]}"; do
  sz=$(find "Data_$s" -maxdepth 2 -type d -name 'seizure_*' | wc -l)
  cl=$(find "Data_$s" -maxdepth 2 -type d -name 'clip_*_vs_seizure_*' | wc -l)
  echo "--- $s CUT DONE : $sz seizure, $cl non-seizure $(date +%T) ---" | tee -a "$MASTER"
  echo "--- $s CROP start $(date +%T) ---" | tee -a "$MASTER"
  bash sh/_crop_roomd.sh "$s" >> "logs/crop_roomd/$s.log" 2>&1
  cr=$(find "data/Data_${s}_cropped" -name video.mp4 2>/dev/null | wc -l)
  echo "--- $s CROP DONE : $cr cropped $(date +%T) ---" | tee -a "$MASTER"
done
echo "=== PRIORITY finished $(date +%F_%T) ===" | tee -a "$MASTER"
