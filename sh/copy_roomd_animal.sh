#!/usr/bin/env bash
# Copy one RoomD animal from the Element drive to the back_up_1 staging area,
# parallelising across session directories. Resumable: rsync skips files that
# already match, so re-running finishes an interrupted copy.
#
# Usage: sh/copy_roomd_animal.sh RN208 [JOBS]
set -euo pipefail

ANIMAL="${1:?usage: copy_roomd_animal.sh ANIMAL [JOBS]}"
JOBS="${2:-16}"
SRC="/path/to/archive$ANIMAL"
DST="/path/to/archive$ANIMAL"
LOG="/path/to/repo/logs/copy_roomd/$ANIMAL"

[ -d "$SRC" ] || { echo "no such source: $SRC" >&2; exit 1; }
mkdir -p "$DST" "$LOG"

# one rsync per session subdir, fanned out with xargs -P
export SRC DST LOG
copy_one() {
  d="$1"; name="$(basename "$d")"
  rsync -a --partial "$d/" "$DST/$name/" \
    > "$LOG/$name.log" 2>&1 \
    && echo "ok   $name" \
    || echo "FAIL $name"
}
export -f copy_one

echo "[$ANIMAL] $(ls -d "$SRC"/*/ | wc -l) session dirs -> $DST  (jobs=$JOBS)"
ls -d "$SRC"/*/ | tr '\n' '\0' \
  | xargs -0 -P "$JOBS" -I{} bash -c 'copy_one "$@"' _ {} \
  | tee "$LOG/_summary.txt"

# verification: regular-file count + regular-file bytes, source vs dest.
# Compare only files: directory inode sizes differ across filesystems (exFAT vs
# ext4), and du would report a spurious mismatch. awk sums as double + %.0f
# because the totals exceed 2^31.
src_files=$(find "$SRC" -type f | wc -l)
dst_files=$(find "$DST" -type f | wc -l)
src_bytes=$(find "$SRC" -type f -printf '%s\n' | awk '{s+=$1}END{printf "%.0f",s}')
dst_bytes=$(find "$DST" -type f -printf '%s\n' | awk '{s+=$1}END{printf "%.0f",s}')
echo "[$ANIMAL] files src=$src_files dst=$dst_files   bytes src=$src_bytes dst=$dst_bytes"
if [ "$src_files" = "$dst_files" ] && [ "$src_bytes" = "$dst_bytes" ]; then
  echo "[$ANIMAL] VERIFIED complete"
else
  echo "[$ANIMAL] MISMATCH -- re-run to finish" >&2
  exit 2
fi
