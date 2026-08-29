#!/bin/bash
# Second pass of the clip-EDF re-cut, to recover the 524 clips that failed pass 1 with
#   ValueError: New channel names are not unique, renaming failed
# caused by naive 16-char truncation colliding two channels ('ECG [FIR-HP: 5Hz]' vs
# 'ECG [FIR-HP: 5.0Hz]'). recut_clip_edfs.py now carries the same collision-avoiding
# rename that cut_seizure_clips.py already had.
#
# WAITS for pass 1 to exit first. Both passes walk the same clip directories, and two
# processes writing the same eeg.edf concurrently would produce a torn file -- worse
# than the defect being repaired. Pass 1 already attempted the failures (its error count
# has been flat since the 1000-clip mark), so nothing is lost by waiting.
#
# Pass 2 is cheap: recut_clip_edfs.py skips any clip whose EDF already has >=100 unique
# values, so it only revisits failures and genuinely degenerate files.
set -u
cd /path/to/repo
log(){ echo "[$(date +%F_%T)] $*"; }

log "waiting for pass 1 to finish"
while pgrep -f "[r]ecut_clip_edfs.py" > /dev/null; do sleep 60; done
log "pass 1 done -- starting pass 2"

python3 recut_clip_edfs.py --workers 6 > logs/recut_clip_edfs_pass2.log 2>&1
log "pass 2 exited rc=$?"
tail -12 logs/recut_clip_edfs_pass2.log

log "=== re-quantifying clip quality after repair ==="
python3 quantify_clip_edf.py > logs/clip_edf_quality_after.log 2>&1
sed -n '/=== overall/,/per subject/p' logs/clip_edf_quality_after.log
