#!/bin/bash
# Detached watcher: fires a desktop notification when phase B of the video-seeding sweep
# finishes (slowfast / swin / swin_s, the three 32-frame architectures).
#
# WHY THE CMDLINE PATTERN ACCEPTS TWO NAMES. sh/_phase_b.sh:60 ends with
# `exec bash sh/_v3_vidseeds.sh`, and exec keeps the pid while replacing the process
# image -- so this one pid is "bash sh/_phase_b.sh" during the wait and cache build, then
# "bash sh/_v3_vidseeds.sh" for the actual sweep. Matching only the first name would fire
# the instant phase B started its real work.
#
# WHY THE START-TIME GUARD. This watcher may wait a day or more (110 GB cache rebuild plus
# 36 cells at batch 4), which is long enough for the pid to be recycled. Field 22 of
# /proc/pid/stat is the process start time in clock ticks; pinning it means a recycled pid
# running some other bash cannot be mistaken for phase B.
set -u
cd /path/to/repo
PID=${1:?usage: _notify_phase_b.sh <phase-B pid>}
MARK=logs/v3_vidseeds/_phase_b_done.txt
BLOG=logs/v3_vidseeds/_phase_b_a3.log

export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XDG_RUNTIME_DIR=/run/user/1000
export DISPLAY=:1

# strip through "pid (comm) " so a comm containing spaces cannot shift the field index;
# after that strip, stat field 22 sits at position 20
starttime(){ sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
START0=$(starttime "$PID")
[ -n "$START0" ] || { echo "pid $PID not running at arm time" >&2; exit 1; }

alive(){
  [ -r "/proc/$PID/cmdline" ] || return 1
  [ "$(starttime "$PID")" = "$START0" ] || return 1
  tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null \
    | grep -qE "_phase_b\.sh|_v3_vidseeds\.sh"
}

while alive; do sleep 60; done

# ---- summarise ------------------------------------------------------------------
SUMMARY=$(python3 - <<'PY' 2>/dev/null || echo "phase B finished (count unavailable)"
import os
B   = ["slowfast","swin","swin_s"]
ALL = ["r2plus1d","mvit","mvit_v1","x3d","s3d","videomae","tsf"] + B
def count(archs):
    cells = [(a,t,s) for a in archs for t in ("bin","g3","g5") for s in (1,2,3,5)]
    done  = [c for c in cells
             if os.path.exists("output/v3_vidseeds/%s_%s_s%d/results.json" % c)]
    return done, cells
bd, bc = count(B)
ad, ac = count(ALL)
line = "phase B: %d/%d cells -- sweep total %d/%d" % (len(bd), len(bc), len(ad), len(ac))
miss = [c for c in bc if c not in bd]
if miss:
    line += " -- missing: " + ", ".join("%s_%s_s%d" % c for c in miss[:5])
    if len(miss) > 5:
        line += " (+%d more)" % (len(miss) - 5)
print(line)
PY
)

# phase B aborts rather than runs if the 110 GB cache build fails (sh/_phase_b.sh:49);
# that exits the same pid, so distinguish it or the ping would read as success
if grep -q "ABORT" "$BLOG" 2>/dev/null; then
  TITLE="EEG sweep: phase B ABORTED"
  SUMMARY="cache build failed -- $SUMMARY"
else
  TITLE="EEG sweep: phase B finished"
fi

{
  echo "[$(date +%F_%T)] phase B (pid $PID) exited"
  echo "$SUMMARY"
} >> "$MARK"

notify-send -u critical -t 0 "$TITLE" "$SUMMARY" \
  && echo "  notified via notify-send" >> "$MARK" \
  || echo "  notify-send FAILED" >> "$MARK"

echo "$TITLE -- $SUMMARY" | wall 2>/dev/null \
  && echo "  notified via wall" >> "$MARK"
