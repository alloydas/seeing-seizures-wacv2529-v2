#!/bin/bash
# Detached watcher: fires a desktop notification when phase A of the video-seeding sweep
# finishes. Survives the terminal it was launched from, which is the whole point -- the
# previous driver died at 15:56 on 2026-08-20 because it went down with its shell.
#
# WHY PID AND NOT pgrep BY NAME. sh/_phase_b.sh ends with `exec bash sh/_v3_vidseeds.sh`,
# so the moment phase B starts running its own cells, a name-based pgrep matches it too
# and this watcher would wait for phase B instead -- i.e. never fire on phase A. So we
# watch one specific pid and re-verify its cmdline on every poll, which also means a
# recycled pid cannot masquerade as the driver.
set -u
cd /path/to/repo
PID=${1:?usage: _notify_phase_a.sh <phase-A driver pid>}
MARK=logs/v3_vidseeds/_phase_a_done.txt

# A detached process inherits no desktop session, so notify-send has no bus to talk to
# unless we hand it one. Values probed from the running desktop session on 2026-08-21.
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XDG_RUNTIME_DIR=/run/user/1000
export DISPLAY=:1

alive(){
  [ -r "/proc/$PID/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q "_v3_vidseeds\.sh"
}

while alive; do sleep 60; done

# ---- summarise what phase A actually finished ----------------------------------
SUMMARY=$(python3 - <<'PY' 2>/dev/null || echo "phase A finished (count unavailable)"
import os
A = ["r2plus1d","mvit","mvit_v1","x3d","s3d","videomae","tsf"]
cells = [(a,t,s) for a in A for t in ("bin","g3","g5") for s in (1,2,3,5)]
done = [c for c in cells if os.path.exists("output/v3_vidseeds/%s_%s_s%d/results.json" % c)]
miss = [c for c in cells if c not in done]
line = "phase A done: %d/%d cells" % (len(done), len(cells))
if miss:
    line += " -- still missing: " + ", ".join("%s_%s_s%d" % c for c in miss[:6])
    if len(miss) > 6:
        line += " (+%d more)" % (len(miss) - 6)
print(line)
PY
)

{
  echo "[$(date +%F_%T)] phase A driver (pid $PID) exited"
  echo "$SUMMARY"
} >> "$MARK"

# Best-effort across channels: notify-send is silent if no notification daemon is up, so
# fall back to wall, which reaches any open terminal. Record what we managed to send.
notify-send -u critical -t 0 "EEG sweep: phase A finished" "$SUMMARY" \
  && echo "  notified via notify-send" >> "$MARK" \
  || echo "  notify-send FAILED" >> "$MARK"

echo "EEG sweep: phase A finished -- $SUMMARY" | wall 2>/dev/null \
  && echo "  notified via wall" >> "$MARK"
