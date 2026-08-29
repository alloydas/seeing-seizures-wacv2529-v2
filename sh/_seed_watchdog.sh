#!/bin/bash
# Keeps the six-day seeding sweep alive. Checks every 10 min: if no driver is running and
# work remains, restarts the right phase. Individual run failures are already handled by
# one()'s three retries -- this covers the driver itself dying (which has happened: a
# killed subshell, an OOM, a stray signal), where the whole sweep would otherwise sit
# idle for hours unnoticed.
set -u
cd /path/to/repo
log(){ echo "[$(date +%F_%T)] $*"; }
A="r2plus1d mvit mvit_v1 x3d s3d videomae tsf"
B="slowfast swin swin_s"

remaining(){  # $1 = space-separated archs
  python3 - "$1" <<'PY'
import os, sys
n=0
for a in sys.argv[1].split():
    for t in ("bin","g3","g5"):
        for s in (1,2,3,5):
            if not os.path.exists(f"output/v3_vidseeds/{a}_{t}_s{s}/results.json"): n+=1
print(n)
PY
}

driver_up(){ pgrep -f "_v3_vidseeds\.sh" > /dev/null || pgrep -f "_restart_seeding_cached\.sh" > /dev/null \
             || pgrep -f "_phase_b\.sh" > /dev/null || pgrep -f "build_frame_cache\.py" > /dev/null; }

log "watchdog started"
while true; do
  sleep 600
  driver_up && continue
  ra=$(remaining "$A"); rb=$(remaining "$B")
  if [ "$ra" -gt 0 ]; then
    log "no driver alive and $ra phase-A runs remain -- restarting phase A"
    ARCH_FILTER="$A" setsid nohup bash sh/_v3_vidseeds.sh >> logs/v3_vidseeds/_driver.log 2>&1 &
  elif [ "$rb" -gt 0 ]; then
    log "phase A complete, $rb phase-B runs remain and no driver -- restarting phase B"
    setsid nohup bash sh/_phase_b.sh >> logs/v3_vidseeds/_phaseb.log 2>&1 &
  else
    log "all 120 runs complete -- watchdog exiting"
    python3 make_tab_vidarch_meansd.py 2>&1 | tail -14
    exit 0
  fi
done
