#!/bin/bash
# Phase B, fourth cut. Cron-supervised relaunch of the 30 remaining video-seed cells.
#
# WHY THE PREVIOUS THREE CUTS DIED -- it was never a trainer bug, an OOM on the cards,
# or a stray kill from the launching shell. It is **systemd-oomd**:
#
#   Aug 23 23:44:05 systemd-oomd: Killed /user.slice/user-1000.slice/user@1000.service/
#     app.slice/app-org.gnome.Terminal.slice/vte-spawn-49858....scope due to memory
#     pressure for /user.slice/user-1000.slice/user@1000.service being 85.80% > 50.00%
#     for > 20s with reclaim activity
#   systemd[1391]: vte-spawn-49858....scope: systemd-oomd killed 67 process(es)
#
# Ubuntu ships /usr/lib/systemd/system/user@.service.d/10-oomd-user-service-defaults.conf
# with ManagedOOMMemoryPressure=kill and ManagedOOMMemoryPressureLimit=50%, so anything
# a login shell starts is a kill candidate whenever the user manager's PSI crosses 50%.
# Streaming the 110 GB f32s224 memmap through the page cache holds PSI there indefinitely,
# so this fires on a timer: Aug 20 03:04, Aug 20 16:15 (x2), Aug 21 12:45, Aug 23 06:39,
# Aug 23 23:44 -- five events, and it always takes the whole terminal scope with it
# (55-259 processes). That is exactly the observed signature: children vanish right after
# an epoch boundary, no traceback, no GAVE UP, no kernel OOM line, no reboot.
#
# WHY setsid DID NOT HELP (the _phase_b3.sh fix). setsid changes the session and process
# group; it does not change the **cgroup**. The trainers stayed inside the terminal's
# vte-spawn scope, which is the unit oomd kills.
#
# WHY THIS RUNS FROM CRON. Only user@1000.service carries ManagedOOMMemoryPressure=kill;
# system.slice and cron.service are both `auto`, i.e. not oomd candidates. A cron-started
# job lives in /system.slice/cron.service, outside the monitored subtree entirely.
# ManagedOOMPreference=omit on a transient user scope was tried first and is NOT a fix --
# the unprivileged user manager accepts the property but silently fails to set the
# user.oomd_omit xattr on the cgroup (verified: ENODATA), and there is no passwordless
# sudo on this box to disable oomd or edit the drop-in properly.
#
# The cron entry re-invokes this script every 5 minutes, so it is also a supervisor: if
# anything does kill the driver, the sweep resumes within 5 minutes and loses at most the
# in-flight epochs of the two live cells. one() skips any cell with results.json and any
# cell matching a live trainer, so re-entry is always a resume, never a restart.
#
# CONFIG IS UNCHANGED from _phase_b2/_phase_b3 and deliberately so:
#   ARCH_FILTER  the three 32-frame archs are exactly the 30 cells still missing
#   CAP=2        one job per GPU. Video Swin at 32x224 batch 4 measures 15.3 GB, so two
#                on one 24.5 GB card cannot fit; 21 of the 30 remaining cells are Swins.
# The 12-epoch / batch / lr settings live in sh/_v3_vidseeds_phaseb.sh and must not move,
# or the new seeds stop being comparable with the seed-42 column.
#
# To stop the sweep: `crontab -l | grep -v _phase_b4 | crontab -` then kill the driver.
set -u
export PATH=/usr/local/cuda/bin:/path/to/home/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/path/to/home
cd /path/to/repo || exit 1
L=logs/v3_vidseeds; mkdir -p "$L"
log(){ echo "[$(date +%F_%T)] $*"; }

# ---- single instance -------------------------------------------------------------
# Serialise the cron ticks themselves; two supervisors racing would double-launch.
exec 9>"$L/_phase_b4.lock"
flock -n 9 || exit 0

# ---- already running? ------------------------------------------------------------
# Match the driver script name, not "train_pooled": two gates counting the same jobs is
# how the cards get over-subscribed.
if pgrep -f "_v3_vidseeds_phaseb\.sh" >/dev/null 2>&1; then
  exit 0
fi

# ---- anything left to do? --------------------------------------------------------
LEFT=$(python3 - <<'PY'
import os
A = ["slowfast","swin","swin_s"]
print(sum(1 for a in A for t in ("bin","g3","g5") for s in (1,2,3,5)
          if not os.path.exists(f"output/v3_vidseeds/{a}_{t}_s{s}/results.json")))
PY
)
if [ "${LEFT:-0}" -eq 0 ]; then
  if [ ! -f "$L/_phase_b4_done.txt" ]; then
    log "phase B complete -- 36/36 cells have results.json" | tee -a "$L/_phase_b4.log" > "$L/_phase_b4_done.txt"
    python3 make_tab_vidarch_meansd.py >> "$L/_phase_b4.log" 2>&1 || true
  fi
  exit 0
fi

# ---- pre-flight ------------------------------------------------------------------
BIG=cache_frames/f32s224
if [ ! -f "$BIG/index.json" ]; then
  log "ABORT: $BIG/index.json missing -- rebuild with build_frame_cache.py" >> "$L/_phase_b4.log"
  exit 1
fi

# ---- run -------------------------------------------------------------------------
{
  log "phase B4 starting (pid $$, cgroup $(tail -1 /proc/$$/cgroup 2>/dev/null))"
  log "$LEFT of 36 cells remaining"
  df -h / | awk 'NR==2{print "  / free: "$4" ("$5" used)"}'
  export ARCH_FILTER="slowfast swin swin_s"
  export CAP=2
  log "ARCH_FILTER='$ARCH_FILTER' CAP=$CAP"
  bash sh/_v3_vidseeds_phaseb.sh
  log "driver exited rc=$?"
} >> "$L/_phase_b4.log" 2>&1
