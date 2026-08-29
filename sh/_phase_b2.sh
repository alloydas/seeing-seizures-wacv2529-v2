#!/bin/bash
# Phase B, second cut. Replaces sh/_phase_b.sh (pid 17768), which was killed because two of
# its assumptions no longer hold.
#
# 1. THE CACHE IS ALREADY BUILT. sh/_prebuild_f32_and_swin.sh built f32s224 on NVMe at
#    17:44 on 2026-08-22, while phase A's last two cells were still running -- so the
#    reclaim-and-build stage is done and only f8s224 is left to free.
#
# 2. cap() 4 WOULD OOM THIS PHASE. That cap means two jobs per card and was tuned for
#    phase A, whose archs take ~10.7 GB each (two TimeSformer cells sit at 21.5 GB on one
#    24.5 GB card). Measured 2026-08-22 17:48: Video Swin-T at 32x224 batch 4 takes
#    15.3 GB, so two on one card need 30.6 GB and cannot fit. 24 of phase B's 33 cells are
#    Swins. Left alone, pick_gpu would have stacked two per card and reproduced the
#    2026-08-19 cascade (11 cells OOMed and marked GAVE UP) straight from the config.
#    Hence CAP=2 -- one job per GPU -- via sh/_v3_vidseeds_phaseb.sh, a copy of the driver
#    whose only changes are `cap(){ echo "${CAP:-4}"; }` and F32 pointed at the NVMe cache.
#    The original driver is left byte-identical because pid 17026 is still executing it.
#
#    COST: phase B runs at half the job concurrency. If SlowFast (batch 8) measures light
#    enough to pair on one card, its 9 cells could be re-run at CAP=4 separately; that is
#    not assumed here because no SlowFast memory figure has been measured on this box.
set -u
cd /path/to/repo
L=logs/v3_vidseeds
BIG=cache_frames/f32s224
log(){ echo "[$(date +%F_%T)] $*"; }

# Watch the phase A driver by pid. Name matching is useless here: the driver forks a
# subshell per running cell that inherits its cmdline verbatim, and this script's own
# successor execs into a driver copy with a similar name.
APID=17026
log "waiting for phase A driver (pid $APID) to exit"
while [ -r "/proc/$APID/cmdline" ] \
      && tr '\0' ' ' < "/proc/$APID/cmdline" 2>/dev/null | grep -q "_v3_vidseeds\.sh"; do
  sleep 60
done
log "phase A finished"

python3 - <<'PY'
import os
A = ["r2plus1d","mvit","mvit_v1","x3d","s3d","videomae","tsf"]
miss = [(a,t,s) for a in A for t in ("bin","g3","g5") for s in (1,2,3,5)
        if not os.path.exists(f"output/v3_vidseeds/{a}_{t}_s{s}/results.json")]
print(f"  phase A: {84-len(miss)}/84 complete" + (f" -- MISSING {miss}" if miss else ""))
PY

# ---- free the last phase-A cache -------------------------------------------------
# f8s224 is the only one left (f16s112 and f16s224 were reclaimed at 17:20 to make room
# for the f32 build). Rebuildable from data/ in ~14 min if more TimeSformer seeds are ever
# wanted. This takes free space from ~24 GB back to ~52 GB.
if [ -d cache_frames/f8s224 ]; then
  log "freeing cache_frames/f8s224 ($(du -sh cache_frames/f8s224 | cut -f1))"
  rm -rf cache_frames/f8s224
fi
df -h / | awk 'NR==2{print "  / free now: "$4}'

[ -f "$BIG/index.json" ] || { log "ABORT: $BIG/index.json missing"; exit 1; }
log "32x224 ready ($(du -sh "$BIG" | cut -f1))"

# ---- run the three 32-frame architectures ----------------------------------------
# swin_bin_s1 is already running on GPU 1 (launched 17:44). The driver's one() will skip
# it: busy() greps running trainers for `--output <path>` and that process matches.
export ARCH_FILTER="slowfast swin swin_s"
export CAP=2
log "phase B: ARCH_FILTER='$ARCH_FILTER' CAP=$CAP"
exec bash sh/_v3_vidseeds_phaseb.sh
