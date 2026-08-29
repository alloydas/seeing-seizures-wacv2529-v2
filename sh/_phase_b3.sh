#!/bin/bash
# Phase B, third cut. Relaunches the driver after the 2026-08-23 06:40 kill.
#
# WHAT HAPPENED. The phase B driver (started by sh/_phase_b2.sh) stopped logging at
# 05:50 and its two children -- swin_s_bin_s1 and swin_s_g3_s1 -- each died right after
# saving an epoch-1 best.pt, at 06:05 and 06:39. No traceback, no GAVE UP, no partial
# epoch in either log; no OOM or kernel message in journalctl; no reboot (box up since
# 2026-08-21 14:24). That signature is a SIGKILL to the process group, not a crash in
# the trainer -- so the fix here is detachment, not a config change.
#
# HENCE setsid. This script is launched with `setsid nohup`, which makes it a session
# leader in its own process group with no controlling terminal, so a kill aimed at the
# shell that started it (or that shell's group) cannot reach it. sh/_phase_b2.sh was an
# ordinary child of its launching shell and had no such protection.
#
# CONFIG IS UNCHANGED from _phase_b2.sh and deliberately so:
#   ARCH_FILTER  the three 32-frame archs are exactly the 30 cells still missing
#   CAP=2        one job per GPU. Video Swin at 32x224 batch 4 measures 15.3 GB, so two
#                on one 24.5 GB card cannot fit; 21 of the 30 remaining cells are Swins.
#                See sh/_phase_b2.sh and the 2026-08-19 GAVE UP cascade.
# The 12-epoch / batch / lr settings live in sh/_v3_vidseeds_phaseb.sh and must not move,
# or the new seeds stop being comparable with the seed-42 column.
set -u
cd /path/to/repo
L=logs/v3_vidseeds; mkdir -p "$L"
BIG=cache_frames/f32s224
log(){ echo "[$(date +%F_%T)] $*"; }

echo $$ > "$L/_phase_b3.pid"
log "phase B3 starting (pid $$, sid $(ps -o sid= -p $$ | tr -d ' '))"

# ---- pre-flight ------------------------------------------------------------------
# Never start a second driver on top of a live one: two gates counting the same jobs is
# how cards get over-subscribed. Match the driver script name, not "train_pooled".
if pgrep -f "_v3_vidseeds_phaseb\.sh" | grep -qv "^$$\$"; then
  log "ABORT: a phase B driver is already running -- $(pgrep -af '_v3_vidseeds_phaseb\.sh' | head -3)"
  exit 1
fi
if pgrep -f "python3 train_pooled" | grep -qv "^$$\$"; then
  log "NOTE: trainers already running; the driver's busy()/nvid() gate will account for them"
  pgrep -af "python3 train_pooled" | grep -o -- "--output [^ ]*" | sort -u | sed 's/^/  /'
fi
[ -f "$BIG/index.json" ] || { log "ABORT: $BIG/index.json missing -- rebuild with build_frame_cache.py"; exit 1; }
log "32x224 cache present"
df -h / | awk 'NR==2{print "  / free: "$4" ("$5" used)"}'

python3 - <<'PY'
import os
A = ["slowfast","swin","swin_s"]
miss = [f"{a}_{t}_s{s}" for a in A for t in ("bin","g3","g5") for s in (1,2,3,5)
        if not os.path.exists(f"output/v3_vidseeds/{a}_{t}_s{s}/results.json")]
print(f"  phase B: {36-len(miss)}/36 done, {len(miss)} to run")
print("  " + ", ".join(miss))
PY

# ---- run -------------------------------------------------------------------------
# one() skips any cell that already has results.json and any cell whose --output matches
# a live trainer, so this is a resume, not a restart: the 90 completed cells are untouched.
export ARCH_FILTER="slowfast swin swin_s"
export CAP=2
log "phase B: ARCH_FILTER='$ARCH_FILTER' CAP=$CAP"
exec bash sh/_v3_vidseeds_phaseb.sh
