#!/bin/bash
# Seed-1 cells for BOTH Video Swin rows, uncached, strictly one at a time.
#
# WHY A QUEUE AND NOT TWO CHAINS. Only one 32-frame Swin fits on this box alongside the
# driver's four cached jobs. Measured 2026-08-19: swin (Swin-T, 28.2M params) took
# 14.4 GiB at batch 4, taking GPU1 from 14.6 GiB free to 0.2. swin_s (Swin-S, 49.8M --
# same widths, stage-3 depth 18 vs 6) needs more still, so the two cannot be co-resident,
# and neither fits in the ~7.6 GiB that two driver jobs leave on a card.
#
# ORDER IS A PREFERENCE, NOT FIFO. The list is interleaved (swin_s bin, swin g3, swin_s
# g3, swin g5, swin_s g5) so both rows advance together rather than leaving swin_s at
# "+/-0.000 n=1" -- the artefact this exists to remove -- for an extra day. But the loop
# runs the first cell that FITS RIGHT NOW, not the first in the list. Strict FIFO
# deadlocks: swin_s (18.5 GiB) would hold the head while the 14.6 GiB slot freed by
# swin_bin_s1 -- big enough for a swin cell, too small for a swin_s one -- sat unused
# until the driver's seed barrier ~19 h later.
#
# swin_bin_s1 is already running from the earlier chain, hence its absence from the list.
#
# ~7.8 h per cell uncached, 5 cells -> ~2 days, still well ahead of phase B (~Aug 24).
set -u
cd /path/to/repo
L=logs/v3_vidseeds
log(){ echo "[$(date +%F_%T)] $*"; }

# GiB of free VRAM to require. swin measured at 14.4 allocated and the slot it vacates on
# a two-driver-job card is 14.6, so this margin is deliberately thin -- any higher and a
# swin cell could never reclaim its own slot. _one_now.sh retries 3x, so losing a narrow
# race against the driver costs 2 min, not the run.
need(){ [ "$1" = "swin_s" ] && echo 18.5 || echo 14.5; }

# NVML is dead on this host (driver/library mismatch) so nvidia-smi cannot report VRAM;
# the CUDA runtime still can. Prints the emptiest GPU index if it clears $1 GiB, else "".
free_gpu(){
  python3 - "$1" <<'PY' 2>/dev/null
import sys, torch
want = float(sys.argv[1]) * 2**30
best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
if torch.cuda.mem_get_info(best)[0] >= want:
    print(best)
PY
}

done_cell(){ [ -f "output/v3_vidseeds/$1_$2_s1/results.json" ]; }
busy_cell(){ pgrep -af "python3 train_pooled" | grep -q -- "--output output/v3_vidseeds/$1_$2_s1\( \|$\)"; }

PENDING="swin_s:bin swin:g3 swin_s:g3 swin:g5 swin_s:g5"
log "=== swin + swin_s seed-1 queue: 5 cells, one at a time, first-that-fits ==="
waited=0
while [ -n "${PENDING// /}" ]; do
  launched=0; remaining=""
  for cell in $PENDING; do
    arch=${cell%%:*}; task=${cell##*:}
    if done_cell "$arch" "$task"; then log "$arch $task s1 already done"; continue; fi
    if busy_cell "$arch" "$task"; then log "$arch $task s1 already running elsewhere"; continue; fi
    g=""
    [ "$launched" -eq 0 ] && g=$(free_gpu "$(need "$arch")")
    if [ -n "$g" ]; then
      log "launching $arch $task s1 on GPU$g (waited ${waited} min)"
      UNCACHED=1 bash sh/_one_now.sh "$arch" "$task" 1 "$g" >> "$L/_one_now_${arch}_${task}.log" 2>&1
      launched=1; waited=0
      done_cell "$arch" "$task" || { log "  $arch $task s1 produced no results.json -- requeued"; remaining="$remaining $cell"; }
    else
      remaining="$remaining $cell"
    fi
  done
  PENDING="$remaining"
  if [ "$launched" -eq 0 ] && [ -n "${PENDING// /}" ]; then
    sleep 300; waited=$(( waited + 5 ))
    [ $(( waited % 60 )) -eq 0 ] && log "  nothing fits yet; pending:$PENDING (${waited} min)"
  fi
done
log "===== swin/swin_s seed-1 queue finished ====="
python3 make_tab_vidarch_meansd.py 2>&1 | tail -14 || true
