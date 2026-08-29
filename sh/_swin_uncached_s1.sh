#!/bin/bash
# Fill the one spare slot with the Video Swin-T seed-1 cells, decoding on the fly.
#
# WHY. tab:vidarch prints Video Swin-T/S as "+/-0.000 n=1" (the legacy seed-42 run alone),
# which reads as a zero-variance measurement rather than an honest gap. Phase B would fix
# it, but not before ~Aug 24 because it waits on all of phase A plus a 110 GB f32s224
# build. The uncached path needs neither: slowfast_g5_s1 was produced this way on
# 2026-08-18 and landed at 0.6684, in line with its row, so the decode path is not
# biasing results.
#
# ONE AT A TIME, deliberately. This is a 5th job on top of the driver's cap of 4, so the
# three tasks run sequentially -- concurrency stays at exactly +1 and load stays near the
# ~22 measured on 2026-08-18 with one uncached decoder. Do not parallelise these.
#
# Each cell is ~7.8 h uncached (seed-42 swin_bin 7.85 h, swin_g3 7.74 h, swin_g5 7.73 h),
# so the row completes in ~23 h. _one_now.sh already retries 3x, and the sweep's
# busy()/results.json guards make phase B skip whatever this finishes first.
set -u
cd /path/to/repo
L=logs/v3_vidseeds
log(){ echo "[$(date +%F_%T)] $*"; }

# NVML is dead on this host (driver/library mismatch), so nvidia-smi cannot report VRAM.
# The CUDA runtime still can -- pick the card with more free memory at each step, since
# the co-resident driver jobs come and go over the ~23 h this runs.
pick_gpu(){
  python3 - <<'PY' 2>/dev/null || echo 1
import torch
print(max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0]))
PY
}

for task in bin g3 g5; do
  out="output/v3_vidseeds/swin_${task}_s1"
  [ -f "$out/results.json" ] && { log "swin $task s1 already done"; continue; }
  gpu=$(pick_gpu)
  log "launching swin $task s1 on GPU$gpu (uncached)"
  UNCACHED=1 bash sh/_one_now.sh swin "$task" 1 "$gpu" >> "$L/_one_now_swin_${task}.log" 2>&1
done
log "swin seed-1 chain finished"
