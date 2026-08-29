#!/bin/bash
# Phase B of the video seeding sweep: the three 32-frame architectures.
#
# WHY A SECOND PHASE. The four frame-cache geometries total 207 GB (14 + 55 + 28 + 110)
# against ~206 GB of page cache, and the seed-major sweep interleaves geometries
# continuously, so holding all four live would thrash. Worse, 110 GB does not fit on the
# NVMe alongside the other three, and the only other volume is a 164 MB/s spinning disk
# (measured; NVMe is 2.5 GB/s) -- putting the actively-read cache there would make the
# disk the bottleneck and undo the point of caching at all.
#
# So: phase A runs the seven archs served by the 97 GB of NVMe caches, then this script
# reclaims that space and builds 32x224 on NVMe for slowfast / swin / swin_s.
#
# Seed-major order is preserved WITHIN each phase, so every cell in a phase carries the
# same seed count; the phases just complete at different times.
set -u
cd /path/to/repo
L=logs/v3_vidseeds; mkdir -p "$L"
log(){ echo "[$(date +%F_%T)] $*"; }

log "waiting for phase A to finish"
while pgrep -f "_v3_vidseeds\.sh" > /dev/null || pgrep -f "_restart_seeding_cached\.sh" > /dev/null; do
  sleep 300
done
log "phase A finished"

python3 - <<'PY'
import os
A = ["r2plus1d","mvit","mvit_v1","x3d","s3d","videomae","tsf"]
n = sum(os.path.exists(f"output/v3_vidseeds/{a}_{t}_s{s}/results.json")
        for a in A for t in ("bin","g3","g5") for s in (1,2,3,5))
print(f"  phase A: {n}/{len(A)*12} runs complete")
PY

# ---- reclaim the phase-A caches ------------------------------------------------
log "removing phase-A caches to make room (they are rebuildable from data/)"
for c in cache_frames/f16s112 cache_frames/f16s224 cache_frames/f8s224; do
  du -sh "$c" 2>/dev/null | sed 's/^/    freeing /'
  rm -rf "$c"
done
df -h / | awk 'NR==2{print "    / free now: "$4}'

# ---- build the 32x224 cache on NVMe --------------------------------------------
BIG=cache_frames/f32s224
if [ ! -f "$BIG/index.json" ]; then
  log "building 32x224 on NVMe (110 GB)"
  python3 build_frame_cache.py --frames 32 --size 224 \
      --out "$BIG" --workers 14 > logs/cache_f32s224.log 2>&1
  [ -f "$BIG/index.json" ] || { log "ABORT: 32x224 build failed"; tail -5 logs/cache_f32s224.log; exit 1; }
fi
log "32x224 ready ($(du -sh "$BIG" | cut -f1))"

# point the sweep's F32 at the NVMe copy rather than the backup disk
sed -i 's|^F32=/path/to/archive|F32=cache_frames/f32s224|' sh/_v3_vidseeds.sh
grep -m1 '^F32=' sh/_v3_vidseeds.sh | sed 's/^/  /'

# ---- run the remaining three architectures -------------------------------------
export ARCH_FILTER="slowfast swin swin_s"
log "phase B: $ARCH_FILTER"
exec bash sh/_v3_vidseeds.sh
