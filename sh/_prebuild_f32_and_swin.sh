#!/bin/bash
# Pre-build phase B's 32x224 frame cache on NVMe *while* phase A's last two TimeSformer
# cells are still running, then put one Swin cell on the otherwise-idle GPU 1.
#
# WHY THIS IS SAFE NOW AND WAS NOT WHEN sh/_phase_b.sh WAS WRITTEN. That script waits for
# phase A because back then all seven phase-A archs were live and all three caches were
# hot -- 207 GB of geometries against ~206 GB of page cache, which would thrash. Phase A
# is now down to two TimeSformer cells, which read f8s224 (28 GB) and nothing else:
# verified via /proc/*/fd, 10 open handles into f8s224 and zero into the other two. So
# f16s112 (14 GB) + f16s224 (55 GB) were idle and reclaimable, taking free space from
# 65 GB to 134 GB -- enough for the 110 GB build -- and the live footprint becomes
# 28 + 110 = 138 GB, comfortably inside 376 GB of RAM.
#
# HOW THIS HANDS OFF TO sh/_phase_b.sh. That script rm -rf's the three phase-A caches
# (a no-op for the two already gone), then guards its own build with
# `if [ ! -f "$BIG/index.json" ]` -- so finishing the build here makes it skip straight to
# training. It also does the F32= sed itself, which is why this script does NOT: editing
# sh/_v3_vidseeds.sh while pid 17026 is still executing it buys a risk for no gain, since
# the cell launched below passes --cache_dir explicitly.
set -u
cd /path/to/repo
L=logs/v3_vidseeds
BIG=cache_frames/f32s224
log(){ echo "[$(date +%F_%T)] $*"; }

if [ ! -f "$BIG/index.json" ]; then
  # 8 workers, not the 14 sh/_phase_b.sh uses: phase A's two cells are running 5 DataLoader
  # workers each, and the point is to not slow their finish down.
  log "building 32x224 on NVMe (110 GB, 8 workers)"
  python3 build_frame_cache.py --frames 32 --size 224 \
      --out "$BIG" --workers 8 > logs/cache_f32s224.log 2>&1
  if [ ! -f "$BIG/index.json" ]; then
    log "ABORT: 32x224 build failed"
    tail -5 logs/cache_f32s224.log
    exit 1
  fi
fi
log "32x224 ready ($(du -sh "$BIG" | cut -f1)) -- $(df --output=avail -BG / | tail -1) free"

# ---- one cell on the idle GPU 1 -------------------------------------------------
# Concurrency stays at 3 against the driver's cap of 4, and this lands on the empty card,
# so it is not the 2026-08-19 failure mode (two hand-launched squatters putting a third
# heavy job on ONE 24 GB card). Seed-major order is preserved: every phase-B cell still
# missing at seed 1 is a Swin, and this is the first of them.
#
# When phase B reaches this cell it will skip it -- one() checks `busy()`, which greps for
# `--output <path>` among running trainers, and this process matches verbatim. If it has
# already finished, the results.json check skips it instead.
out=output/v3_vidseeds/swin_bin_s1
if [ -f "$out/results.json" ]; then
  log "swin_bin_s1 already complete -- nothing to launch"
  exit 0
fi
log "launching swin_bin_s1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 train_pooled.py --arch swin --group2 --epochs 12 \
    --batch_size 4 --lr 1e-4 --workers 5 --seed 1 --split_seed 49 \
    --cache_dir "$BIG" --output "$out" > "$L/swin_bin_s1.log" 2>&1

if [ -f "$out/results.json" ]; then
  # same disposal rule as the driver: val_preds.npz is what the table generator reads,
  # so the checkpoint is dead weight once both exist
  [ -f "$out/val_preds.npz" ] && [ "${KEEP_CKPT:-1}" = "0" ] && rm -f "$out/best.pt"
  log "DONE swin_bin_s1"
else
  log "FAILED swin_bin_s1 -- tail:"
  tail -3 "$L/swin_bin_s1.log" | cut -c1-140
fi
