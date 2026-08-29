#!/bin/bash
# Waits for all four frame caches, proves the cached path reproduces the uncached one
# end to end, then restarts the 120-run video seeding sweep against the caches.
#
# The input tensors were already shown to be bit-identical (max abs diff 0.000e+00 over
# 40 clips), but that only proves the reader is correct, not that a whole training run
# lands in the same place. Before committing 108 runs this re-trains one cell that has
# already finished uncached -- r2plus1d_bin_s1 -- and compares. Training is seeded, so
# an exact match is the expectation, not an approximation; a mismatch aborts the restart.
set -u
cd /path/to/repo
L=logs/v3_vidseeds; mkdir -p "$L"
F112=cache_frames/f16s112
F224=cache_frames/f16s224
F8=cache_frames/f8s224
F32=/path/to/archive
log(){ echo "[$(date +%F_%T)] $*"; }

log "waiting for the three phase-A caches"
while pgrep -f "build_frame_cache.py" > /dev/null || pgrep -f "_build_caches.sh" > /dev/null; do
  sleep 120
done
for c in "$F112" "$F224" "$F8"; do
  if [ ! -f "$c/index.json" ]; then log "ABORT: $c never completed"; exit 1; fi
  log "cache ready: $c ($(du -sh "$c" 2>/dev/null | cut -f1))"
done

# ---- end-to-end validation on an already-completed cell -----------------------
REF=output/v3_vidseeds/r2plus1d_bin_s1/results.json
CHK=output/_cachecheck/r2plus1d_bin_s1
if [ -f "$REF" ]; then
  log "validating: re-running r2plus1d_bin_s1 from cache"
  rm -rf "$CHK"
  CUDA_VISIBLE_DEVICES=0 python3 train_pooled.py --arch r2plus1d --group2 \
      --epochs 12 --batch_size 16 --lr 1e-4 --workers 5 --seed 1 --split_seed 49 \
      --cache_dir "$F112" --output "$CHK" > "$L/_cachecheck.log" 2>&1
  if [ ! -f "$CHK/results.json" ]; then
    log "ABORT: validation run produced no results"; tail -5 "$L/_cachecheck.log"; exit 1
  fi
  python3 - "$REF" "$CHK/results.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
worst = 0.0
for k in ("macro_f1", "balanced_accuracy", "accuracy"):
    d = abs(a[k] - b[k]); worst = max(worst, d)
    print(f"  {k:20s} uncached {a[k]:.6f}   cached {b[k]:.6f}   delta {d:.2e}")
print(f"  worst delta {worst:.2e}")
sys.exit(0 if worst < 1e-6 else 3)
PY
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ABORT: cached run does not reproduce the uncached one -- not restarting"
    exit 1
  fi
  log "validation passed; cached and uncached agree"
else
  log "no reference run to validate against; proceeding on the bit-identity check alone"
fi

# ---- restart the sweep --------------------------------------------------------
log "restarting the seeding sweep against the caches"
export ARCH_FILTER="r2plus1d mvit mvit_v1 x3d s3d videomae tsf"
log "phase A: $ARCH_FILTER"
exec bash sh/_v3_vidseeds.sh
