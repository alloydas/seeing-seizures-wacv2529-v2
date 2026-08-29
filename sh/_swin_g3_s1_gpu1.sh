#!/bin/bash
# Second phase-B cell on GPU 1, alongside swin_bin_s1 (launched by
# sh/_prebuild_f32_and_swin.sh). Total concurrency becomes 4 -- exactly the driver's cap()
# -- with 2 TimeSformer cells on GPU 0 and 2 Swin cells here.
#
# WHY THIS MEASURES BEFORE IT LAUNCHES. Swin-T's footprint at batch 4 is not known ahead of
# time, and stacking a second job on a 24 GB card that cannot hold it is exactly the
# 2026-08-19 failure (11 cells OOMed, marked GAVE UP, after hand-launched jobs over-packed
# a card). Note that an OOM on this box no longer disguises itself as the NVML assert --
# the 2026-08-21 reboot fixed the driver skew -- but a dead cell is a dead cell. So: wait
# for cell one to reach steady state, read what GPU 1 actually has free, and only launch if
# the headroom is real. Skipping is the correct outcome if it is not.
set -u
cd /path/to/repo
L=logs/v3_vidseeds
BIG=cache_frames/f32s224
NEED_MIB=11000          # refuse to launch below this much free on GPU 1
log(){ echo "[$(date +%F_%T)] $*"; }

# ---- 1. wait for the cache the first cell is also waiting on ---------------------
log "waiting for $BIG/index.json"
for _ in $(seq 1 120); do
  [ -f "$BIG/index.json" ] && break
  sleep 30
done
[ -f "$BIG/index.json" ] || { log "ABORT: cache never appeared"; exit 1; }
log "cache ready"

# ---- 2. wait for swin_bin_s1 to allocate and settle ------------------------------
gpu1_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1; }
log "waiting for swin_bin_s1 to allocate on GPU 1"
for _ in $(seq 1 60); do
  [ "$(gpu1_used)" -gt 2000 ] && break
  sleep 30
done
if [ "$(gpu1_used)" -le 2000 ]; then
  log "ABORT: swin_bin_s1 never allocated on GPU 1 -- not stacking a second job blind"
  tail -3 "$L/swin_bin_s1.log" 2>/dev/null | cut -c1-140
  exit 1
fi
# peak allocation lands within the first training steps, not the first epoch; 4 min of
# settle is enough to read a representative number without idling the card for ~30 min
log "allocated ($(gpu1_used) MiB used) -- settling 4 min before measuring"
sleep 240

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 1)
USED=$(gpu1_used)
FREE=$(( TOTAL - USED ))
log "GPU 1: ${USED}/${TOTAL} MiB used, ${FREE} MiB free (need ${NEED_MIB})"
if [ "$FREE" -lt "$NEED_MIB" ]; then
  log "SKIP swin_g3_s1 -- not enough headroom; leaving it for phase B to run on its own"
  exit 0
fi

# ---- 3. launch ------------------------------------------------------------------
out=output/v3_vidseeds/swin_g3_s1
if [ -f "$out/results.json" ]; then log "swin_g3_s1 already complete"; exit 0; fi
if pgrep -af "python3 train_pooled" | grep -qE -- "--output $out( |\$)"; then
  log "swin_g3_s1 already running"; exit 0
fi
log "launching swin_g3_s1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 train_pooled.py --arch swin --group3 --epochs 12 \
    --batch_size 4 --lr 1e-4 --workers 5 --seed 1 --split_seed 49 \
    --cache_dir "$BIG" --output "$out" > "$L/swin_g3_s1.log" 2>&1

if [ -f "$out/results.json" ]; then
  [ -f "$out/val_preds.npz" ] && [ "${KEEP_CKPT:-1}" = "0" ] && rm -f "$out/best.pt"
  log "DONE swin_g3_s1"
else
  log "FAILED swin_g3_s1 -- tail:"
  tail -3 "$L/swin_g3_s1.log" | cut -c1-140
fi
