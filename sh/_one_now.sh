#!/bin/bash
# Launch a single seeding cell by hand, on a chosen GPU, with the same recipe and cache
# the sweep would use. Handy while the phase-A gate is serially validating and a card
# would otherwise idle. The sweep's busy()/results.json guards make it skip anything
# started this way rather than duplicating it.
#
# usage: bash sh/_one_now.sh <arch> <bin|g3|g5> <seed> <gpu>
#
# UNCACHED=1 runs with no --cache_dir, decoding H.264 on the fly. Needed for the
# 32-frame archs (slowfast/swin/swin_s) before phase B builds f32s224 -- that is how
# slowfast_bin_s1 and slowfast_g3_s1 were actually produced, so a run started this way
# is comparable with them. It is an explicit opt-in, NOT a fallback: silently dropping
# a missing cache would turn a typo into a quiet 2.7x-slower run for the cached archs.
# Costs ~5 CPU cores of decode, so check `uptime` against nproc before using it.
set -u
cd /path/to/repo
ARCH=$1; TASK=$2; SEED=$3; GPU=$4
L=logs/v3_vidseeds; mkdir -p "$L"
log(){ echo "[$(date +%F_%T)] $*"; }

case "$ARCH" in
  r2plus1d) TR="train_pooled.py --arch r2plus1d"; EX="--batch_size 16 --lr 1e-4"; C=cache_frames/f16s112 ;;
  mvit|mvit_v1|x3d|s3d) TR="train_pooled.py --arch $ARCH"; EX="--batch_size 8 --lr 1e-4"; C=cache_frames/f16s224 ;;
  videomae) TR="train_pooled_videomae.py"; EX="--batch_size 8 --lr 1e-4"; C=cache_frames/f16s224 ;;
  tsf)      TR="train_pooled_timesformer.py"; EX="--batch_size 8 --lr 5e-5"; C=cache_frames/f8s224 ;;
  slowfast) TR="train_pooled.py --arch slowfast"; EX="--batch_size 8 --lr 1e-4"; C=cache_frames/f32s224 ;;
  swin|swin_s) TR="train_pooled.py --arch $ARCH"; EX="--batch_size 4 --lr 1e-4"; C=cache_frames/f32s224 ;;
  *) log "unknown arch $ARCH"; exit 1 ;;
esac
case "$TASK" in bin) FLAG="--group2" ;; g3) FLAG="--group3" ;; g5) FLAG="" ;; *) log "bad task"; exit 1 ;; esac

OUT="output/v3_vidseeds/${ARCH}_${TASK}_s${SEED}"
[ -f "$OUT/results.json" ] && { log "$ARCH $TASK s$SEED already done"; exit 0; }
if [ "${UNCACHED:-0}" = "1" ]; then
  CARG=""; log "UNCACHED=1 -- decoding on the fly, ignoring $C"
else
  [ -d "$C" ] || { log "cache $C missing (UNCACHED=1 to decode on the fly)"; exit 1; }
  CARG="--cache_dir $C"
fi

for try in 1 2 3; do
  log "START ${ARCH}_${TASK}_s${SEED} on GPU$GPU (try $try)"
  CUDA_VISIBLE_DEVICES=$GPU python3 $TR $FLAG --epochs 12 $EX \
      --workers 5 --seed "$SEED" --split_seed 49 $CARG --output "$OUT" \
      > "$L/${ARCH}_${TASK}_s${SEED}.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "DONE ${ARCH}_${TASK}_s${SEED}"; exit 0; }
  log "try $try failed -- tail:"; tail -3 "$L/${ARCH}_${TASK}_s${SEED}.log" | cut -c1-140
  sleep 120
done
log "GAVE UP ${ARCH}_${TASK}_s${SEED}"
