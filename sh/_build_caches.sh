#!/bin/bash
# Builds the two larger frame caches after the 16x112 one finishes.
#   16x224 -> NVMe   (55 GB; serves mvit, mvit_v1, s3d, x3d, videomae, tsf)
#   32x224 -> backup disk (110 GB; serves swin, swin_s, slowfast -- will not fit on /)
set -u
cd /path/to/repo
log(){ echo "[$(date +%F_%T)] $*"; }
log "waiting for the 16x112 build"
while pgrep -f "build_frame_cache.py --frames 16 --size 112" > /dev/null; do sleep 60; done
log "16x112 done"

log "building 16x224 (55 GB, NVMe)"
python3 build_frame_cache.py --frames 16 --size 224 \
    --out cache_frames/f16s224 --workers 14 > logs/cache_f16s224.log 2>&1
log "16x224 done: $(du -sh cache_frames/f16s224 2>/dev/null | cut -f1)"

log "building 8x224 (28 GB, NVMe) -- TimeSformer defaults to 8 frames, not 16"
python3 build_frame_cache.py --frames 8 --size 224 \
    --out cache_frames/f8s224 --workers 14 > logs/cache_f8s224.log 2>&1
log "8x224 done: $(du -sh cache_frames/f8s224 2>/dev/null | cut -f1)"

BIG=/path/to/archive
log "building 32x224 (110 GB, backup disk)"
python3 build_frame_cache.py --frames 32 --size 224 \
    --out "$BIG" --workers 14 > logs/cache_f32s224.log 2>&1
log "32x224 done: $(du -sh "$BIG" 2>/dev/null | cut -f1)"
log "===== all caches built ====="
df -h / /path/to/archive | awk 'NR==1||/dev/'
