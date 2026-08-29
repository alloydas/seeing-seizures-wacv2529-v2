#!/bin/bash
# Runs mvit_g3_s1 on GPU 1 while the phase-A gate is still serially validating the cache
# on GPU 0. The sweep's busy()/results.json guards make it skip this cell rather than
# duplicate it when it starts. Same recipe and cache as the sweep would use.
set -u
cd /path/to/repo
OUT=output/v3_vidseeds/mvit_g3_s1
L=logs/v3_vidseeds
log(){ echo "[$(date +%F_%T)] $*"; }
[ -f "$OUT/results.json" ] && { log "already done"; exit 0; }
for try in 1 2 3; do
  log "START mvit_g3_s1 on GPU1 (try $try)"
  CUDA_VISIBLE_DEVICES=1 python3 train_pooled.py --arch mvit --group3 \
      --epochs 12 --batch_size 8 --lr 1e-4 --workers 5 --seed 1 --split_seed 49 \
      --cache_dir cache_frames/f16s224 --output "$OUT" \
      > "$L/mvit_g3_s1.log" 2>&1
  [ -f "$OUT/results.json" ] && { log "DONE mvit_g3_s1"; break; }
  log "try $try failed -- tail:"; tail -3 "$L/mvit_g3_s1.log" | cut -c1-140; sleep 120
done
