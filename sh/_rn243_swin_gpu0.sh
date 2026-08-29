#!/bin/bash
# Completes the RN243 held-out sweep: the two Swin backbones that failed on 2026-08-12.
#
# WHY THEY FAILED. Both died with
#   RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
#   at "../c10/cuda/CUDACachingAllocator.cpp":806
# The host still has the NVML driver/library mismatch (nvidia-smi cannot initialise), so
# any allocator path that queries NVML aborts instead of raising a normal OOM. That path
# is the *OOM retry* path -- so the crash is a memory-pressure symptom, not a driver bug
# that blocks CUDA outright: plain CUDA contexts, and the eight backbones that did finish,
# work fine. The fix is to stay under the memory ceiling so the allocator never has to
# query NVML.
#
# Swin-T/S at 32 frames x 224px are the two heaviest models in the set, which is exactly
# why only these two failed at --batch_size 16. This script walks a batch ladder
# 8 -> 4 -> 2 and keeps the first size that produces results.json.
#
# GPU 0 by request; the EEG ablation's w4 cell holds GPU 1.
set -u
cd /path/to/repo
L=logs/rn243; mkdir -p "$L" output/rn243
SESS="/path/to/archive"
log(){ echo "[$(date +%F_%T)] $*"; }

run_one(){
  local arch=$1
  local ckpt=$2
  local out="output/rn243/video_${arch}_10-16-2023"
  if [ ! -f "$ckpt" ]; then log "SKIP $arch (no checkpoint)"; return 0; fi
  if [ -f "$out/results.json" ]; then log "SKIP $arch (done)"; return 0; fi
  local bs
  for bs in 8 4 2; do
    log "START $arch at batch_size=$bs"
    python3 sweep_session.py --session "$SESS" --ckpt "$ckpt" --arch "$arch" \
        --subject RN243 --out "$out" --batch_size "$bs" --device cuda:0 \
        > "$L/video_${arch}.log" 2>&1
    if [ -f "$out/results.json" ]; then log "DONE $arch (batch_size=$bs)"; return 0; fi
    log "$arch failed at batch_size=$bs -- tail:"; tail -3 "$L/video_${arch}.log"
    sleep 30
  done
  log "GAVE UP $arch"
}

export CUDA_VISIBLE_DEVICES=0
run_one swin     output/vid_swin_bin/best.pt
run_one swin_s   output/vid_swin_s_bin/best.pt

log "===== RN243 Swin sweep finished; full backbone table ====="
python3 - <<'PY'
import json, glob, os
rows=[]
for f in sorted(glob.glob('output/rn243/video_*_10-16-2023/results.json')) + \
         ['output/rn243/video_10-16-2023/results.json']:
    if not os.path.exists(f): continue
    name=os.path.basename(os.path.dirname(f)).replace('video_','').replace('_10-16-2023','') or 'r2plus1d'
    r=json.load(open(f))
    best=None
    for row in r.get('sweep', r if isinstance(r,list) else []):
        if isinstance(row,dict) and row.get('thr')==0.5: best=row
    rows.append((name, best))
for n,b in rows:
    if b: print(f"  {n:12s} thr0.5  recall={b.get('recall',0):.3f}  FP/h={b.get('fp_per_h',0):.1f}")
    else: print(f"  {n:12s} (no thr=0.5 row)")
PY
