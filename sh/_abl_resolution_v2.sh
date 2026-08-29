#!/bin/bash
# Spatial-resolution ablation: R(2+1)D detection at 112 / 224 / 448 px, 16 frames,
# everything else fixed. Re-run under CURRENT code.
#
# Two reasons this needs redoing rather than citing the existing numbers:
#
#  1. The 448 run NEVER FINISHED. output/abl_res/r2p1d_448/ has best.pt and
#     val_preds.npz but no results.json and no history.json -- logs/abl_res/res448.log
#     stops at epoch 8 of 10. The paper's 0.952 for 448 is epoch 8's val macro-F1
#     (0.9519) from a partial run, not a completed result.
#     (output/abl_scratch/ likewise has no results.json, and the paper quotes 0.952 for
#     "from scratch" too -- worth checking whether the same partial number was reused.)
#
#  2. Both surviving numbers predate the 2026-08-03 train_pooled.py edit (224 was
#     written 2026-08-02), so they cannot sit in a row with anything measured since.
#
# All three cells here are run to completion at 12 epochs (the old sweep used 10) with
# seed 42 / split_seed 49, so the row is internally consistent. Batch scales with
# resolution to keep activation memory bounded: 112 -> 16, 224 -> 8, 448 -> 4.
#
# Sequential on one GPU: 448 at bs 4 is the most memory-hungry job run in this project,
# and NVML is still broken by the driver mismatch, so memory pressure aborts hard
# instead of raising a clean OOM. One at a time avoids that entirely.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/abl_res2; mkdir -p "$L" output/abl_res2
log(){ echo "[$(date +%F_%T)] $*"; }

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

run_size(){          # $1=size  $2=batch
  local size=$1 bs=$2 try
  local out="output/abl_res2/r2p1d_${size}"   # separate stmt: ${size} is not yet
                                              # set while the first `local` line
                                              # is being expanded (set -u aborts)
  [ -f "$out/results.json" ] && { log "SKIP size=$size (done)"; return 0; }
  busy "$out" && { log "SKIP size=$size (running)"; return 0; }
  for try in 1 2; do
    log "START size=${size} bs=${bs} (try $try)"
    python3 train_pooled.py --arch r2plus1d --group2 --frames 16 --size "$size" \
        --epochs 12 --batch_size "$bs" --workers 8 --seed 42 --split_seed 49 \
        --output "$out" > "$L/res${size}.log" 2>&1
    rc=$?
    [ -f "$out/results.json" ] && { log "DONE size=$size (rc=$rc)"; return 0; }
    log "FAILED size=$size try $try (rc=$rc); tail:"; tail -4 "$L/res${size}.log"; sleep 120
  done
  log "GAVE UP size=$size"; return 1
}

run_size 112 16
run_size 224 8
run_size 448 4

log "===== resolution ablation (current code, 12 epochs, seed 42) ====="
python3 - <<'PY'
import json, os
print(f"{'size':>6s} {'bal_acc':>9s} {'macro_f1':>9s}")
for s in (112, 224, 448):
    p = f'output/abl_res2/r2p1d_{s}/results.json'
    if os.path.exists(p):
        r = json.load(open(p))
        print(f"{s:6d} {r['balanced_accuracy']:9.4f} {r['macro_f1']:9.4f}")
    else:
        print(f"{s:6d} {'MISSING':>9s}")
print("\nold (pre-2026-08-03, 10 epochs): 112=0.970  224=0.9669  448=0.9519 (partial, ep8/10)")
PY
