#!/bin/bash
# The six unstarted slowfast cells of phase B (seeds 3 and 5), queued to run ONLY after
# sh/_retrain_best3.sh has finished regenerating the three best models.
#
# WHY SEPARATE FROM _phase_b4.sh. That driver's ARCH_FILTER covers slowfast+swin+swin_s,
# i.e. all 23 unstarted cells and ~9 h apiece for the Swins. These six are the cheap ones:
# slowfast at batch 8 measured 2.9 h/cell, and they reuse cache_frames/f32s224, which is
# already on disk -- so they cost no cache build and only 5 dataloader workers each.
#
# WHY GATED. The retrains are the priority; CAP=2 means anything running here would push
# x3d_g3_s2 / mvit_g5_s5 behind it. This script therefore no-ops on every cron tick until
# all three retrain cells have a results.json AND the retrain driver has exited.
#
# Results go to output/v3_vidseeds/<cell>/, the same place the rest of the seed table
# lives, so make_tab_vidarch_meansd.py picks them up and slowfast reaches n=4.
# Checkpoints are KEPT (KEEP_CKPT now defaults to 1) -- 6 x ~135 MB is immaterial.
#
# RUNS FROM CRON: systemd-oomd kills anything under user@1000.service above 50% PSI.
set -u
export PATH=/usr/local/cuda/bin:/path/to/home/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/path/to/home
cd /path/to/repo || exit 1
L=logs/v3_vidseeds; mkdir -p "$L" logs/best_models
exec 9>"logs/best_models/_after_slowfast.lock"; flock -n 9 || exit 0
exec >> "$L/_after_slowfast.log" 2>&1
log(){ echo "[$(date +%F_%T)] $*"; }

CACHE=cache_frames/f32s224

# ---- gate 1: all three retrains must have landed -----------------------------------
for c in slowfast_bin_s1 x3d_g3_s2 mvit_g5_s5; do
  [ -f "output/best_models/$c/results.json" ] || exit 0
done
# ---- gate 2: the retrain driver and its cache build must be gone -------------------
pgrep -f "_retrain_best3\.sh"   >/dev/null 2>&1 && exit 0
pgrep -f "build_frame_cache\.py" >/dev/null 2>&1 && exit 0
# ---- gate 3: nothing left to do ----------------------------------------------------
[ -f "$CACHE/index.json" ] || { log "ABORT: $CACHE missing"; exit 1; }

JOBS=(
  "slowfast_bin_s3|--group2|3"
  "slowfast_g3_s3|--group3|3"
  "slowfast_g5_s3||3"
  "slowfast_bin_s5|--group2|5"
  "slowfast_g3_s5|--group3|5"
  "slowfast_g5_s5||5"
)
LEFT=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r cell flag seed <<< "$spec"
  [ -f "output/v3_vidseeds/$cell/results.json" ] || LEFT=$((LEFT+1))
done
[ "$LEFT" -eq 0 ] && { [ -f "$L/_after_slowfast_done.txt" ] || log "all six slowfast cells complete" > "$L/_after_slowfast_done.txt"; exit 0; }

# Count JOBS not processes: each trainer's DataLoader children inherit the parent command
# line verbatim, so a naive process count pins the gate shut at one job forever.
nvid(){ pgrep -af "python3 train_pooled\.py" 2>/dev/null | grep -o -- "--output [^ ]*" | sort -u | wc -l; }
busy(){ pgrep -af "python3 train_pooled\.py" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }
pick_gpu(){
  local g0=0 g1=0 p g
  for p in $(pgrep -f "python3 train_pooled\.py"); do
    g=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ "$g" = "0" ] && g0=$((g0+1)); [ "$g" = "1" ] && g1=$((g1+1))
  done
  [ "$g0" -le "$g1" ] && echo 0 || echo 1
}

run_one(){
  local cell=$1 flag=$2 seed=$3
  local out="output/v3_vidseeds/$cell"
  [ -f "$out/results.json" ] && return 0
  busy "$out" && return 0
  local gpu; gpu=$(pick_gpu)
  log "launch $cell on gpu$gpu (seed=$seed)"
  local try
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled.py --arch slowfast $flag --epochs 12 \
        --batch_size 8 --lr 1e-4 --workers 5 --seed "$seed" --split_seed 49 \
        --cache_dir "$CACHE" --output "$out" > "$L/$cell.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE $cell"; return 0; }
    log "retry $try failed $cell -- tail:"; tail -3 "$L/$cell.log" | cut -c1-140
    sleep 120
  done
  log "GAVE UP $cell"
}

log "=== after-retrain slowfast starting ($LEFT of 6 left, cgroup $(tail -1 /proc/$$/cgroup 2>/dev/null)) ==="
for spec in "${JOBS[@]}"; do
  IFS='|' read -r cell flag seed <<< "$spec"
  [ -f "output/v3_vidseeds/$cell/results.json" ] && continue
  while [ "$(nvid)" -ge 2 ]; do sleep 60; done
  run_one "$cell" "$flag" "$seed" &
  sleep 20
done
wait
log "=== after-retrain slowfast complete ==="
python3 make_tab_vidarch_meansd.py 2>&1 | tail -20 || true
