#!/bin/bash
# Regenerate the three best-scoring models, WITH their checkpoints this time.
#
# WHY THIS EXISTS. one() in sh/_v3_vidseeds_phaseb.sh deleted every cell's best.pt as soon
# as results.json + val_preds.npz existed (a disk-space policy, on the assumption a run is
# reproducible from its seed). That assumption is being cashed in here: the metrics for the
# best cell of each job survive, the weights do not, so each is re-run at its original seed.
#
#   job        original cell     original macro-F1   cache
#   detection  slowfast_bin_s1   0.9760              f32s224 (present)
#   3-class    x3d_g3_s2         0.8007              f16s224 (deleted -- rebuilt below)
#   5-class    mvit_g5_s5        0.6825              f16s224
#
# Results go to output/best_models/<cell>/ and NOT to output/v3_vidseeds/<cell>/ -- the
# originals are what the paper tables cite and must not be overwritten. Reproduction is
# checked by comparing the new macro_f1 against the number above.
#
# RUNS FROM CRON, never a login shell: systemd-oomd kills anything under
# user@1000.service when its PSI crosses 50%. See sh/_phase_b4.sh for the full story.
set -u
export PATH=/usr/local/cuda/bin:/path/to/home/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/path/to/home
cd /path/to/repo || exit 1
L=logs/best_models; mkdir -p "$L" output/best_models
# Everything this script says goes to a file. Cron would otherwise mail stdout, and
# MAILTO="" throws that away -- which is how a driver ends up with no diagnostics at all.
exec >> "$L/_retrain.log" 2>&1
log(){ echo "[$(date +%F_%T)] $*"; }

# flock alone gives single-instance across cron ticks; a second pgrep gate would just be
# another thing to get wrong (it is how the phase-B driver once double-counted its own jobs).
exec 9>"$L/_retrain.lock"; flock -n 9 || exit 0

# cell | arch | group flag | seed | cache
# Early exit once every cell has landed. Without this the script re-runs on every
# 5-minute tick forever (123 times before this was noticed), re-appending its summary
# and -- worse -- keeping a `_retrain_best3.sh` process alive often enough that
# sh/_after_retrain_slowfast.sh's pgrep gate matched and its queue never started.
if [ -f output/best_models/slowfast_bin_s1/results.json ] \
   && [ -f output/best_models/x3d_g3_s2/results.json ] \
   && [ -f output/best_models/mvit_g5_s5/results.json ]; then
  [ -f "$L/_retrain_done.txt" ] || log "all three cells complete" > "$L/_retrain_done.txt"
  exit 0
fi

JOBS=(
  "slowfast_bin_s1|slowfast|--group2|1|cache_frames/f32s224"
  "x3d_g3_s2|x3d|--group3|2|cache_frames/f16s224"
  "mvit_g5_s5|mvit||5|cache_frames/f16s224"
)

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
  local cell=$1 arch=$2 flag=$3 seed=$4 cache=$5
  local out="output/best_models/$cell"
  [ -f "$out/results.json" ] && return 0
  busy "$out" && return 0
  [ -f "$cache/index.json" ] || { log "SKIP $cell -- $cache not built yet"; return 0; }
  local gpu; gpu=$(pick_gpu)
  log "launch $cell on gpu$gpu (arch=$arch seed=$seed cache=$cache)"
  CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled.py --arch "$arch" $flag --epochs 12 \
      --batch_size 8 --lr 1e-4 --workers 5 --seed "$seed" --split_seed 49 \
      --cache_dir "$cache" --output "$out" > "$L/$cell.log" 2>&1
  if [ -f "$out/results.json" ]; then
    # the whole point: keep the weights
    log "DONE $cell  ckpt=$([ -f "$out/best.pt" ] && du -h "$out/best.pt" | cut -f1 || echo MISSING)"
  else
    log "FAILED $cell -- tail:"; tail -3 "$L/$cell.log" | cut -c1-140
  fi
}

log "=== retrain-best3 starting (cgroup $(tail -1 /proc/$$/cgroup 2>/dev/null)) ==="

# ---- 1. anything runnable with a cache that already exists -----------------------
for spec in "${JOBS[@]}"; do
  IFS='|' read -r cell arch flag seed cache <<< "$spec"
  [ -f "$cache/index.json" ] || continue
  while [ "$(nvid)" -ge 2 ]; do sleep 60; done
  run_one "$cell" "$arch" "$flag" "$seed" "$cache" &
  sleep 20
done

# ---- 2. rebuild the f16s224 cache the other two need ------------------------------
if [ ! -f cache_frames/f16s224/index.json ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${FREE:-0}" -lt 60 ]; then
    log "ABORT cache build: only ${FREE}G free, need ~55G + headroom"
  else
    log "building cache_frames/f16s224 (~55 GiB, ${FREE}G free)"
    python3 build_frame_cache.py --frames 16 --size 224 --out cache_frames/f16s224 \
        --workers 12 > "$L/_build_f16s224.log" 2>&1
    log "cache build exited rc=$? ($(du -sh cache_frames/f16s224 2>/dev/null | cut -f1))"
  fi
fi

# ---- 3. the two that were waiting on it -------------------------------------------
for spec in "${JOBS[@]}"; do
  IFS='|' read -r cell arch flag seed cache <<< "$spec"
  [ -f "$cache/index.json" ] || continue
  [ -f "output/best_models/$cell/results.json" ] && continue
  while [ "$(nvid)" -ge 2 ]; do sleep 60; done
  run_one "$cell" "$arch" "$flag" "$seed" "$cache" &
  sleep 20
done

wait
log "=== retrain-best3 complete ==="
python3 - <<'PY' 2>&1 | tee -a "$L/_reproduction.txt"
import json, os
ORIG = {"slowfast_bin_s1": 0.9760, "x3d_g3_s2": 0.8007, "mvit_g5_s5": 0.6825}
print("cell                 original   reproduced   delta    ckpt")
for c, o in ORIG.items():
    p = f"output/best_models/{c}/results.json"
    if not os.path.exists(p):
        print(f"{c:20s} {o:.4f}     (not run)"); continue
    n = json.load(open(p)).get("macro_f1", float("nan"))
    ck = f"output/best_models/{c}/best.pt"
    sz = f"{os.path.getsize(ck)/2**20:.0f} MB" if os.path.exists(ck) else "MISSING"
    print(f"{c:20s} {o:.4f}     {n:.4f}     {n-o:+.4f}  {sz}")
PY
