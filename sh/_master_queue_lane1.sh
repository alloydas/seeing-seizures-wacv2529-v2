#!/bin/bash
# MASTER QUEUE -- LANE 1 (GPU1). Runs alongside sh/_master_queue.sh on GPU0.
#
# Works the SAME backlog from the opposite end so the two lanes do not collide:
#
#   lane 0 (running):  stage2 EEG g3 -> EEG g5 -> video g3 -> video g5 -> stage3 -> #8
#   lane 1 (this):     stage2 video g5 (reverse fold order) -> stage3 seeds
#
# Lane 0 reaches video-g5 last (~50 h away) and stage 3 after that, so in practice the
# lanes never contend. If they ever do meet in the middle, the guard below stops the
# duplicate: unlike lane 0's job(), this checks BOTH results.json AND whether a live
# python process is already training into that output directory.
#
# Guard note: the pgrep pattern is matched against python processes only. A plain
# `pgrep -f <path>` also matches shells whose command line merely mentions the log
# path, which silently aborted a launch earlier in this session.
set -u
cd /path/to/repo

GPU="${GPU:-1}"
L=logs/master_queue
mkdir -p "$L" output/subject_cv output/seed_runs
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

DONE=0; SKIPPED=0; FAILED=0
log(){ echo "[$(date +%F_%T)] [lane1] $*"; }

busy(){   # is another lane already training into this output dir?
  pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"
}

job(){
  local name=$1 out=$2; shift 2
  if [ -f "$out/results.json" ]; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (results.json exists)"; return 0
  fi
  if busy "$out"; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (lane 0 is running it)"; return 0
  fi
  local try
  for try in 1 2; do
    log "START $name (try $try)"
    "$@" > "$L/lane1_$name.log" 2>&1
    local rc=$?
    if [ -f "$out/results.json" ]; then
      DONE=$((DONE+1)); log "DONE  $name (rc=$rc)"; return 0
    fi
    log "FAIL  $name try $try (rc=$rc); tail:"; tail -4 "$L/lane1_$name.log"
    sleep 60
  done
  FAILED=$((FAILED+1)); log "GAVE UP $name"; return 1
}

log "===== lane 1 up on gpu$GPU (sequential, opposite end of the queue) ====="

# ---- plan #2, video 5-class subject-disjoint folds, REVERSE order ------------------
log "--- plan #2: video 5-class subject-CV (folds 4->0) ---"
for f in 4 3 2 1 0; do
  job "vid_sub_g5_f$f" "output/subject_cv/vid_g5_fold$f" \
      python3 train_pooled.py --arch r2plus1d --split subject --fold "$f" \
        --n_folds 5 --epochs 12 --batch_size 16 --workers 8 --seed 42 --split_seed 49 \
        --output "output/subject_cv/vid_g5_fold$f"
done

# ---- plan #2, video 3-class subject-disjoint folds, REVERSE order ------------------
log "--- plan #2: video 3-class subject-CV (folds 4->0) ---"
for f in 4 3 2 1 0; do
  job "vid_sub_g3_f$f" "output/subject_cv/vid_g3_fold$f" \
      python3 train_pooled.py --arch r2plus1d --group3 --split subject --fold "$f" \
        --n_folds 5 --epochs 12 --batch_size 16 --workers 8 --seed 42 --split_seed 49 \
        --output "output/subject_cv/vid_g3_fold$f"
done

# ---- plan #1, remaining video seeds to n>=5 ---------------------------------------
log "--- plan #1: remaining video seeds ---"
for s in 3 4 5; do
  job "vid_g5_s$s" "output/seed_runs/vid_g5_s$s" \
      python3 train_pooled.py --arch r2plus1d --epochs 12 --batch_size 16 \
        --workers 8 --seed "$s" --split_seed 49 --output "output/seed_runs/vid_g5_s$s"
done
for s in 3 4; do
  job "vid_g3_s$s" "output/seed_runs/vid_g3_s$s" \
      python3 train_pooled.py --arch r2plus1d --group3 --epochs 12 --batch_size 16 \
        --workers 8 --seed "$s" --split_seed 49 --output "output/seed_runs/vid_g3_s$s"
done
job vid_s5 output/seed_runs/vid_s5 \
    python3 train_pooled.py --arch r2plus1d --group2 --epochs 12 --batch_size 16 \
      --workers 8 --seed 5 --split_seed 49 --output output/seed_runs/vid_s5

log "===== lane 1 finished: $DONE done, $SKIPPED skipped, $FAILED failed ====="
