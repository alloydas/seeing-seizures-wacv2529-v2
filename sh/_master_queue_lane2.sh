#!/bin/bash
# MASTER QUEUE -- LANE 2 (GPU0, EEG only). Third lane, alongside:
#   lane 0  sh/_master_queue.sh        gpu0  EEG g3 f0->f4, then EEG g5 f0->f4, then video, ...
#   lane 1  sh/_master_queue_lane1.sh  gpu1  video g5 f4->f0, then video g3 f4->f0, then seeds
#   lane 2  this                       gpu0  EEG g5 f4->f0, then EEG g3 f4->f0
#
# EEG-only on purpose. These are 150k-param GRUs trained from the npz window cache:
# they use little GPU memory and, critically, NO video decode -- which is the resource
# the video lanes actually contend for (32 cores, and identical configs ran 3-7x slower
# at five concurrent video jobs). So this lane adds throughput without slowing
# vid_g5_fold4 on gpu1 or the video stages lane 0 reaches later.
#
# Reverse fold order so lane 2 and lane 0 converge on the EEG folds from both ends.
# The busy() guard checks BOTH results.json and a live python process writing to that
# output dir, so whichever lane arrives second skips instead of duplicating.
set -u
cd /path/to/repo

GPU="${GPU:-0}"
L=logs/master_queue
mkdir -p "$L" output/subject_cv
export CUDA_VISIBLE_DEVICES="$GPU"

DONE=0; SKIPPED=0; FAILED=0
log(){ echo "[$(date +%F_%T)] [lane2] $*"; }

busy(){
  pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"
}

job(){
  local name=$1 out=$2; shift 2
  if [ -f "$out/results.json" ]; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (results.json exists)"; return 0
  fi
  if busy "$out"; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (another lane is running it)"; return 0
  fi
  local try
  for try in 1 2; do
    log "START $name (try $try)"
    "$@" > "$L/lane2_$name.log" 2>&1
    local rc=$?
    if [ -f "$out/results.json" ]; then
      DONE=$((DONE+1)); log "DONE  $name (rc=$rc)"; return 0
    fi
    log "FAIL  $name try $try (rc=$rc); tail:"; tail -4 "$L/lane2_$name.log"
    sleep 60
  done
  FAILED=$((FAILED+1)); log "GAVE UP $name"; return 1
}

log "===== lane 2 up on gpu$GPU (EEG subject-CV, reverse fold order) ====="

log "--- plan #2: EEG 5-class subject-CV (folds 4->0) ---"
for f in 4 3 2 1 0; do
  job "eeg_sub_g5_f$f" "output/subject_cv/eeg_g5_fold$f" \
      python3 train_pooled_eeg.py --arch gru --split subject --fold "$f" \
        --n_folds 5 --seed 42 --split_seed 49 --output "output/subject_cv/eeg_g5_fold$f"
done

log "--- plan #2: EEG 3-class subject-CV (folds 4->0) ---"
for f in 4 3 2 1 0; do
  job "eeg_sub_g3_f$f" "output/subject_cv/eeg_g3_fold$f" \
      python3 train_pooled_eeg.py --arch gru --group3 --split subject --fold "$f" \
        --n_folds 5 --seed 42 --split_seed 49 --output "output/subject_cv/eeg_g3_fold$f"
done

log "===== lane 2 finished: $DONE done, $SKIPPED skipped, $FAILED failed ====="
python3 - <<'PY'
import json, os, statistics as st
for tag, lbl in [('eeg_g3', 'EEG 3-class'), ('eeg_g5', 'EEG 5-class')]:
    v = []
    for f in range(5):
        p = f'output/subject_cv/{tag}_fold{f}/results.json'
        if os.path.exists(p):
            v.append(json.load(open(p))['balanced_accuracy'])
    if v:
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {lbl:14s} folds={len(v)}  {st.mean(v):.4f} +/- {sd:.4f}  '
              f'[{min(v):.4f}-{max(v):.4f}]')
PY
