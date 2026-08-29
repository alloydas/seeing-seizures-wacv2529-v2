#!/bin/bash
# Second video job on GPU1: vid_g5_fold3, pulled forward from lane 1's sequential list.
#
# Lane 1 runs video 5-class folds 4->3->2->1->0 one at a time. Adding fold 3 alongside
# fold 4 gives that fold set two workers, so plan #2's video 5-class result -- a
# complete 5-fold spread, which is the number that actually matters -- lands sooner
# instead of trickling in over four sequential runs.
#
# Two video jobs on one GPU measured well earlier this week (~0.41 h/epoch for an
# identical config); the degradation only appeared past ~4 concurrent video trainers.
#
# Same busy() guard as the lanes: checks results.json AND a live python process
# writing to that output dir, matching python processes only (a plain `pgrep -f
# <path>` also matches shells that merely mention the log path, which silently
# aborted a launch earlier in this session). Lane 1 will skip fold 3 when it arrives.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/master_queue
out=output/subject_cv/vid_g5_fold3

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

[ -f "$out/results.json" ] && { echo "[$(date +%F_%T)] already has results.json"; exit 0; }
busy "$out" && { echo "[$(date +%F_%T)] a lane is already running it"; exit 0; }

for try in 1 2; do
  [ -f "$out/results.json" ] && break
  echo "[$(date +%F_%T)] START vid_g5_fold3 (try $try)"
  python3 train_pooled.py --arch r2plus1d --split subject --fold 3 \
      --n_folds 5 --epochs 12 --batch_size 16 --workers 6 \
      --seed 42 --split_seed 49 --output "$out" > "$L/vid_g5_fold3.log" 2>&1
  rc=$?
  if [ -f "$out/results.json" ]; then
    echo "[$(date +%F_%T)] DONE vid_g5_fold3 (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED try $try (rc=$rc); tail:"; tail -4 "$L/vid_g5_fold3.log"
  sleep 120
done
echo "[$(date +%F_%T)] GAVE UP on vid_g5_fold3"
