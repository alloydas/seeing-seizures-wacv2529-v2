#!/bin/bash
# Pull ONE queued subject-CV fold forward and run it immediately on a chosen GPU.
# Generic replacement for the per-fold _now.sh scripts.
#
#   Usage:  bash sh/_run_one_now.sh <gpu> <kind> <fold>
#     gpu   0 | 1
#     kind  vid_g5 | vid_g3 | eeg_g5 | eeg_g3
#     fold  0..4
#
#   e.g.  nohup bash sh/_run_one_now.sh 1 vid_g5 2 > logs/master_queue/_now_vid_g5_2.log 2>&1 &
#
# Safe to run alongside the three lanes: the guard checks BOTH results.json and a live
# python process writing to that output dir, matching python processes only (a plain
# `pgrep -f <path>` also matches shells that merely mention the log path -- that
# self-match silently aborted a launch earlier in this session). Whichever lane later
# reaches this fold will skip it.
set -u
cd /path/to/repo

GPU=${1:?usage: _run_one_now.sh <gpu> <kind> <fold>}
KIND=${2:?usage: _run_one_now.sh <gpu> <kind> <fold>}
FOLD=${3:?usage: _run_one_now.sh <gpu> <kind> <fold>}
WORKERS="${WORKERS:-6}"

L=logs/master_queue; mkdir -p "$L"
out="output/subject_cv/${KIND}_fold${FOLD}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

case "$KIND" in
  vid_g5) cmd=(python3 train_pooled.py --arch r2plus1d              --split subject --fold "$FOLD" --n_folds 5 --epochs 12 --batch_size 16 --workers "$WORKERS") ;;
  vid_g3) cmd=(python3 train_pooled.py --arch r2plus1d --group3     --split subject --fold "$FOLD" --n_folds 5 --epochs 12 --batch_size 16 --workers "$WORKERS") ;;
  eeg_g5) cmd=(python3 train_pooled_eeg.py --arch gru               --split subject --fold "$FOLD" --n_folds 5) ;;
  eeg_g3) cmd=(python3 train_pooled_eeg.py --arch gru --group3      --split subject --fold "$FOLD" --n_folds 5) ;;
  # plan #1 seed runs: the third arg is the SEED, not a fold. Session-disjoint split.
  vid_g5_seed) cmd=(python3 train_pooled.py --arch r2plus1d          --epochs 12 --batch_size 16 --workers "$WORKERS") ;;
  vid_g3_seed) cmd=(python3 train_pooled.py --arch r2plus1d --group3 --epochs 12 --batch_size 16 --workers "$WORKERS") ;;
  vid_det_seed) cmd=(python3 train_pooled.py --arch r2plus1d --group2 --epochs 12 --batch_size 16 --workers "$WORKERS") ;;
  *) echo "unknown kind '$KIND' (want vid_g5|vid_g3|eeg_g5|eeg_g3|vid_g5_seed|vid_g3_seed|vid_det_seed)"; exit 2 ;;
esac
case "$KIND" in
  *_seed)
    # seed runs live in output/seed_runs/, named by seed rather than fold
    case "$KIND" in
      vid_g5_seed)  out="output/seed_runs/vid_g5_s${FOLD}" ;;
      vid_g3_seed)  out="output/seed_runs/vid_g3_s${FOLD}" ;;
      vid_det_seed) out="output/seed_runs/vid_s${FOLD}" ;;
    esac
    cmd+=(--seed "$FOLD" --split_seed 49 --output "$out") ;;
  *)
    cmd+=(--seed 42 --split_seed 49 --output "$out") ;;
esac

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

[ -f "$out/results.json" ] && { echo "[$(date +%F_%T)] ${KIND}_fold${FOLD} already has results.json"; exit 0; }
busy "$out" && { echo "[$(date +%F_%T)] ${KIND}_fold${FOLD} already running in another lane"; exit 0; }

for try in 1 2; do
  [ -f "$out/results.json" ] && break
  echo "[$(date +%F_%T)] START ${KIND}_fold${FOLD} on gpu${GPU} (try $try)"
  "${cmd[@]}" > "$L/${KIND}_fold${FOLD}.log" 2>&1
  rc=$?
  if [ -f "$out/results.json" ]; then
    echo "[$(date +%F_%T)] DONE ${KIND}_fold${FOLD} (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED ${KIND}_fold${FOLD} try $try (rc=$rc); tail:"
  tail -4 "$L/${KIND}_fold${FOLD}.log"
  sleep 120
done
echo "[$(date +%F_%T)] GAVE UP on ${KIND}_fold${FOLD}"
