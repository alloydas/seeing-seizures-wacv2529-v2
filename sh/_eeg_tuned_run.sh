#!/bin/bash
# Generic runner for tuned-EEG-baseline jobs at plan #8's winning preprocessing
# (4 s window, 1 s stride, 125 Hz). Replaces the per-task one-offs
# (_eeg_tuned_baseline.sh, _eeg_tuned_g3.sh) now that seeds may follow.
#
#   Usage:  bash sh/_eeg_tuned_run.sh <gpu> <bin|g3|g5> [seed]
#   e.g.    nohup bash sh/_eeg_tuned_run.sh 0 bin > logs/eeg_tuned/_now_bin.log 2>&1 &
#           nohup bash sh/_eeg_tuned_run.sh 1 g5 7 > logs/eeg_tuned/_now_g5_s7.log 2>&1 &
#
# Reuses cache_abl/seg_w4_s1_d8.npz (built 2026-08-09 08:23, 453 MB). Same live-process
# guard as the queue lanes, so it is safe to fire alongside anything else.
set -u
cd /path/to/repo

GPU=${1:?usage: _eeg_tuned_run.sh <gpu> <bin|g3|g5> [seed]}
TASK=${2:?usage: _eeg_tuned_run.sh <gpu> <bin|g3|g5> [seed]}
SEED=${3:-42}

export CUDA_VISIBLE_DEVICES="$GPU"
L=logs/eeg_tuned; mkdir -p "$L" output/eeg_tuned
CACHE=cache_abl/seg_w4_s1_d8.npz
log(){ echo "[$(date +%F_%T)] $*"; }

case "$TASK" in
  bin) flag="--group2" ;;
  g3)  flag="--group3" ;;
  g5)  flag=""         ;;
  *)   echo "unknown task '$TASK' (want bin|g3|g5)"; exit 2 ;;
esac

suffix=$([ "$SEED" = "42" ] && echo "" || echo "_s$SEED")
out="output/eeg_tuned/gru_${TASK}_w4s1${suffix}"

[ -f "$CACHE" ] || { log "cache $CACHE missing -- build it first"; exit 1; }
busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }
[ -f "$out/results.json" ] && { log "$out already done"; exit 0; }
busy "$out" && { log "$out already running"; exit 0; }

for try in 1 2; do
  [ -f "$out/results.json" ] && break
  log "START tuned EEG $TASK seed=$SEED (try $try)"
  python3 train_pooled_eeg.py --arch gru $flag --cache "$CACHE" \
      --seed "$SEED" --split_seed 49 --output "$out" \
      > "$L/gru_${TASK}_w4s1${suffix}.log" 2>&1
  rc=$?
  [ -f "$out/results.json" ] && { log "DONE $out (rc=$rc)"; break; }
  log "FAILED try $try (rc=$rc); tail:"; tail -4 "$L/gru_${TASK}_w4s1${suffix}.log"; sleep 60
done
[ -f "$out/results.json" ] || log "GAVE UP $out"
