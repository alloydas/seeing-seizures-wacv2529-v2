#!/bin/bash
# Seed-42 reproductions of the two remaining Jul-22 EEG baselines, on GPU1.
#
# Context: the 5-class baseline (output/eeg_pooled20_g5, bal_acc 0.360) did NOT
# reproduce under current code -- eeg_g5_s42 gave 0.478 on an identical val split,
# and all five current-code seeds land at 0.448 +/- 0.027, i.e. the baseline is below
# the entire distribution. Cause: train_pooled_eeg.py gained ~134 uncommitted lines on
# 2026-08-03 (multi-arch + configurable seed), which shifts RNG consumption before
# weight init, so nominal seed 42 no longer reproduces the Jul-22 initialization.
#
# These two runs test whether the same drift affects the other baselines:
#   eeg_pooled20_bin  bal_acc 0.8476  vs seeds 0.855-0.857
#   eeg_pooled20_g3   bal_acc 0.6010  vs seeds 0.610-0.615
# Both gaps are small and in the same (low) direction as g5 was.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs

run_retry(){          # $1=name  $2..=train args
  local name=$1; shift
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    echo "[$(date +%F_%T)] START $name (try $try)"
    python3 train_pooled_eeg.py --arch gru "$@" --seed 42 --split_seed 49 \
        --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
    rc=$?
    if [ -f "output/seed_runs/$name/results.json" ]; then
      echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; return 0
    fi
    echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
    tail -3 "$L/$name.log"
    sleep 120
  done
  echo "[$(date +%F_%T)] GAVE UP on $name"; return 1
}

run_retry eeg_bin_s42 --group2 &
sleep 60
run_retry eeg_g3_s42 --group3 &
wait
echo "[$(date +%F_%T)] eeg baseline reproductions done"
