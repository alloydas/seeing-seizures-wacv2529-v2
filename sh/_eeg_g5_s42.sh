#!/bin/bash
# Reproduce the 5-class EEG baseline at its original seed (42) on the same split.
#
# Why: output/eeg_pooled20_g5 reports balanced_acc 0.360, but seeds 1/2/3 just gave
# 0.441/0.408/0.450 on an identical val set (same per-class supports). Either seed 42
# is an unlucky draw -- in which case tab:eegarch should quote mean+/-std over 4 seeds
# instead of the single 0.360 -- or the baseline dir is stale w.r.t. current code and
# should be discarded. This run decides which.
#
# Runs on GPU1 alongside vid_g3_s1. GRU is small (149.8k params), but note that GPU1
# memory pressure is what triggered the NVML-assert crashes at 21:29.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_g5_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch gru --seed 42 --split_seed 49 \
      --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
  rc=$?
  if [ -f "output/seed_runs/$name/results.json" ]; then
    echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
  tail -3 "$L/$name.log"
  sleep 120
done
echo "[$(date +%F_%T)] GAVE UP on $name"
