#!/bin/bash
# 5th seed for the 5-class EEG cell -- the only cell with meaningful seed variance.
#
# Seeds 1/2/3 gave balanced_acc 0.441/0.408/0.450 and the seed-42 baseline gave 0.360,
# i.e. std ~0.036 vs ~0.001-0.002 for detection and 3-class. With eeg_g5_s42 re-running
# seed 42, this makes 5 points for the mean+/-std that tab:eegarch needs.
#
# Third job on GPU1 (with vid_g3_s1 + eeg_g5_s42). Three concurrent EEG GRUs ran fine
# earlier tonight; the 21:29 crashes needed 4 EEG jobs plus an R(2+1)D.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_g5_s5

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch gru --seed 5 --split_seed 49 \
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
