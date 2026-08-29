#!/bin/bash
# EEG Conformer, 5-class, seed 42 -- third probe of tab:eegarch reproducibility.
#
# Deliberately an EEG run, not a video one: GPU1 already holds vid_f64_s42 (12 h,
# 40 min in) and vid_g3_s42. A third video trainer there is the configuration most
# likely to trigger the NVML abort, and it would cost hours of work already done.
# A Conformer is small enough to add almost no memory pressure.
#
# Why Conformer specifically. The 5-class arch table so far:
#     stored   TCN .4622 | conformer .4261 | XGB .4262 | EEGNet .3991 | LSTM .3595 | RF .2785
#     verified TCN .4665 (reproduced, +0.004)   GRU .4483 +/- .0267 (n=5, was stored .3598)
# Two probes gave opposite answers -- TCN reproduced, GRU did not -- so the remaining
# rows cannot be assumed either way. Conformer sits closest to the contested GRU/TCN
# band, so its reproducibility decides whether the table's middle is meaningful or
# whether the whole thing needs regenerating under current code.
#
# Original command (sh/_eeg_arch_gpu1.sh:12): --arch conformer, no group flag,
# all other defaults (seed 42, split_seed 49).
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_conformer_g5_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " && { echo "$name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch conformer --seed 42 --split_seed 49 \
      --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
  rc=$?
  if [ -f "output/seed_runs/$name/results.json" ]; then
    echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
  tail -3 "$L/$name.log"
  sleep 180
done
echo "[$(date +%F_%T)] GAVE UP on $name"
