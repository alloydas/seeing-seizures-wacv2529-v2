#!/bin/bash
# EEG EEGNet, 5-class, seed 42 -- fourth probe of tab:eegarch reproducibility.
# Original command (sh/_eeg_arch_gpu0.sh:13): --arch eegnet, no group flag, defaults.
#
# Again an EEG run rather than a video one: EEG trains from the npz window cache, so
# it adds neither meaningful GPU memory nor video-decode CPU, which is what the five
# concurrent video trainers are actually contending for.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_eegnet_g5_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " && { echo "$name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch eegnet --seed 42 --split_seed 49 \
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
