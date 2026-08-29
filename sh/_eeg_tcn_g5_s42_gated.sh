#!/bin/bash
# Backstop for eeg_tcn_g5_s42, which aborted at 09:30 with the NVML assert -- it was
# the 4th job on GPU1 (alongside vid_g3_s1 + eeg_bin_s42 + eeg_g3_s42). Its own retry
# loop may burn tries 2 and 3 against the same full GPU. This waits for that loop to
# exit, then for GPU1 to drain, then runs it once conditions are actually favourable.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_tcn_g5_s42
PEERS="vid_g3_s1 eeg_bin_s42 eeg_g3_s42"

alive(){ pgrep -f "seed_runs/$1\b" > /dev/null; }
nalive(){ local n=0; for p in $PEERS; do alive "$p" && n=$((n+1)); done; echo $n; }

# 1. let the original retry loop finish so we never double-launch
while alive "$name"; do sleep 60; done
[ -f "output/seed_runs/$name/results.json" ] && { echo "[$(date +%F_%T)] already done"; exit 0; }

# 2. wait for GPU1 to drop to <=1 peer (vid_g3_s1 is the memory hog; it is at ep10/12)
echo "[$(date +%F_%T)] waiting for GPU1 to drain ($(nalive) peers alive)"
while [ "$(nalive)" -gt 1 ]; do sleep 60; done
echo "[$(date +%F_%T)] GPU1 quiet ($(nalive) peers) -> launching $name"

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  alive "$name" && { echo "$name already running elsewhere"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch tcn --seed 42 --split_seed 49 \
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
