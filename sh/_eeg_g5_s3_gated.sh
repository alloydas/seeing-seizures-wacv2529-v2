#!/bin/bash
# eeg_g5_s3 hit "CUDA error: out of memory" on GPU1 at 21:29 -- it was the 4th EEG
# job plus the R(2+1)D grading run that landed on GPU1 at the same second.
# Rather than burn its retries against a full GPU, wait for a slot to free, then run it.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_g5_s3
PEERS="eeg_g3_s1 eeg_g5_s1 eeg_g5_s2 vid_g3_s1"

alive(){ pgrep -f "seed_runs/$1\b" > /dev/null; }
nalive(){ local n=0; for p in $PEERS; do alive "$p" && n=$((n+1)); done; echo $n; }

# 1. let the original retry loop finish giving up so we never double-launch
while alive "$name"; do sleep 30; done
[ -f "output/seed_runs/$name/results.json" ] && { echo "already done"; exit 0; }

# 2. wait for GPU1 to drop to <=2 concurrent jobs
echo "[$(date +%F_%T)] waiting for a GPU1 slot ($(nalive) peers alive)"
while [ "$(nalive)" -gt 2 ]; do sleep 60; done
echo "[$(date +%F_%T)] slot free ($(nalive) peers alive)"

# 3. run with retries
for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch gru --seed 3 --split_seed 49 \
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
