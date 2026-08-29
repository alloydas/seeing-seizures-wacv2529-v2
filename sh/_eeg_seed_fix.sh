#!/bin/bash
# Relaunch the 4 missing EEG seed runs on GPU1:
#   eeg_g3_s1  (3-class, seed 1 -- died silently 21:08)
#   eeg_g5_s1/s2/s3  (5-class -- never launched; second half of _eeg_grade_seeds.sh)
# Each run retries up to 3 times if it exits without writing results.json.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs

run_retry(){          # $1=name  $2..=train args
  local name=$1; shift
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    echo "[$(date +%F_%T)] START $name (try $try)"
    python3 train_pooled_eeg.py --arch gru "$@" --split_seed 49 \
        --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
    rc=$?
    if [ -f "output/seed_runs/$name/results.json" ]; then
      echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; return 0
    fi
    echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
    tail -3 "$L/$name.log"
    sleep 30
  done
  echo "[$(date +%F_%T)] GAVE UP on $name"; return 1
}

run_retry eeg_g3_s1 --group3 --seed 1 &
sleep 60
run_retry eeg_g5_s1 --seed 1 &
sleep 60
run_retry eeg_g5_s2 --seed 2 &
sleep 60
run_retry eeg_g5_s3 --seed 3 &
wait
echo "[$(date +%F_%T)] eeg seed fix done"
