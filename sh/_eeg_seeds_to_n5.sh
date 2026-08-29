#!/bin/bash
# Plan item #1 -- close the two cheapest n>=5 gaps, on GPU0.
#
# The frames-grading queue is empty (all 8 cells done or running), so this is the next
# identified work. Plan #1 asks for >=5 seeds per cell; the current shortfall is:
#     video 5-class   n=2  (needs 3)   ~6 h each on a loaded machine
#     video 3-class   n=3  (needs 2)   ~6 h each
#     video detection n=4  (needs 1)   ~6 h
#     EEG detection   n=4  (needs 1)   <1 h
#     EEG 3-class     n=4  (needs 1)   <1 h
#
# The two EEG cells are picked first deliberately: they are GRU runs that train from
# the npz window cache, so they cost minutes rather than hours and -- critically --
# they add no video-decode load. Five video trainers are already running at load ~76,
# which is the regime where identical configs ran 3-7x slower yesterday. Adding a
# sixth video job would slow all five; these two do not.
#
# Completing them takes EEG detection and EEG 3-class from n=4 to n=5, which is two of
# the six cells fully satisfying plan #1's seed requirement.
#
# Seed 7 chosen: unused so far (1,2,3,5,42 are taken across the EEG cells).
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
L=logs/seed_runs

run_retry(){          # $1=name  $2=task-flag
  local name=$1 flag=$2 try
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
      && { echo "[$(date +%F_%T)] $name already running"; return 0; }
    echo "[$(date +%F_%T)] START $name (try $try)"
    python3 train_pooled_eeg.py --arch gru $flag --seed 7 --split_seed 49 \
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

run_retry eeg_s7    --group2
run_retry eeg_g3_s7 --group3

echo "[$(date +%F_%T)] ===== EEG cells at n=5 ====="
python3 - <<'PY'
import json, os, statistics as st
for lbl, runs in [('EEG detection', ['eeg_s1','eeg_s2','eeg_s3','eeg_bin_s42','eeg_s7']),
                  ('EEG 3-class',   ['eeg_g3_s1','eeg_g3_s2','eeg_g3_s3','eeg_g3_s42','eeg_g3_s7'])]:
    v = [json.load(open(f'output/seed_runs/{r}/results.json'))['balanced_accuracy']
         for r in runs if os.path.exists(f'output/seed_runs/{r}/results.json')]
    if len(v) > 1:
        print(f'  {lbl:16s} n={len(v)}  {st.mean(v):.4f} +/- {st.stdev(v):.4f}')
PY
