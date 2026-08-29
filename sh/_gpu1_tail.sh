#!/bin/bash
# Follow-on driver for GPU1, replacing sh/_gpu1_queue.sh (detached so vid_g3_s42 could
# run in parallel with vid_f64_s42; both are now orphaned from their launcher).
#
# Responsibilities:
#   1. wait until neither vid_f64_s42 nor vid_g3_s42 is running
#   2. restart either one if it died without writing results.json (nothing else will
#      now that the original driver is gone)
#   3. run vid_f32_s42 last -- the 32-frame re-run under current code, so vid_f64_s42
#      is not compared against the stale Jul-25 0.972
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs

running(){ pgrep -af "seed_runs/$1\b" | grep -qE "^[0-9]+ python3? "; }
done_(){ [ -f "output/seed_runs/$1/results.json" ]; }

echo "[$(date +%F_%T)] tail driver up; waiting for vid_f64_s42 + vid_g3_s42"
while running vid_f64_s42 || running vid_g3_s42; do sleep 120; done
echo "[$(date +%F_%T)] both cleared"

run_retry(){          # $1=name  $2..=extra train args
  local name=$1; shift
  for try in 1 2 3; do
    done_ "$name" && break
    running "$name" && { echo "$name already running"; return 0; }
    echo "[$(date +%F_%T)] START $name (try $try)"
    python3 train_pooled.py --arch r2plus1d "$@" --epochs 12 \
        --workers 5 --seed 42 --split_seed 49 \
        --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
    rc=$?
    if done_ "$name"; then echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; return 0; fi
    echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
    tail -3 "$L/$name.log"
    sleep 120
  done
  echo "[$(date +%F_%T)] GAVE UP on $name"; return 1
}

# rescue anything that died unattended
done_ vid_f64_s42 || { echo "[$(date +%F_%T)] vid_f64_s42 has no results -- rerunning"
                       run_retry vid_f64_s42 --group2 --frames 64 --batch_size 4 --lr 3e-4; }
done_ vid_g3_s42  || { echo "[$(date +%F_%T)] vid_g3_s42 has no results -- rerunning"
                       run_retry vid_g3_s42 --group3 --batch_size 16; }
# then the 32-frame anchor
run_retry vid_f32_s42 --group2 --frames 32 --batch_size 8 --lr 3e-4
echo "[$(date +%F_%T)] gpu1 tail queue done"
