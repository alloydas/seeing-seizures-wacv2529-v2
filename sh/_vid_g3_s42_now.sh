#!/bin/bash
# video 3-class at seed 42, launched in PARALLEL with vid_f64_s42 on GPU1.
#
# Why parallel rather than queued behind f64: this is the run that decides whether the
# "video 3-class baseline (0.7765) is stale" claim survives. That claim rests only on
# the baseline sitting far from two new seeds (0.7972 +/- 0.0027) -- the identical
# reasoning that FAILED for EEG 3-class, where a 4-sigma-looking gap collapsed to
# 0.1 sigma once seed 42 widened the spread from +/-0.0023 to +/-0.0194. Queued behind
# a ~12 h 64-frame run, that answer would not arrive until tomorrow.
#
# Risk, stated plainly: this puts 2 video trainers on GPU1 (plus 3 on GPU0). With NVML
# broken by the driver mismatch, memory pressure aborts hard rather than OOMing
# cleanly -- that is what killed jobs at 21:29 and 09:30. Retry loop included; if it
# aborts twice the honest read is that GPU1 has no room and this should wait for f64.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g3_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " && { echo "$name already running"; exit 0; }
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled.py --arch r2plus1d --group3 --epochs 12 \
      --batch_size 16 --workers 5 --seed 42 --split_seed 49 \
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
