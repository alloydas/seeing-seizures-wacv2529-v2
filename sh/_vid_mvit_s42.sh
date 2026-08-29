#!/bin/bash
# Reproduce the top-ranked video backbone (MViT, bal_acc 0.9745, written 2026-07-24)
# at its original settings under current code, on GPU0.
#
# Two reasons this is the run worth doing:
#
# 1. Same staleness exposure as the EEG table. train_pooled.py has 131 uncommitted
#    lines with mtime 2026-08-03 16:02:05 -- the same edit session as
#    train_pooled_eeg.py (16:02:05), five minutes before the seed sweep. Every
#    vid_*_bin result was written 2026-07-22..07-25, all pre-edit. The EEG twin of
#    this situation moved the GRU 5-class number from 0.360 to 0.448.
#
# 2. The video arch ranking is already inside seed noise. R(2+1)D detection measures
#    0.9681 +/- 0.0024 over 4 seeds, but the stored table separates the top five by
#    less than 2 std:
#        mvit .9745 | slowfast .9737 | r2p1d_f32 .9716 | r2plus1d .9705 | x3d .9704
#    No backbone other than R(2+1)D has seed replicates, so a 0.004 gap is being read
#    as a ranking when it is roughly 1.7 sigma of a single-seed measurement.
#
# Original command (sh/_video_arch.sh:15): --arch mvit --group2 --epochs 12
#   --batch_size 8 --lr 1e-4, defaults seed=42 split_seed=49.
# Fewer workers (5) than default: grading runs are competing for video decode.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_mvit_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  echo "[$(date +%F_%T)] START $name on gpu0 (try $try)"
  python3 train_pooled.py --arch mvit --group2 --epochs 12 \
      --batch_size 8 --lr 1e-4 --workers 5 --seed 42 --split_seed 49 \
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
