#!/bin/bash
# Run the first video GRADING seed on GPU1 instead of waiting for the GPU0
# orchestrator (which is blocked behind vid_s1 -> vid_s2, ~07:00 tomorrow).
#
# First attempt (21:28) failed: launching R(2+1)D alongside 4 EEG jobs pushed GPU1
# into memory pressure, and because nvidia-smi/NVML is broken by the driver/library
# mismatch, PyTorch's caching allocator aborts with
#   NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
# instead of a clean OOM. So: wait until GPU1 is clear of EEG jobs, then run alone.
#
# sh/video_grade_orch.py skips any run that is already running or has results.json,
# so this will not double-launch.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_g3_s1
EEG="eeg_g3_s1 eeg_g5_s1 eeg_g5_s2 eeg_g5_s3"

eeg_busy(){ for e in $EEG; do pgrep -f "seed_runs/$e\b" > /dev/null && return 0; done; return 1; }

echo "[$(date +%F_%T)] waiting for GPU1 EEG jobs to clear"
while eeg_busy; do sleep 120; done
echo "[$(date +%F_%T)] GPU1 clear -> launching $name"

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  pgrep -f "seed_runs/$name\b" > /dev/null && { echo "$name already running elsewhere"; exit 0; }
  echo "[$(date +%F_%T)] START $name on gpu1 (try $try)"
  python3 train_pooled.py --arch r2plus1d --group3 --seed 1 \
      --epochs 12 --batch_size 16 --workers 6 --split_seed 49 \
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
