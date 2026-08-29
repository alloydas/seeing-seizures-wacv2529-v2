#!/bin/bash
# Temporal-sampling ablation for GRADING: frames in {8,16,32,64} x {3-class, 5-class}.
#
# The existing frames table (tab:frames) is detection-only and pre-dates the
# 2026-08-03 train_pooled.py edit. This builds the grading equivalent from scratch
# under current code, so all 8 cells are mutually comparable.
#
# 8 runs total, arranged as two per-GPU streams that run their 4 budgets sequentially:
#     GPU0 : 3-class  8 -> 16 -> 32 -> 64
#     GPU1 : 5-class  8 -> 16 -> 32 -> 64
# Each stream waits until its own GPU has no train_pooled job left, so this queues
# behind the runs already in flight rather than stacking on them (memory pressure
# aborts hard here -- NVML is broken by the driver mismatch).
#
# Batch size is scaled to hold frames-per-batch constant at 256, following the
# original sweep (sh/_video_more.sh:14-15 used f8/bs16 and f32/bs8):
#     f8 -> bs 32   f16 -> bs 16   f32 -> bs 8   f64 -> bs 4
# lr 3e-4 throughout, matching that sweep. seed 42, split_seed 49, 12 epochs.
#
# Runtime: roughly 3 + 5 + 6 + 12 = ~26 h per stream, the two streams in parallel.
#
# Usage:  nohup bash sh/_abl_frames_grading.sh > logs/seed_runs/_abl_frames_grading.log 2>&1 &
cd /path/to/repo
L=logs/seed_runs
mkdir -p "$L"

# frames -> batch size (frames*batch = 256)
bs_for(){ case "$1" in 8) echo 32;; 16) echo 16;; 32) echo 8;; 64) echo 4;; *) echo 8;; esac; }

# count python train_pooled jobs pinned to a given CUDA device
gpu_jobs(){
  local dev=$1 n=0 p
  for p in $(pgrep -f "python3 train_pooled.py" 2>/dev/null); do
    grep -qz "CUDA_VISIBLE_DEVICES=$dev" "/proc/$p/environ" 2>/dev/null && n=$((n+1))
  done
  echo $n
}

# one stream: $1=gpu  $2=tag (g3|g5)  $3=extra train arg ("--group3" or "")
stream(){
  local dev=$1 tag=$2 flag=$3

  echo "[$(date +%F_%T)] [$tag] waiting for gpu$dev to drain (currently $(gpu_jobs "$dev") jobs)"
  while [ "$(gpu_jobs "$dev")" -gt 0 ]; do sleep 120; done
  echo "[$(date +%F_%T)] [$tag] gpu$dev free -- starting 4 frame budgets"

  for fr in 8 16 32 64; do
    local name="vid_${tag}_f${fr}" bs
    bs=$(bs_for "$fr")
    for try in 1 2 3; do
      [ -f "output/seed_runs/$name/results.json" ] && break
      # guard must match only real trainers -- a plain pgrep -f also matches shells
      # whose command line mentions the log path
      pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
        && { echo "[$(date +%F_%T)] [$tag] $name already running"; break; }
      echo "[$(date +%F_%T)] [$tag] START $name (frames=$fr bs=$bs, try $try)"
      CUDA_VISIBLE_DEVICES=$dev PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
      python3 train_pooled.py --arch r2plus1d $flag --frames "$fr" \
          --epochs 12 --batch_size "$bs" --lr 3e-4 --workers 5 \
          --seed 42 --split_seed 49 \
          --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
      rc=$?
      if [ -f "output/seed_runs/$name/results.json" ]; then
        echo "[$(date +%F_%T)] [$tag] DONE $name (rc=$rc)"; break
      fi
      echo "[$(date +%F_%T)] [$tag] FAILED $name try $try (rc=$rc); tail:"
      tail -3 "$L/$name.log"
      sleep 180
    done
    [ -f "output/seed_runs/$name/results.json" ] || \
      echo "[$(date +%F_%T)] [$tag] GAVE UP on $name"
  done
  echo "[$(date +%F_%T)] [$tag] stream done"
}

stream 0 g3 "--group3" &
stream 1 g5 ""         &
wait

echo "[$(date +%F_%T)] ===== frames-grading ablation complete ====="
python3 - <<'PY'
import json, os
print(f"{'run':16s} {'frames':>6s} {'bal_acc':>8s} {'macro_f1':>9s} {'acc':>7s}")
for tag in ('g3', 'g5'):
    for fr in (8, 16, 32, 64):
        n = f'vid_{tag}_f{fr}'
        p = f'output/seed_runs/{n}/results.json'
        if os.path.exists(p):
            r = json.load(open(p))
            print(f"{n:16s} {fr:6d} {r['balanced_accuracy']:8.4f} "
                  f"{r['macro_f1']:9.4f} {r['accuracy']:7.4f}")
        else:
            print(f"{n:16s} {fr:6d} {'MISSING':>8s}")
PY
