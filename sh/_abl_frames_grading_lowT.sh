#!/bin/bash
# Plan item #4, grading half -- the T=1 and T=4 points for 3-class and 5-class.
#
# Gap this closes: #4 asks to "add the T=1 and T=4 points to the frames table", but the
# runs done so far were --group2 only (vid_f1_s42 = 0.5688, vid_f4_s42 = 0.9404, both
# detection). The frames-GRADING ablation covered T=8/16/32/64. So the grading rows
# have no low-T points at all, and the single-frame result -- the one that actually
# defends the "video-understanding" framing -- exists only for detection.
#
# This matters more for grading than it did for detection. Detection collapsed to
# 0.5688 at T=1, which is near chance for a binary task and settles the question. For
# 3-class and 5-class, chance is much lower, so a single frame could plausibly score
# well above chance on class priors alone -- and grading kept improving with frames
# (3-class +0.048 from T=8 to T=64) where detection had saturated. The low-T shape of
# the grading curve is therefore genuinely unknown.
#
# Run on GPU0: its co-tenants are EEG GRUs off the npz cache, so there is no video
# decode to contend with there, unlike GPU1's three video folds. Sequential, cheapest
# first. Batch scaled to hold frames-per-batch at 256, matching the rest of the sweep
# (T=1 -> bs 64 as used by vid_f1_s42; T=4 -> bs 32). lr 3e-4, seed 42, split_seed 49.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

# Count trainers pinned to GPU0. The first attempt at 15:42 aborted with
#   NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
# because three EEG CUDA contexts plus an R(2+1)D exceeded GPU0. NVML is broken by the
# driver/library mismatch, so memory pressure aborts hard instead of raising a clean
# OOM. Wait for the EEG folds to drain rather than fight them.
gpu0_jobs(){
  local n=0 p
  for p in $(pgrep -f "python3 train_pooled" 2>/dev/null); do
    grep -qz "CUDA_VISIBLE_DEVICES=0" "/proc/$p/environ" 2>/dev/null && n=$((n+1))
  done
  echo $n
}
echo "[$(date +%F_%T)] waiting for GPU0 to drop to <=1 job (currently $(gpu0_jobs))"
while [ "$(gpu0_jobs)" -gt 1 ]; do sleep 120; done
echo "[$(date +%F_%T)] GPU0 has room -- starting"

# $1=name  $2=frames  $3=batch  $4=task flag ("--group3" or "")
run_one(){
  local name=$1 fr=$2 bs=$3 flag=$4 out="output/seed_runs/$1" try
  [ -f "$out/results.json" ] && { echo "[$(date +%F_%T)] SKIP $name (done)"; return 0; }
  busy "$out" && { echo "[$(date +%F_%T)] SKIP $name (running elsewhere)"; return 0; }
  for try in 1 2; do
    echo "[$(date +%F_%T)] START $name (frames=$fr bs=$bs, try $try)"
    python3 train_pooled.py --arch r2plus1d $flag --frames "$fr" \
        --epochs 12 --batch_size "$bs" --lr 3e-4 --workers 5 \
        --seed 42 --split_seed 49 --output "$out" > "$L/$name.log" 2>&1
    rc=$?
    [ -f "$out/results.json" ] && { echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; return 0; }
    echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"; tail -3 "$L/$name.log"
    sleep 120
  done
  echo "[$(date +%F_%T)] GAVE UP $name"; return 1
}

run_one vid_g3_f1 1  64 "--group3"
run_one vid_g5_f1 1  64 ""
run_one vid_g3_f4 4  32 "--group3"
run_one vid_g5_f4 4  32 ""

echo "[$(date +%F_%T)] ===== plan #4 grading low-T complete ====="
python3 - <<'PY'
import json, os
print(f"{'T':>4s}  {'detection':>11s}  {'3-class':>11s}  {'5-class':>11s}   (macro-F1)")
for T in (1, 4, 8, 16, 32, 64):
    cells = []
    for name in (f'vid_f{T}_s42', f'vid_g3_f{T}', f'vid_g5_f{T}'):
        p = f'output/seed_runs/{name}/results.json'
        cells.append(f"{json.load(open(p))['macro_f1']:.4f}" if os.path.exists(p) else '--')
    if T == 16 and cells[0] == '--':
        cells[0] = '0.9681*'
    print(f'{T:4d}  {cells[0]:>11s}  {cells[1]:>11s}  {cells[2]:>11s}')
print('* mean of 4 seeds')
PY
