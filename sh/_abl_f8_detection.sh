#!/bin/bash
# Close the last hole in the temporal-sampling grid: T=8 DETECTION under current code.
#
# The stored T=8 detection number (0.960) is from the Jul-25 sweep, i.e. before the
# 2026-08-03 train_pooled.py edit, so it cannot sit in a row with the T=1/4/16/32/64
# points measured this week. Every other cell of the 6x3 grid is either measured under
# current code or queued; this is the one remaining gap.
#
# Runs AFTER sh/_abl_frames_grading_lowT.sh finishes, so the two do not contend for the
# same GPU0 slot -- that script is itself parked waiting for the EEG folds to drain,
# and stacking a second R(2+1)D behind it would re-trigger the NVML abort that killed
# the 15:42 attempt.
#
# Params match the rest of the sweep: bs 32 holds frames-per-batch at 256, lr 3e-4,
# seed 42, split_seed 49, 12 epochs.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs
name=vid_f8_s42
out="output/seed_runs/$name"

# 1. wait for the low-T grading driver to finish (bracketed so this shell does not
#    match its own command line -- that self-match has bitten twice this session)
echo "[$(date +%F_%T)] waiting for the low-T grading driver to finish"
while pgrep -f "[_]abl_frames_grading_lowT.sh" > /dev/null; do sleep 120; done
echo "[$(date +%F_%T)] low-T driver done"

# 2. wait for GPU0 headroom
gpu0_jobs(){
  local n=0 p
  for p in $(pgrep -f "python3 train_pooled" 2>/dev/null); do
    grep -qz "CUDA_VISIBLE_DEVICES=0" "/proc/$p/environ" 2>/dev/null && n=$((n+1))
  done
  echo $n
}
echo "[$(date +%F_%T)] waiting for GPU0 <=1 job (currently $(gpu0_jobs))"
while [ "$(gpu0_jobs)" -gt 1 ]; do sleep 120; done

# 3. run
for try in 1 2; do
  [ -f "$out/results.json" ] && break
  echo "[$(date +%F_%T)] START $name (frames=8 bs=32, try $try)"
  python3 train_pooled.py --arch r2plus1d --group2 --frames 8 \
      --epochs 12 --batch_size 32 --lr 3e-4 --workers 5 \
      --seed 42 --split_seed 49 --output "$out" > "$L/$name.log" 2>&1
  rc=$?
  [ -f "$out/results.json" ] && { echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; break; }
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"; tail -3 "$L/$name.log"
  sleep 120
done
[ -f "$out/results.json" ] || echo "[$(date +%F_%T)] GAVE UP $name"

echo "[$(date +%F_%T)] ===== temporal-sampling grid, current code ====="
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
print('* mean of 4 seeds (vid_s1..s4)')
PY
