#!/bin/bash
# PLAN ITEM #4 (Tier 1, blocking, "Small") -- single-frame appearance-only baseline.
#
# Defends the "video-understanding problem" framing. The plan's warning is direct:
# your own frames ablation (0.960 @ 8 -> 0.970 @ 16 -> 0.972 @ 32) "hints a single
# frame may already reach ~0.95 -- either you prove temporal dynamics matter, or you
# learn to soften the framing." At a video venue this is close to mandatory.
#
# The plan asks for "a 2D image classifier (or the backbone with T=1)" plus the T=1
# and T=4 points added to the frames table. This runs the backbone at T=1 and T=4,
# holding everything else fixed against the existing detection seeds (which are the
# T=16 point under current code: 0.9681 +/- 0.0024 macro-F1 over 4 seeds).
#
# Pre-flight checks already done:
#   - build_video_model("r2plus1d") forward pass OK at T=1,2,4,8
#   - load_clip(clip, 1, 112) -> (3,1,112,112), all finite
#
# Cost: T=1 reads one frame per clip instead of 16, so this is by far the cheapest
# run in the queue -- it should finish in ~1-2 h even against the current load,
# versus 34 h remaining on vid_f64_s42. Chosen for GPU1 for exactly that reason.
#
# Batch is raised to keep frames-per-batch sane (T=1/bs64, T=4/bs32); lr 3e-4 and
# seed 42 match the frames sweep in sh/_video_more.sh so the new points slot into
# the same table.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
L=logs/seed_runs

run_retry(){          # $1=name  $2=frames  $3=batch
  local name=$1 fr=$2 bs=$3 try
  for try in 1 2 3; do
    [ -f "output/seed_runs/$name/results.json" ] && break
    pgrep -af "seed_runs/$name\b" | grep -qE "^[0-9]+ python3? " \
      && { echo "[$(date +%F_%T)] $name already running"; return 0; }
    echo "[$(date +%F_%T)] START $name (frames=$fr bs=$bs, try $try)"
    python3 train_pooled.py --arch r2plus1d --group2 --frames "$fr" \
        --epochs 12 --batch_size "$bs" --lr 3e-4 --workers 4 \
        --seed 42 --split_seed 49 \
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

run_retry vid_f1_s42 1 64
run_retry vid_f4_s42 4 32

echo "[$(date +%F_%T)] ===== single-frame baseline (plan #4) done ====="
python3 - <<'PY'
import json, os, statistics as st
print("Temporal-sampling, detection macro-F1 (current code, seed 42 unless noted)")
for n, lbl in [('vid_f1_s42', 'T=1  (appearance only)'), ('vid_f4_s42', 'T=4')]:
    p = f'output/seed_runs/{n}/results.json'
    if os.path.exists(p):
        r = json.load(open(p))
        print(f"  {lbl:24s} macro_f1={r['macro_f1']:.4f}  bal_acc={r['balanced_accuracy']:.4f}")
    else:
        print(f"  {lbl:24s} MISSING")
v = [json.load(open(f'output/seed_runs/vid_s{s}/results.json'))['macro_f1'] for s in (1, 2, 3, 4)]
print(f"  {'T=16 (4 seeds)':24s} macro_f1={st.mean(v):.4f} +/- {st.stdev(v):.4f}")
PY
