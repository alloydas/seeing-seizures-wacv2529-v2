#!/bin/bash
# Seed replication for the ten video backbones (tab:vidarch), so that table can report
# mean +/- std like tab:results now does.
#
# Each cell currently has a single seed-42 run. This adds seeds 1, 2, 3, 5 for every
# backbone x task, i.e. 10 x 3 x 4 = 120 new runs.
#
# SEED-MAJOR ORDER, deliberately. The loop runs seed 1 across all thirty cells, then
# seed 2, and so on. If the sweep is stopped part-way, every cell has the *same* number
# of seeds and the table stays internally consistent -- whereas arch-major order would
# leave some rows at n=5 and others at n=1, which cannot be tabulated.
#
# Per-arch hyper-parameters are copied verbatim from the drivers that produced the
# seed-42 runs (sh/_video_arch.sh, sh/_video_grading.sh): batch 4 for the Swins, 8
# elsewhere, lr 1e-4 except TimeSformer at 5e-5, 12 epochs throughout. Changing them
# would make the new seeds incomparable with the existing seed-42 column.
#
# CONCURRENCY. Video training is CPU-decode-bound and the EEG Table 2 sweep is still
# running, so this uses 1 concurrent video job until the EEG sweep clears, then 2.
# Two video jobs plus the EEG sweep drove load to 41 on 32 cores earlier today.
set -u
cd /path/to/repo
L=logs/v3_vidseeds; mkdir -p "$L" output/v3_vidseeds
SEEDS="1 2 3 5"
# Which architectures this invocation covers. The sweep is split in two phases because
# the four frame-cache geometries total 207 GB against ~206 GB of page cache, and the
# seed-major order interleaves them continuously -- so all four live at once would thrash.
# Phase A runs the seven archs served by the 97 GB of NVMe caches; phase B rebuilds
# 32x224 on NVMe in their place for the remaining three.
ARCH_FILTER="${ARCH_FILTER:-r2plus1d mvit mvit_v1 x3d s3d videomae tsf slowfast swin swin_s}"
log(){ echo "[$(date +%F_%T)] $*"; }

# arch | trainer | extra args
# arch | trainer | extra args | frame cache (geometry must match the trainer's
# --frames/--size defaults; see build_frame_cache.py)
F112=cache_frames/f16s112
F224=cache_frames/f16s224
F8=cache_frames/f8s224
F32=cache_frames/f32s224
SPECS=(
  "r2plus1d|train_pooled.py --arch r2plus1d|--batch_size 16 --lr 1e-4|$F112"
  "mvit|train_pooled.py --arch mvit|--batch_size 8 --lr 1e-4|$F224"
  "mvit_v1|train_pooled.py --arch mvit_v1|--batch_size 8 --lr 1e-4|$F224"
  "x3d|train_pooled.py --arch x3d|--batch_size 8 --lr 1e-4|$F224"
  "slowfast|train_pooled.py --arch slowfast|--batch_size 8 --lr 1e-4|$F32"
  "s3d|train_pooled.py --arch s3d|--batch_size 8 --lr 1e-4|$F224"
  "swin|train_pooled.py --arch swin|--batch_size 4 --lr 1e-4|$F32"
  "swin_s|train_pooled.py --arch swin_s|--batch_size 4 --lr 1e-4|$F32"
  "videomae|train_pooled_videomae.py|--batch_size 8 --lr 1e-4|$F224"
  "tsf|train_pooled_timesformer.py|--batch_size 8 --lr 5e-5|$F8"
)

# Four jobs, two per GPU. This was 2 while training decoded video on the fly: back then
# 4 concurrent made every run 8x slower per epoch (2.7 -> 21.7) because H.264 decode
# saturated the CPU. The frame cache removed that entirely -- measured 2026-08-16 with
# one cached job running: 1.67 min/epoch (vs 2.7 uncached) at load 5.8 on 32 cores, with
# 2.9 MB read from disk against 204 MB served from page cache. Four jobs should land near
# load 23, still inside the core count. Re-measure min/epoch before raising further.
cap(){ echo "${CAP:-4}"; }
# Count JOBS, not processes. Each trainer spawns --workers DataLoader children that
# inherit the parent command line verbatim, so a single video run matches pgrep six
# times and a naive process count pinned the gate shut at one job forever. Distinct
# --output paths give the true job count.
nvid(){ pgrep -af "python3 train_pooled(_videomae|_timesformer)?\.py" 2>/dev/null \
          | grep -o -- "--output [^ ]*" | sort -u | wc -l; }
# Round-robin on a counter drifts badly once runs finish out of order -- the EEG sweep
# ended up 3 jobs on one card and 1 on the other that way. Pick the emptier card instead.
pick_gpu(){
  local g0=0 g1=0 p g
  for p in $(pgrep -f "python3 train_pooled"); do
    g=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ "$g" = "0" ] && g0=$(( g0 + 1 ))
    [ "$g" = "1" ] && g1=$(( g1 + 1 ))
  done
  [ "$g0" -le "$g1" ] && echo 0 || echo 1
}

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }

one(){
  local gpu=$1 arch=$2 trainer=$3 extra=$4 tag=$5 flag=$6 seed=$7 cache=$8
  local out="output/v3_vidseeds/${arch}_${tag}_s${seed}"
  [ -f "$out/results.json" ] && return 0
  busy "$out" && return 0
  local try
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 $trainer $flag --epochs 12 $extra \
        --workers 5 --seed "$seed" --split_seed 49 --cache_dir "$cache" --output "$out" \
        > "$L/${arch}_${tag}_s${seed}.log" 2>&1
    if [ -f "$out/results.json" ]; then
      # the checkpoint is dead weight once results.json + val_preds.npz exist: the table
      # generator reads val_preds.npz, and 120 of these would be ~19 GB that phase B's
      # 110 GB cache build needs. The run is reproducible from the seed if ever wanted.
      [ -f "$out/val_preds.npz" ] && [ "${KEEP_CKPT:-1}" = "0" ] && rm -f "$out/best.pt"
      log "DONE ${arch}_${tag}_s${seed}"; return 0
    fi
    log "retry $try failed ${arch}_${tag}_s${seed} -- tail:"
    tail -3 "$L/${arch}_${tag}_s${seed}.log" | cut -c1-140
    sleep 120
  done
  log "GAVE UP ${arch}_${tag}_s${seed}"
}

log "=== video backbone seeding: 10 archs x 3 tasks x 4 seeds, seed-major ==="
i=0
for seed in $SEEDS; do
  log "----- seed $seed -----"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r arch trainer extra cache <<< "$spec"
    case " $ARCH_FILTER " in *" $arch "*) ;; *) continue ;; esac
    for ts in "bin --group2" "g3 --group3" "g5 "; do
      set -- $ts; tag=$1; shift; flag="${*:-}"
      while [ "$(nvid)" -ge "$(cap)" ]; do sleep 60; done
      gpu=$(pick_gpu); i=$(( i + 1 ))
      one "$gpu" "$arch" "$trainer" "$extra" "$tag" "$flag" "$seed" "$cache" &
      sleep 20
    done
  done
  # No `wait` here any more. It used to barrier on the whole seed, idling one card for
  # the seed's imbalance tail. Sizing it honestly: the seed-1 boundary cost 7.9 h
  # (swin_s_bin_s1 finished 03:41, swin_s_g5_s1 ran to 11:29) only because seed 1 had
  # just 3 cells left to pack. Seeds 2/3/5 have all 9 outstanding, and greedy packing of
  # 3x2.9 h slowfast + 3x10 h swin + 3x9 h swin_s over 2 slots leaves a tail of only
  # ~1.9 h, so removing the barrier saves ~3.8 h across the two interior boundaries --
  # real, but small. Concurrency is ALREADY capped by the `nvid >= cap` gate above (the
  # barrier was never the throttle), so dropping it is safe regardless: the next seed's
  # first cell simply takes a slot the moment one frees, never exceeding CAP.
  log "----- seed $seed dispatched -----"
done

# The per-seed table refresh moved here for the same reason: calling it mid-loop would
# either block (reintroducing the barrier) or report a seed whose cells are still running.
# Run `python3 make_tab_vidarch_meansd.py` by hand any time for an up-to-date partial.
wait
log "===== video seeding complete ====="
python3 make_tab_vidarch_meansd.py 2>&1 | tail -20 || true
