#!/bin/bash
# MASTER QUEUE -- every remaining experiment, run strictly one at a time, in priority
# order. Nothing else is running; this is the whole backlog.
#
# Order is by value against the ablation plan, cheapest-first inside each stage:
#
#   Stage 1  RF + XGBoost seed-42 reproductions        2 jobs   CPU, ~10 min each
#            Closes the last two unverified tab:eegarch rows. Every other row has
#            been checked; only the EEG GRU failed to reproduce.
#
#   Stage 2  PLAN #2 -- subject-disjoint grading CV   20 jobs
#            The plan's most load-bearing item and the one both expert reviews put
#            first. Grading is currently validated on a session-disjoint split while
#            the paper claims subject-wise generalisation; this runs 3-class and
#            5-class, video and EEG, over the same 5 subject-disjoint folds already
#            used for detection. EEG folds first (~1 h) so the cheap half lands early.
#            NOTE: rare stages may be absent from some folds -- that is inherent to
#            subject-wise evaluation (see split_subjects docstring), and S4/S5 are
#            concentrated in a few animals, so expect high fold-to-fold variance.
#            That variance IS the result.
#
#   Stage 3  PLAN #1 -- remaining seeds to n>=5        6 jobs   video, ~4 h each
#            video 5-class needs 3, 3-class needs 2, detection needs 1.
#            The three EEG cells already reached n=5.
#
#   Stage 4  PLAN #8 -- EEG window/stride/rate/pooling  8 cells + cache builds
#            Delegates to sh/_abl_eeg_window.sh, which self-gates on video trainers
#            clearing -- correct here, since this queue is sequential.
#
# Every job is skipped if its results.json already exists, so the queue is resumable:
# kill it and relaunch and it picks up where it stopped.
#
# Usage:  nohup bash sh/_master_queue.sh > logs/master_queue/_driver.log 2>&1 &
#         GPU=1 nohup bash sh/_master_queue.sh > ... &     # to run the lane on GPU1
set -u
cd /path/to/repo

GPU="${GPU:-0}"
L=logs/master_queue
mkdir -p "$L" output/subject_cv output/seed_runs
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

TOTAL=0; DONE=0; FAILED=0; SKIPPED=0
log(){ echo "[$(date +%F_%T)] $*"; }

# job <name> <output_dir> <cmd...>
# Is another lane/driver already training into this output dir? Lane 0 originally
# checked only results.json, so on 2026-08-07 it started vid_g3_fold1 at 09:00 while a
# pulled-forward copy (started 23:17) was still running -- ~4 h of duplicated GPU0 work.
# Lanes 1/2 and _run_one_now.sh already had this guard; this backports it.
busy(){
  pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"
}

job(){
  local name=$1 out=$2; shift 2
  TOTAL=$((TOTAL+1))
  if [ -f "$out/results.json" ]; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (already has results.json)"; return 0
  fi
  if busy "$out"; then
    SKIPPED=$((SKIPPED+1)); log "SKIP $name (another lane is running it)"; return 0
  fi
  local try
  for try in 1 2; do
    log "START $name (try $try)"
    "$@" > "$L/$name.log" 2>&1
    local rc=$?
    if [ -f "$out/results.json" ]; then
      DONE=$((DONE+1)); log "DONE  $name (rc=$rc)"; return 0
    fi
    log "FAIL  $name try $try (rc=$rc); tail:"; tail -4 "$L/$name.log"
    sleep 60
  done
  FAILED=$((FAILED+1)); log "GAVE UP $name"; return 1
}

log "===== master queue up (gpu$GPU, strictly sequential) ====="

# ---------- Stage 1: classical reproductions (CPU) ---------------------------------
log "--- Stage 1: RF/XGB seed-42 reproductions ---"
job rf_g5_repro  output/subject_cv/rf_g5_repro \
    python3 train_pooled_eeg_classical.py --arch rf  --split_seed 49 \
      --output output/subject_cv/rf_g5_repro
job xgb_g5_repro output/subject_cv/xgb_g5_repro \
    python3 train_pooled_eeg_classical.py --arch xgb --split_seed 49 \
      --output output/subject_cv/xgb_g5_repro

# ---------- Stage 2: PLAN #2 subject-disjoint grading CV ---------------------------
log "--- Stage 2: plan #2, subject-disjoint 5-fold grading CV ---"
for f in 0 1 2 3 4; do          # EEG first: ~1 h per fold
  job "eeg_sub_g3_f$f" "output/subject_cv/eeg_g3_fold$f" \
      python3 train_pooled_eeg.py --arch gru --group3 --split subject --fold "$f" \
        --n_folds 5 --seed 42 --split_seed 49 --output "output/subject_cv/eeg_g3_fold$f"
done
for f in 0 1 2 3 4; do
  job "eeg_sub_g5_f$f" "output/subject_cv/eeg_g5_fold$f" \
      python3 train_pooled_eeg.py --arch gru --split subject --fold "$f" \
        --n_folds 5 --seed 42 --split_seed 49 --output "output/subject_cv/eeg_g5_fold$f"
done
for f in 0 1 2 3 4; do          # video: ~4 h per fold
  job "vid_sub_g3_f$f" "output/subject_cv/vid_g3_fold$f" \
      python3 train_pooled.py --arch r2plus1d --group3 --split subject --fold "$f" \
        --n_folds 5 --epochs 12 --batch_size 16 --workers 8 --seed 42 --split_seed 49 \
        --output "output/subject_cv/vid_g3_fold$f"
done
for f in 0 1 2 3 4; do
  job "vid_sub_g5_f$f" "output/subject_cv/vid_g5_fold$f" \
      python3 train_pooled.py --arch r2plus1d --split subject --fold "$f" \
        --n_folds 5 --epochs 12 --batch_size 16 --workers 8 --seed 42 --split_seed 49 \
        --output "output/subject_cv/vid_g5_fold$f"
done

# ---------- Stage 3: PLAN #1 remaining seeds to n>=5 -------------------------------
log "--- Stage 3: plan #1, remaining video seeds to n=5 ---"
for s in 3 4 5; do              # video 5-class: n=2 -> 5
  job "vid_g5_s$s" "output/seed_runs/vid_g5_s$s" \
      python3 train_pooled.py --arch r2plus1d --epochs 12 --batch_size 16 \
        --workers 8 --seed "$s" --split_seed 49 --output "output/seed_runs/vid_g5_s$s"
done
for s in 3 4; do                # video 3-class: n=3 -> 5
  job "vid_g3_s$s" "output/seed_runs/vid_g3_s$s" \
      python3 train_pooled.py --arch r2plus1d --group3 --epochs 12 --batch_size 16 \
        --workers 8 --seed "$s" --split_seed 49 --output "output/seed_runs/vid_g3_s$s"
done
job vid_s5 output/seed_runs/vid_s5 \
    python3 train_pooled.py --arch r2plus1d --group2 --epochs 12 --batch_size 16 \
      --workers 8 --seed 5 --split_seed 49 --output output/seed_runs/vid_s5

# ---------- Stage 4: PLAN #8 EEG window/stride/rate/pooling sweep ------------------
log "--- Stage 4: plan #8, EEG window/stride/rate/pooling sweep ---"
bash sh/_abl_eeg_window.sh > "$L/abl_eeg_window.log" 2>&1
log "stage 4 exited rc=$?"

# ---------- summary ----------------------------------------------------------------
log "===== master queue finished: $DONE done, $SKIPPED skipped, $FAILED failed, $TOTAL total ====="
python3 - <<'PY'
import json, os, glob, statistics as st
print('\n--- plan #2: subject-disjoint grading CV ---')
for tag, lbl in [('eeg_g3', 'EEG 3-class'), ('eeg_g5', 'EEG 5-class'),
                 ('vid_g3', 'video 3-class'), ('vid_g5', 'video 5-class')]:
    v = []
    for f in range(5):
        p = f'output/subject_cv/{tag}_fold{f}/results.json'
        if os.path.exists(p):
            v.append(json.load(open(p))['balanced_accuracy'])
    if v:
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {lbl:15s} folds={len(v)}  {st.mean(v):.4f} +/- {sd:.4f}  '
              f'[{min(v):.4f}-{max(v):.4f}]')
    else:
        print(f'  {lbl:15s} no folds completed')
print('\n--- plan #1: seed cells ---')
cells = {
    'video detection': ['vid_s1','vid_s2','vid_s3','vid_s4','vid_s5'],
    'video 3-class':   ['vid_g3_s1','vid_g3_s2','vid_g3_s42','vid_g3_s3','vid_g3_s4'],
    'video 5-class':   ['vid_g5_s1','vid_g5_s2','vid_g5_s3','vid_g5_s4','vid_g5_s5'],
}
for lbl, runs in cells.items():
    v = [json.load(open(f'output/seed_runs/{r}/results.json'))['balanced_accuracy']
         for r in runs if os.path.exists(f'output/seed_runs/{r}/results.json')]
    if len(v) > 1:
        print(f'  {lbl:15s} n={len(v)}  {st.mean(v):.4f} +/- {st.stdev(v):.4f}')
PY
