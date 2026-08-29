#!/bin/bash
# Tuned EEG baseline, 3-class -- companion to sh/_eeg_tuned_baseline.sh (5-class).
#
# The 33-job queue is finished; this is re-plan work. Reuses the 4 s / 1 s / 125 Hz
# cache built at 08:23 (cache_abl/seg_w4_s1_d8.npz, 453 MB), so there is no build cost
# and no CPU contention with the 5-class run on GPU0.
#
# Runs on GPU1, which is idle. Two EEG GRUs across two GPUs is well inside what the
# machine handled during the subject-CV phase.
#
# Rationale is the same as the 5-class run: plan #8 showed every published EEG number
# used a 6 s window that ranked third of four, and since the "you hobbled the EEG
# baseline" objection cannot be answered with a multi-channel montage (no EEG channel
# exists in these recordings), tuning is the only available answer. Both grading cells
# need the tuned number for that argument to cover the tables it appears in.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/eeg_tuned
mkdir -p "$L" output/eeg_tuned

CACHE=cache_abl/seg_w4_s1_d8.npz
out=output/eeg_tuned/gru_g3_w4s1
log(){ echo "[$(date +%F_%T)] $*"; }

[ -f "$CACHE" ] || { log "cache $CACHE missing -- run sh/_eeg_tuned_baseline.sh first"; exit 1; }

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }
[ -f "$out/results.json" ] && { log "already done"; exit 0; }
busy "$out" && { log "already running"; exit 0; }

for try in 1 2; do
  [ -f "$out/results.json" ] && break
  log "START tuned EEG 3-class (try $try)"
  python3 train_pooled_eeg.py --arch gru --group3 --cache "$CACHE" \
      --seed 42 --split_seed 49 --output "$out" > "$L/gru_g3_w4s1.log" 2>&1
  rc=$?
  [ -f "$out/results.json" ] && { log "DONE (rc=$rc)"; break; }
  log "FAILED try $try (rc=$rc); tail:"; tail -4 "$L/gru_g3_w4s1.log"; sleep 60
done

log "===== tuned vs production EEG baseline, 3-class ====="
python3 - <<'PY'
import json, os
p = 'output/eeg_tuned/gru_g3_w4s1/results.json'
if os.path.exists(p):
    r = json.load(open(p))
    print(f"  tuned      (4s/1s/125Hz): bal_acc={r['balanced_accuracy']:.4f} "
          f"macro_f1={r['macro_f1']:.4f}")
    print(f"  production (6s/3s/125Hz): bal_acc=0.6063 +/- 0.0180 (n=5)")
    print(f"  recalls: { {k: round(v['recall'], 3) for k, v in r['per_class'].items()} }")
else:
    print('  no results.json')
PY
