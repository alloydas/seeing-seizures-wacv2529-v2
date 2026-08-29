#!/bin/bash
# Re-train the EEG baseline at the configuration plan #8 just identified as best,
# instead of the 6 s / 3 s / 125 Hz used to produce every published EEG number.
#
# Why this is the highest-value job now that the queue is empty:
#
#   Plan #8 found the EEG baseline is under-tuned on all three preprocessing axes.
#   Its best cell was s1 = 4 s window, 1 s stride, 125 Hz: bal_acc 0.4740 / macro_f1
#   0.3672, against 0.4483 +/- 0.0267 for the production 6 s / 3 s cache. Shorter
#   windows and denser strides both helped, and the stage-A ranking (w4 > w2 > w6 > w8
#   by macro-F1) contradicts the earlier "4 s > 6 s" note only in ordering, not in
#   direction -- 6 s was never the right choice.
#
#   This matters for fairness, not just for a better number. Plan item #3 existed to
#   defend against "you hobbled the EEG baseline", and it is impossible here: the
#   recordings contain no EEG channel at all (single-lead ECG only), so there is no
#   montage to expand to. Tuning is therefore the ONLY remaining answer to that
#   objection, which makes a properly-tuned baseline the strongest available response.
#
# Sequence: rebuild the 4 s / 1 s / 125 Hz cache (deleted by the sweep, KEEP_CACHE=0),
# then train the 5-class GRU on it -- 5-class being the contested cell where video's
# margin is largest. 3-class and detection can reuse the same cache afterwards; the
# cache is kept for exactly that reason.
#
# The cache build reads ~24k clip EDFs with 8 workers. The machine is idle (load ~1.6),
# so this is the right moment for it.
set -u
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
L=logs/eeg_tuned
mkdir -p "$L" output/eeg_tuned cache_abl

CACHE=cache_abl/seg_w4_s1_d8.npz
log(){ echo "[$(date +%F_%T)] $*"; }

if [ ! -f "$CACHE" ]; then
  log "building cache win=4s stride=1s decim=8 (125 Hz) -> $CACHE"
  python3 build_stage_segments_pooled.py --win 4 --stride 1 --decim 8 \
      --workers 8 --out "$CACHE" > "$L/build_w4_s1_d8.log" 2>&1
  if [ ! -f "$CACHE" ]; then
    log "CACHE BUILD FAILED -- tail:"; tail -6 "$L/build_w4_s1_d8.log"; exit 1
  fi
  log "cache built ($(du -h "$CACHE" | cut -f1))"
else
  log "reusing existing $CACHE"
fi

out=output/eeg_tuned/gru_g5_w4s1
for try in 1 2; do
  [ -f "$out/results.json" ] && break
  log "START tuned EEG 5-class (try $try)"
  python3 train_pooled_eeg.py --arch gru --cache "$CACHE" \
      --seed 42 --split_seed 49 --output "$out" > "$L/gru_g5_w4s1.log" 2>&1
  rc=$?
  [ -f "$out/results.json" ] && { log "DONE (rc=$rc)"; break; }
  log "FAILED try $try (rc=$rc); tail:"; tail -4 "$L/gru_g5_w4s1.log"; sleep 60
done

log "===== tuned vs production EEG baseline, 5-class ====="
python3 - <<'PY'
import json, os
p = 'output/eeg_tuned/gru_g5_w4s1/results.json'
if os.path.exists(p):
    r = json.load(open(p))
    print(f"  tuned      (4s/1s/125Hz): bal_acc={r['balanced_accuracy']:.4f} "
          f"macro_f1={r['macro_f1']:.4f}")
    print(f"  production (6s/3s/125Hz): bal_acc=0.4483 +/- 0.0267  macro_f1=0.3567 (n=5)")
    print(f"  sweep cell s1 measured  : bal_acc=0.4740  macro_f1=0.3672")
    print(f"  recalls: { {k: round(v['recall'], 3) for k, v in r['per_class'].items()} }")
else:
    print('  no results.json')
PY
