#!/bin/bash
# Subject-disjoint EEG cross-validation (Table 3 rows) on the REPAIRED clips.
#
# The existing Table 3 EEG rows -- 3-class 0.539 +/- 0.040, 5-class 0.274 +/- 0.049 --
# were computed on clips whose biopotential had been crushed to 2-4 unique values by the
# mne.export shared-physical-range defect. Session-disjoint EEG moved +0.13 to +0.19
# macro-F1 once the clips were repaired, so these rows will move similarly, and they
# carry the paper's "video generalises across animals better than EEG" claim.
#
# 10 folds total (3-class and 5-class x 5 folds). Folds already done or already running
# are skipped, so this composes with anything launched by hand.
#
# Concurrency counts ALL train_pooled_eeg processes, not just this script's, so it shares
# the machine correctly with the tail of the Table-1 retrain. Cap 4 (2 per GPU): six
# concurrent EEG jobs produced NVML aborts on 2026-08-11, and one Table-1 cell
# (eeg_g5_s1) had to be relaunched for exactly that reason.
set -u
cd /path/to/repo
CACHE=stage_segments_pooled_v3.npz
L=logs/v3_subject_cv; mkdir -p "$L" output/v3_subject_cv
log(){ echo "[$(date +%F_%T)] $*"; }

[ -f "$CACHE" ] || { log "ABORT: $CACHE missing"; exit 1; }

busy(){ pgrep -af "python3 train_pooled_eeg" 2>/dev/null | grep -qE -- "--output $1( |\$)"; }

fold(){
  local gpu=$1 tag=$2 flag=$3 f=$4
  local out="output/v3_subject_cv/eeg_${tag}_fold${f}"
  [ -f "$out/results.json" ] && { log "SKIP eeg_${tag}_fold${f} (done)"; return 0; }
  busy "$out" && { log "SKIP eeg_${tag}_fold${f} (already running)"; return 0; }
  local try
  for try in 1 2 3; do
    log "START eeg_${tag}_fold${f} gpu$gpu (try $try)"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_pooled_eeg.py --arch gru $flag \
        --split subject --fold "$f" --n_folds 5 --cache "$CACHE" \
        --seed 42 --split_seed 49 --output "$out" \
        > "$L/eeg_${tag}_fold${f}.log" 2>&1
    [ -f "$out/results.json" ] && { log "DONE eeg_${tag}_fold${f}"; return 0; }
    log "retry $try failed eeg_${tag}_fold${f}"; sleep 120
  done
  log "GAVE UP eeg_${tag}_fold${f}"
}

log "=== subject-disjoint EEG CV on repaired data, 4 concurrent ==="
i=0
for spec in "g3 --group3" "g5 "; do
  set -- $spec; tag=$1; shift; flag="${*:-}"
  for f in 0 1 2 3 4; do
    while [ "$(pgrep -cf 'python3 train_pooled_eeg\.py')" -ge 4 ]; do sleep 30; done
    gpu=$(( i % 2 )); i=$(( i + 1 ))
    fold "$gpu" "$tag" "$flag" "$f" &
    sleep 10
  done
done
wait

log "===== Table 3 EEG rows: corrupt vs repaired ====="
python3 - <<'PY'
import json, os, statistics as st
def cv(base, tag):
    v=[json.load(open(f'{base}/eeg_{tag}_fold{f}/results.json'))['macro_f1']
       for f in range(5) if os.path.exists(f'{base}/eeg_{tag}_fold{f}/results.json')]
    return (len(v), st.mean(v), st.stdev(v) if len(v)>1 else 0.0) if v else None
print(f"{'task':9s} {'CORRUPT (in paper)':>24s} {'REPAIRED':>24s} {'delta':>9s}")
for tag, name in (('g3','3-class'), ('g5','5-class')):
    o = cv('output/subject_cv', tag)
    n = cv('output/v3_subject_cv', tag)
    fo = f"{o[1]:.4f} +/- {o[2]:.4f} n={o[0]}" if o else "--"
    fn = f"{n[1]:.4f} +/- {n[2]:.4f} n={n[0]}" if n else "--"
    d  = f"{n[1]-o[1]:+.4f}" if (o and n) else ""
    print(f"{name:9s} {fo:>24s} {fn:>24s} {d:>9s}")
print("\nvideo subject-disjoint for reference: 3-class 0.720 +/- 0.028, 5-class 0.475 +/- 0.024")
PY
log "===== subject-CV sweep finished ====="
