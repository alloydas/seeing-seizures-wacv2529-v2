#!/bin/bash
# Plan item #8 -- EEG window / stride / pooling + sampling-rate sweep.
#
# Defends "is the EEG baseline strong enough?": the video side has a temporal ablation
# (tab:frames) and the EEG side currently has none, so "architecture diversity is not
# tuning" is an open attack. The plan also flags that a prior analysis found 4 s > 6 s
# and is worth confirming -- the pooled cache is built at 6 s.
#
# THREE AXES, run as a staged sweep rather than a full cross product (which would be
# 4 x 3 x 3 = 36 cache builds):
#
#   A. window   2 / 4 / 6 / 8 s, stride = win/2, decim 8 (125 Hz)        4 builds
#   B. stride   at the best window: win/4, win/2, win                   +2 builds
#   C. rate     at the best window: decim 4 / 8 / 16 (250/125/62.5 Hz)  +2 builds
#
#   D. aggregation (mean/max/topk/logmean/attn) is FREE -- it is applied after the
#      model runs, so sweep_eeg_agg.py scores all five from val_window_preds.npz
#      without retraining. Run it at the end over every run dir.
#
# Each stage picks its winner by clip macro-F1 before starting the next, so B and C
# are evaluated at the best window rather than at the arbitrary 6 s default.
#
# COST WARNING. A cache build reads ~24k clip EDFs with N_WORKERS processes; that is a
# heavy CPU job and video training is CPU-decode-bound. Running both at once slows the
# video queue and risks nothing crashing but everything crawling. This script therefore
# waits for the video trainers to clear before it starts, and uses --workers 8.
# Shorter windows produce more of them: 2 s / 1 s stride is ~6x the windows of the
# 6 s / 3 s cache (433 MB), so budget ~10 GB of scratch for the sweep. Caches are
# deleted after their run unless KEEP_CACHE=1.
#
# Usage:  nohup bash sh/_abl_eeg_window.sh > logs/abl_eeg_window/_driver.log 2>&1 &
#         GPU=1 KEEP_CACHE=1 nohup bash sh/_abl_eeg_window.sh > ... &
set -u
cd /path/to/repo

GPU="${GPU:-1}"
ARCH="${ARCH:-gru}"           # best EEG model; TCN is the other candidate
TASK="${TASK:-}"              # "" = 5-class, or --group3 / --group2
SEED="${SEED:-42}"
KEEP_CACHE="${KEEP_CACHE:-0}"
CACHE_DIR="${CACHE_DIR:-cache_abl}"
export OUT_BASE="${OUT_BASE:-output/abl_eeg_window}"   # set to a v3 path when
                                                # rerunning on repaired clips
L=logs/abl_eeg_window
mkdir -p "$L" "$CACHE_DIR" "$OUT_BASE"

export CUDA_VISIBLE_DEVICES="$GPU"

log(){ echo "[$(date +%F_%T)] $*"; }

# --- wait for the video queue so cache builds do not fight video decode -----------
video_jobs(){ pgrep -f "python3 train_pooled.py" 2>/dev/null | wc -l; }
log "waiting for video trainers to clear (currently $(video_jobs))"
while [ "$(video_jobs)" -gt 0 ]; do sleep 300; done
log "video queue clear -- starting sweep (arch=$ARCH task='${TASK:-5-class}' gpu=$GPU)"

# --- one (build cache -> train) cell ---------------------------------------------
# $1=tag  $2=win  $3=stride  $4=decim
cell(){
  local tag=$1 win=$2 stride=$3 decim=$4
  local cache="$CACHE_DIR/seg_w${win}_s${stride}_d${decim}.npz"
  local out="$OUT_BASE/$tag"

  if [ -f "$out/results.json" ]; then log "$tag already done -- skipping"; return 0; fi

  if [ ! -f "$cache" ]; then
    log "$tag building cache win=${win}s stride=${stride}s decim=$decim"
    python3 build_stage_segments_pooled.py --win "$win" --stride "$stride" \
        --decim "$decim" --workers 8 --out "$cache" > "$L/${tag}_build.log" 2>&1
    if [ ! -f "$cache" ]; then
      log "$tag CACHE BUILD FAILED -- tail:"; tail -5 "$L/${tag}_build.log"; return 1
    fi
  else
    log "$tag reusing cache $cache"
  fi

  local try
  for try in 1 2 3; do
    [ -f "$out/results.json" ] && break
    log "$tag START train (try $try)"
    python3 train_pooled_eeg.py --arch "$ARCH" $TASK --cache "$cache" \
        --seed "$SEED" --split_seed 49 --output "$out" > "$L/${tag}.log" 2>&1
    [ -f "$out/results.json" ] && { log "$tag DONE"; break; }
    log "$tag FAILED try $try -- tail:"; tail -3 "$L/${tag}.log"; sleep 120
  done

  [ "$KEEP_CACHE" = "1" ] || { rm -f "$cache"; log "$tag removed cache"; }
  [ -f "$out/results.json" ] || { log "$tag GAVE UP"; return 1; }
}

metric(){ python3 -c "
import json,sys
try: print('%.6f'%json.load(open(sys.argv[1]+'/results.json'))['macro_f1'])
except Exception: print('-1')
" "$OUT_BASE/$1"; }

best_of(){   # echo the tag with the highest macro-F1
  local best='' bv=-2 t v
  for t in "$@"; do v=$(metric "$t"); awk "BEGIN{exit !($v > $bv)}" && { bv=$v; best=$t; }; done
  echo "$best"
}

# ================= A. window length ==============================================
log "=== stage A: window length ==="
cell w2 2  1   8
cell w4 4  2   8
cell w6 6  3   8
cell w8 8  4   8
BEST=$(best_of w2 w4 w6 w8)
log "stage A winner: $BEST (macro-F1 $(metric "$BEST"))"
BW=${BEST#w}                      # winning window in seconds

# ================= B. stride at the best window ==================================
log "=== stage B: stride at win=${BW}s ==="
S_QUARTER=$(python3 -c "print(f'{$BW/4:g}')")
S_FULL=$(python3 -c "print(f'{$BW:g}')")
cell "s${S_QUARTER}" "$BW" "$S_QUARTER" 8      # dense
cell "s${S_FULL}"    "$BW" "$S_FULL"    8      # non-overlapping
# the win/2 point is stage A's winner, already computed
log "stage B done"

# ================= C. sampling rate at the best window ===========================
log "=== stage C: sampling rate at win=${BW}s ==="
S_HALF=$(python3 -c "print(f'{$BW/2:g}')")
cell r250  "$BW" "$S_HALF" 4     # 250 Hz
cell r62   "$BW" "$S_HALF" 16    # 62.5 Hz
# 125 Hz is stage A's winner
log "stage C done"

# ================= D. aggregation (free, no retraining) ==========================
log "=== stage D: clip aggregation ==="
python3 sweep_eeg_agg.py --glob "$OUT_BASE/*" \
    --json_out $OUT_BASE/agg_sweep.json 2>&1 | tee "$L/agg_sweep.log"

log "=== summary ==="
python3 - <<'PY'
import json, os, glob
rows = []
for d in sorted(glob.glob(os.environ.get('OUT_BASE','output/abl_eeg_window')+'/*')):
    p = os.path.join(d, 'results.json')
    if os.path.isfile(p):
        r = json.load(open(p))
        rows.append((os.path.basename(d), r['balanced_accuracy'], r['macro_f1'], r['accuracy']))
print(f"{'cell':10s} {'bal_acc':>9s} {'macro_f1':>9s} {'acc':>8s}")
for n, b, m, a in rows:
    print(f"{n:10s} {b:9.4f} {m:9.4f} {a:8.4f}")
PY
log "===== EEG window/stride/rate/pooling sweep complete ====="
