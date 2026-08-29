#!/bin/bash
# Gap-filler for the Table 2 seed sweep.
#
# sh/_tab2_seeds.sh runs each cell exactly ONCE -- its deep() has no retry loop, unlike
# every other driver in this project. That gap showed up at 15:34 when lstm_g3_s2 died
# with the NVML assert (memory pressure aborts hard because the driver/library mismatch
# leaves NVML broken), and nothing re-queued it, so that cell would silently finish the
# sweep at n=4 while the caption claims 5 seeds.
#
# This waits for the main sweep to exit, then re-runs every missing (arch, task, seed)
# with 3 attempts and LOW concurrency (2 at a time). Six concurrent EEG jobs is evidently
# at the edge of what GPU memory tolerates here -- which is also why raising the main
# sweep's concurrency, as I had suggested, would have made failures more likely, not the
# wall-clock shorter.
set -u
cd /path/to/repo
L=logs/tab2; mkdir -p "$L"
log(){ echo "[$(date +%F_%T)] [gapfill] $*"; }
SEEDS="1 2 3 5 42"

busy(){ pgrep -af "python3 train_pooled" 2>/dev/null | grep -qE "^[0-9]+ python3? .*--output $1( |$)"; }

log "waiting for the main sweep to finish"
while pgrep -f "[_]tab2_seeds.sh" > /dev/null; do sleep 300; done
log "main sweep done -- scanning for gaps"

missing=0
for arch in lstm eegnet conformer tcn rf xgb; do
  for spec in "bin --group2" "g3 --group3" "g5 "; do
    set -- $spec; tag=$1; shift; flag="${*:-}"
    for seed in $SEEDS; do
      out="output/tab2/${arch}_${tag}_s${seed}"
      [ -f "$out/results.json" ] && continue
      busy "$out" && continue
      missing=$((missing+1))
      log "GAP ${arch}_${tag}_s${seed} -- re-running"
      for try in 1 2 3; do
        [ -f "$out/results.json" ] && break
        while [ "$(pgrep -cf 'python3 train_pooled_eeg')" -ge 2 ]; do sleep 60; done
        if [ "$arch" = "rf" ] || [ "$arch" = "xgb" ]; then
          python3 train_pooled_eeg_classical.py --arch "$arch" $flag \
              --seed "$seed" --split_seed 49 --output "$out" \
              > "$L/${arch}_${tag}_s${seed}.log" 2>&1
        else
          CUDA_VISIBLE_DEVICES=$(( missing % 2 )) python3 train_pooled_eeg.py --arch "$arch" $flag \
              --seed "$seed" --split_seed 49 --output "$out" \
              > "$L/${arch}_${tag}_s${seed}.log" 2>&1
        fi
        [ -f "$out/results.json" ] && { log "RECOVERED ${arch}_${tag}_s${seed}"; break; }
        log "retry $try failed for ${arch}_${tag}_s${seed}"; sleep 120
      done
      [ -f "$out/results.json" ] || log "GAVE UP ${arch}_${tag}_s${seed}"
    done
  done
done
log "gap scan complete ($missing gaps found)"
python3 make_tab2.py --check
