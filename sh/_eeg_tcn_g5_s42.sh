#!/bin/bash
# Re-run the STRONGEST 5-class competitor (TCN) at seed 42 under current code.
#
# Why this specific job: every row of tab:eegarch was written 2026-07-22..07-30, i.e.
# all before the 2026-08-03 refactor of train_pooled_eeg.py. The GRU row is already
# known stale -- stored 0.3598, current code gives 0.448 +/- 0.027 over 5 seeds.
# That alone reorders the 5-class table:
#     stored:  TCN .4622 > conformer .4261 ~ XGB .4262 > EEGNet .3991 > GRU .3598 > LSTM .3595 > RF .2785
#     GRU under current code (.448) would move from 5th to 2nd, within noise of TCN.
# So "TCN wins at 5-class" may be an artifact of comparing post-refactor GRU numbers
# against pre-refactor everything-else. This run tests the top of the table:
#   - TCN comes back ~0.46  -> GRU .448 +/- .027 is statistically tied, ranking claim unsupported
#   - TCN also shifts up    -> the whole table needs regenerating under current code
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=1
L=logs/seed_runs
name=eeg_tcn_g5_s42

for try in 1 2 3; do
  [ -f "output/seed_runs/$name/results.json" ] && break
  echo "[$(date +%F_%T)] START $name (try $try)"
  python3 train_pooled_eeg.py --arch tcn --seed 42 --split_seed 49 \
      --output "output/seed_runs/$name" > "$L/$name.log" 2>&1
  rc=$?
  if [ -f "output/seed_runs/$name/results.json" ]; then
    echo "[$(date +%F_%T)] DONE $name (rc=$rc)"; exit 0
  fi
  echo "[$(date +%F_%T)] FAILED $name try $try (rc=$rc); tail:"
  tail -3 "$L/$name.log"
  sleep 120
done
echo "[$(date +%F_%T)] GAVE UP on $name"
