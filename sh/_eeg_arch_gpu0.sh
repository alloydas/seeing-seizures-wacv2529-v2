#!/bin/bash
# GPU 0: EEGNet (bin/g3/g5) then TCN (bin/g3/g5) on the 20-subject cache.
# Same split/seed as the GRU runs -> directly comparable.
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
L=logs/eeg_arch
mkdir -p "$L"
run(){ echo "[$(date +%F_%T)] START $1"; shift; "$@"; echo "[$(date +%F_%T)] END rc=$?"; }

for A in eegnet tcn; do
  run "$A-bin" python3 train_pooled_eeg.py --arch $A --group2 --output eeg_${A}_bin > "$L/${A}_bin.log" 2>&1
  run "$A-g3"  python3 train_pooled_eeg.py --arch $A --group3 --output eeg_${A}_g3  > "$L/${A}_g3.log"  2>&1
  run "$A-g5"  python3 train_pooled_eeg.py --arch $A          --output eeg_${A}_g5  > "$L/${A}_g5.log"  2>&1
done
echo "[$(date +%F_%T)] GPU0 eeg-arch driver done"
