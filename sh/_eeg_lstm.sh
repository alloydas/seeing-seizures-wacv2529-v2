#!/bin/bash
cd /path/to/repo
export CUDA_VISIBLE_DEVICES=0
L=logs/eeg_extra
run(){ echo "[$(date +%F_%T)] START $1"; shift; "$@"; echo "[$(date +%F_%T)] END rc=$?"; }
run lstm_bin python3 train_pooled_eeg.py --arch lstm --group2 --output output/eeg_lstm_bin > $L/lstm_bin.log 2>&1
run lstm_g3  python3 train_pooled_eeg.py --arch lstm --group3 --output output/eeg_lstm_g3  > $L/lstm_g3.log  2>&1
run lstm_g5  python3 train_pooled_eeg.py --arch lstm          --output output/eeg_lstm_g5  > $L/lstm_g5.log  2>&1
echo "[$(date +%F_%T)] LSTM done"
