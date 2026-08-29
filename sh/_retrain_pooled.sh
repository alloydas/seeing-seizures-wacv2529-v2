#!/bin/bash
# Retrain both pooled models on the complete cropped set (post straggler-fill).
# Sequential: single GPU, avoid contention. Writes to *_v2 so the old runs are kept.
cd /path/to/repo
echo "==== POOLED RETRAIN START $(date +%T) ===="
echo "---- 3-class severity ----"
python3 train_pooled.py --group3 --epochs 12 --output classifier_pooled3_v2
echo "---- 2-class detection ----"
python3 train_pooled.py --group2 --epochs 8  --output classifier_pooled2_v2
echo "==== POOLED RETRAIN COMPLETE $(date +%T) ===="
