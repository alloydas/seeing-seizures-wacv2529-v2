#!/bin/bash
# Waits for the 105-run backbone sweep (sh/_v3_bestcfg_backbones.sh) and emits the
# tab:eegarch rows. The sweep itself waits on the EEG ablation, so this is the tail of
# a three-link chain: ablation -> pick best config -> 105 runs -> table.
set -u
cd /path/to/repo
log(){ echo "[$(date +%F_%T)] $*"; }
log "waiting for sh/_v3_bestcfg_backbones.sh"
while pgrep -f "_v3_bestcfg_backbones2\.sh" > /dev/null; do sleep 300; done
log "backbone sweep finished"
log "winning configuration recorded by the sweep:"
grep -m1 "winner:" logs/v3_bestcfg/_driver.log 2>/dev/null || echo "  (not found in driver log)"
python3 make_tab_eegarch.py 2>&1
