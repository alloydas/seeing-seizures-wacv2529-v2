#!/usr/bin/env python3
"""tab:vidarch with mean +/- std, pooling the seed-42 runs with the seeds added by
sh/_v3_vidseeds.sh. Reports n per cell so partial progress is legible."""
import glob, json, os, statistics as st
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

ROWS = [("MViTv2-S","mvit"), ("SlowFast-R50","slowfast"), ("R(2+1)D-18","r2plus1d"),
        ("X3D-M","x3d"), ("MViTv1-B","mvit_v1"), ("Video Swin-S","swin_s"),
        ("S3D","s3d"), ("Video Swin-T","swin"), ("VideoMAE-B","videomae"),
        ("TimeSformer","tsf")]
SEED42 = {  # where the existing single-seed runs live (three locations, historically)
    ("r2plus1d","bin"):"output/classifier_pooled20_bin", ("r2plus1d","g3"):"output/classifier_pooled20_g3",
    ("r2plus1d","g5"):"output/classifier_pooled20_g5",
    ("tsf","bin"):"output/timesformer_pooled20_bin", ("tsf","g3"):"vid_tsf_g3", ("tsf","g5"):"vid_tsf_g5",
}
def m(d):
    f=os.path.join(d,"val_preds.npz")
    if not os.path.exists(f):
        # runs predating the val_preds patch (e.g. TimeSformer detection) keep only
        # results.json; F1 is recoverable, the rest is not
        r=os.path.join(d,"results.json")
        if os.path.exists(r):
            j=json.load(open(r))
            return dict(P=float("nan"),R=float("nan"),F1=j["macro_f1"],
                        AUC=float("nan"),MCC=float("nan"))
        return None
    z=np.load(f,allow_pickle=True); y,p,P=z["y"],z["pred"],z["probs"]
    auc=roc_auc_score(y,P[:,1]) if P.shape[1]==2 else roc_auc_score(y,P,multi_class="ovr",average="macro")
    return dict(P=precision_score(y,p,average="macro",zero_division=0),
                R=recall_score(y,p,average="macro",zero_division=0),
                F1=f1_score(y,p,average="macro",zero_division=0),
                AUC=auc, MCC=matthews_corrcoef(y,p))
def cells(arch,tag):
    out=[]
    for cand in (SEED42.get((arch,tag)), f"output/vid_{arch}_{tag}", f"vid_{arch}_{tag}"):
        if cand and (r:=m(cand)): out.append(r); break
    for d in sorted(glob.glob(f"output/v3_vidseeds/{arch}_{tag}_s*")):
        # results.json is the completion marker -- sh/_v3_vidseeds.sh writes it last and
        # treats its absence as "rerun this cell". An interrupted run still leaves
        # val_preds.npz from its best epoch so far, so keying on that alone silently
        # folded 1-epoch fragments into the mean (VideoMAE g3 went 0.737 n=4 -> 0.705 n=5).
        if not os.path.exists(os.path.join(d,"results.json")): continue
        if (r:=m(d)): out.append(r)
    return out
print(f"{'backbone':14s} " + " ".join(f"{t:>22s}" for t in ("detection","3-class","5-class")))
for name,arch in ROWS:
    row=f"{name:14s}"
    for tag in ("bin","g3","g5"):
        c=cells(arch,tag)
        if c:
            f=[x["F1"] for x in c]
            row+=f"  {st.mean(f):.3f}±{(st.stdev(f) if len(f)>1 else 0):.3f} n={len(f)}".rjust(23)
        else: row+=f"{'--':>23s}"
    print(row)
