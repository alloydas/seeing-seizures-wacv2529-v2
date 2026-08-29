"""
Reproduce the RodEpil paper's backbone (TimeSformer) and train it on OUR pooled
10-subject rodent cohort, using the IDENTICAL data discovery + session-disjoint
split (seed 49, 2-class detection) as train_pooled.py. This gives a clean
architecture head-to-head on the same data:
    TimeSformer (this script)   vs   R(2+1)D-18 (classifier_pooled2_v3, macro-F1 0.958).

Model: facebook/timesformer-base-finetuned-k400 (8 frames @ 224, ImageNet/K400
norm), classifier head replaced with 2 classes.

  python train_pooled_timesformer.py --epochs 8 --batch_size 8 --output timesformer_pooled2
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import cv2
import frame_cache, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             roc_auc_score, confusion_matrix)
from transformers import TimesformerForVideoClassification
sys.path.insert(0, "/path/to/repo")
import train_pooled as TP
from train_classifier import sample_frame_indices

MEAN = np.array([0.45, 0.45, 0.45], np.float32)
STD  = np.array([0.225, 0.225, 0.225], np.float32)
NAMES = ["non-seizure", "seizure"]


def load_clip_ts(path, n_frames=8, size=224):
    """Return (T,C,H,W) float32 normalized, or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sample_frame_indices(n_total, n_frames)
    if idx is None:
        cap.release(); return None
    want = set(int(x) for x in idx); got = {}
    i = 0
    while i < n_total and len(got) < len(want):
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                got[i] = fr
        i += 1
    cap.release()
    if not got:
        return None
    ks = sorted(got)
    out = []
    for j in idx:
        f = got.get(int(j))
        if f is None:
            f = got[min(ks, key=lambda k: abs(k - int(j)))]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        out.append(f)
    a = np.stack(out).astype(np.float32) / 255.0          # (T,H,W,C)
    a = (a - MEAN) / STD
    return a.transpose(0, 3, 1, 2)                         # (T,C,H,W)


class Clips(Dataset):
    def __init__(s, items, frames, size, cache_dir=None):
        s.items, s.frames, s.size = items, frames, size
        s.cache = frame_cache.get(cache_dir, frames, size)
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i][0], s.items[i][1]
        a = None
        if s.cache is not None:
            raw = s.cache.raw(p)              # uint8 (T,S,S,3), pre-normalisation
            if raw is not None:
                a = np.asarray(raw, np.float32) / 255.0
                a = ((a - MEAN) / STD).transpose(0, 3, 1, 2)
        if a is None:
            a = load_clip_ts(p, s.frames, s.size)
        if a is None:
            a = np.zeros((s.frames, 3, s.size, s.size), np.float32)
        return torch.from_numpy(a), y, i


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); Y, P, I = [], [], []
    for x, y, idx in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(pixel_values=x.to(dev)).logits
            p = torch.softmax(logits, 1)
        Y.append(y.numpy()); P.append(p.float().cpu().numpy()); I.append(idx.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cache_dir", default=None,
        help="pre-decoded frame cache (build_frame_cache.py)")
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group3", action="store_true")
    ap.add_argument("--group2", action="store_true")
    ap.add_argument("--output", default="timesformer_pooled2")
    a = ap.parse_args()

    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    # same discovery + seed-49 session-disjoint split as train_pooled.py
    raw = TP.discover()
    if a.group2:
        items = [(p, TP.GROUP2[y], s, sub) for p, y, s, sub in raw]; TP.NAMES, TP.NC = TP.NAMES2, 2
    elif a.group3:
        items = [(p, TP.GROUP3[y], s, sub) for p, y, s, sub in raw]; TP.NAMES, TP.NC = TP.NAMES3, 3
    else:
        items = raw; TP.NAMES, TP.NC = TP.NAMES5, 5
    print(f"*** TimeSformer {TP.NC}-class ***")
    tr, va, valsess, seed = TP.split_sessions(items, a.split_seed)
    ctr, cva = Counter(i[1] for i in tr), Counter(i[1] for i in va)
    print(f"TimeSformer on OUR cohort | {len(items)} clips / {len({i[2] for i in items})} sessions "
          f"/ {len({i[3] for i in items})} subjects", flush=True)
    print(f"train {len(tr)} (ns {ctr[0]}, s {ctr[1]}) / val {len(va)} (ns {cva[0]}, s {cva[1]})  "
          f"split_seed={seed}", flush=True)

    dtr = DataLoader(Clips(tr, a.frames, a.size, a.cache_dir), batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    dva = DataLoader(Clips(va, a.frames, a.size, a.cache_dir), batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=True)

    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", num_labels=TP.NC,
        ignore_mismatched_sizes=True).to(dev)
    freq = np.array([max(ctr[i], 1) for i in range(TP.NC)], float)
    w = 1.0 / freq; w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for x, y, _ in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(pixel_values=x).logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        Y, P, I = evaluate(model, dva, dev)
        pred = P.argmax(1)
        mf1 = f1_score(Y, pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(Y, pred); acc = accuracy_score(Y, pred)
        auroc = roc_auc_score(Y, P[:, 1]) if TP.NC == 2 and len(set(Y.tolist())) > 1 else float("nan")
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_acc=float(acc),
                         val_bal_acc=float(bacc), val_macro_f1=float(mf1), val_auroc=float(auroc)))
        print(f"ep{ep:>2} loss={tot/max(n,1):.4f} val_acc={acc:.4f} bal_acc={bacc:.4f} "
              f"macroF1={mf1:.4f} AUROC={auroc:.4f}", flush=True)
        if mf1 > best:
            best = mf1
            torch.save({"model": model.state_dict(), "epoch": ep, "macro_f1": mf1}, out / "best.pt")
            np.savez(out / "val_preds.npz", y=Y, probs=P, pred=pred, idx=I)
            print(f"   saved best (macroF1={mf1:.4f})", flush=True)
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    z = np.load(out / "val_preds.npz")
    Y, P, pred, I = z["y"], z["probs"], z["pred"], z["idx"]
    subs = np.array([va[i][3] for i in I])
    res = TP.report(Y, pred, P, f"TimeSformer POOLED ({len({i[3] for i in items})} subjects) -- held-out val")
    res["per_subject"] = TP.per_subject(Y, pred, subs)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/results.json  (best macroF1={best:.4f})")


if __name__ == "__main__":
    main()
