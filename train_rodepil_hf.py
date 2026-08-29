"""
TimeSformer / VideoMAE on RodEpil, subject-wise 5-fold CV (identical protocol,
discovery and folds as train_rodepil.py), so the HF transformer backbones join
the RodEpil comparison table. TimeSformer especially reproduces / compares against
RodEpil's own baseline under our pipeline.

  python train_rodepil_hf.py --arch timesformer --all_folds --root RodEpil_std --output output/rodepil_timesformer
  python train_rodepil_hf.py --arch videomae    --all_folds --root RodEpil_std --output output/rodepil_videomae
"""
import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import train_rodepil as TR
from train_pooled_timesformer import load_clip_ts
from train_pooled_videomae import load_clip_vmae

MODELS = {
    "timesformer": ("facebook/timesformer-base-finetuned-k400", 8, 224, load_clip_ts),
    "videomae":    ("MCG-NJU/videomae-base-finetuned-kinetics", 16, 224, load_clip_vmae),
}


class ClipsHF(Dataset):
    def __init__(s, items, loader, frames, size): s.items, s.loader, s.frames, s.size = items, loader, frames, size
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y, _ = s.items[i]
        a = s.loader(p, s.frames, s.size)
        if a is None:
            a = np.zeros((s.frames, 3, s.size, s.size), np.float32)
        return torch.from_numpy(a), y, i


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); Y, P, I = [], [], []
    for x, y, idx in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(pixel_values=x.to(dev)).logits, 1)
        Y.append(y.numpy()); P.append(p.float().cpu().numpy()); I.append(idx.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(I)


def run_fold(items, test_subs, a, dev, tag, model_id, frames, size, loader):
    from transformers import TimesformerForVideoClassification, VideoMAEForVideoClassification
    test_subs = set(test_subs)
    tr = [it for it in items if it[2] not in test_subs]
    te = [it for it in items if it[2] in test_subs]
    ctr = Counter(y for _, y, _ in tr)
    print(f"\n=== {tag} === test {sorted(test_subs)}  train {len(tr)} / test {len(te)}", flush=True)
    dtr = DataLoader(ClipsHF(tr, loader, frames, size), batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    dte = DataLoader(ClipsHF(te, loader, frames, size), batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=True)
    Cls = TimesformerForVideoClassification if a.arch == "timesformer" else VideoMAEForVideoClassification
    model = Cls.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True).to(dev)
    freq = np.array([max(ctr[i], 1) for i in range(2)], float); w = 1.0 / freq; w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0; best_res = None
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
        Y, P, I = evaluate(model, dte, dev); res, _ = TR.report(Y, P)
        print(f"ep{ep:>2} loss={tot/max(n,1):.4f} seizF1={res['seizure_f1']:.4f} "
              f"macroF1={res['macro_f1']:.4f} AUROC={res['auroc']:.4f}", flush=True)
        if res["macro_f1"] > best:
            best = res["macro_f1"]; best_res = res
    return best_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["timesformer", "videomae"], required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all_folds", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=TR.ROOT)
    ap.add_argument("--output", default="output/rodepil_hf_out")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    Path(a.output).mkdir(parents=True, exist_ok=True)
    model_id, frames, size, loader = MODELS[a.arch]
    print(f"arch={a.arch} frames={frames} size={size}", flush=True)

    items = TR.discover(a.root); subs = sorted({s for _, _, s in items})
    print(f"RodEpil: {len(items)} clips, {len(subs)} subjects, {a.folds}-fold subject-wise", flush=True)
    folds = TR.subject_folds(subs, a.folds, a.seed)
    which = range(a.folds) if a.all_folds else [a.fold]
    allres = {}
    for fi in which:
        res = run_fold(items, folds[fi], a, dev, f"fold{fi}", model_id, frames, size, loader)
        allres[f"fold{fi}"] = res
        print(f"[fold{fi}] BEST macroF1={res['macro_f1']:.4f} seizF1={res['seizure_f1']:.4f} "
              f"AUROC={res['auroc']:.4f}", flush=True)
    if len(allres) > 1:
        keys = ["seizure_f1", "macro_f1", "auroc", "balanced_accuracy", "accuracy"]
        fr = list(allres.values())
        allres["mean"] = {k: float(np.nanmean([r[k] for r in fr])) for k in keys}
        allres["std"] = {k: float(np.nanstd([r[k] for r in fr], ddof=1)) for k in keys}
        print("=== CV MEAN ===", {k: round(v, 4) for k, v in allres["mean"].items()})
    (Path(a.output) / "results.json").write_text(json.dumps(allres, indent=2))
    print(f"wrote {a.output}/results.json")


if __name__ == "__main__":
    main()
