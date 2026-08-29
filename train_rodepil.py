"""
Train/test the R(2+1)D-18 video detector on the RodEpil rodent seizure dataset
(Perlo et al. 2025, Zenodo 10.5281/zenodo.17601357) using the SAME subject-wise
protocol as the paper -- no subject appears in both train and test -- so our
number is comparable to their TimeSformer baseline (F1 97%, subject-wise 5-fold).

Layout:  RodEpil_zenodo_17601357/Seizure_detection_10sec/{0,1}/*.avi
         folder 0 = normal (label 0), folder 1 = seizure (label 1)
         subject (RatID) = 4th '_'-separated token from the end of the filename.

Reuses build_model + load_clip from train_classifier.py (16 frames @ 112).

  # single subject-wise fold (fast)
  python train_rodepil.py --fold 0 --epochs 8 --output rodepil_out
  # full 5-fold subject-wise CV
  python train_rodepil.py --all_folds --epochs 8 --output rodepil_out
"""
import argparse, glob, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             accuracy_score, roc_auc_score, precision_recall_fscore_support)
sys.path.insert(0, "/path/to/repo")
from train_classifier import build_model, load_clip
from train_pooled import build_video_model

ARCH_FS = {"r2plus1d": (16, 112), "swin": (32, 224), "mvit": (16, 224),
           "swin_s": (32, 224), "mvit_v1": (16, 224), "s3d": (16, 224),
           "x3d": (16, 224), "slowfast": (32, 224)}

ROOT = "RodEpil_zenodo_17601357/Seizure_detection_10sec"
NAMES = ["normal", "seizure"]


def subject_of(fname):
    # filename: Date_VideoID_Position_RatID_frames_Label_Y.(avi|mp4); RatID = 4th token from end
    return os.path.splitext(os.path.basename(fname))[0].split("_")[-4]


def discover(root):
    items = []
    for lab in (0, 1):
        for f in glob.glob(f"{root}/{lab}/*.avi") + glob.glob(f"{root}/{lab}/*.mp4"):
            items.append((f, lab, subject_of(f)))
    return items


def subject_folds(subjects, k, seed):
    subs = sorted(subjects)
    random.Random(seed).shuffle(subs)
    return [subs[i::k] for i in range(k)]     # k round-robin groups


class Clips(Dataset):
    def __init__(s, items, frames, size): s.items, s.frames, s.size = items, frames, size
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y, _ = s.items[i]
        a = load_clip(p, s.frames, s.size)
        if a is None:
            a = np.zeros((3, s.frames, s.size, s.size), np.float32)
        return torch.from_numpy(a), y, i


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); Y, P, I = [], [], []
    for x, y, idx in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(x.to(dev)), 1)
        Y.append(y.numpy()); P.append(p.float().cpu().numpy()); I.append(idx.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(I)


def report(Y, P):
    pred = P.argmax(1)
    out = dict(
        n=int(len(Y)),
        accuracy=float(accuracy_score(Y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(Y, pred)),
        seizure_f1=float(f1_score(Y, pred, pos_label=1, zero_division=0)),
        macro_f1=float(f1_score(Y, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(Y, pred, average="weighted", zero_division=0)),
        auroc=float(roc_auc_score(Y, P[:, 1])) if len(set(Y.tolist())) > 1 else float("nan"),
        confusion=confusion_matrix(Y, pred).tolist(),
    )
    return out, pred


def run_fold(items, test_subs, a, dev, tag):
    test_subs = set(test_subs)
    tr = [it for it in items if it[2] not in test_subs]
    te = [it for it in items if it[2] in test_subs]
    ctr = Counter(y for _, y, _ in tr)
    print(f"\n=== {tag} ===")
    print(f"test subjects: {sorted(test_subs)}")
    print(f"train {len(tr)} (norm {ctr[0]}, seiz {ctr[1]}) / test {len(te)}", flush=True)

    dtr = DataLoader(Clips(tr, a.frames, a.size), batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    dte = DataLoader(Clips(te, a.frames, a.size), batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=True)

    model, _, _ = build_video_model(a.arch, 2); model = model.to(dev)
    freq = np.array([max(ctr[i], 1) for i in range(2)], float)
    w = 1.0 / freq; w = w / w.mean()
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
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        Y, P, I = evaluate(model, dte, dev)
        res, _ = report(Y, P)
        print(f"ep{ep:>2} loss={tot/max(n,1):.4f}  seizF1={res['seizure_f1']:.4f} "
              f"macroF1={res['macro_f1']:.4f}  AUROC={res['auroc']:.4f} "
              f"balAcc={res['balanced_accuracy']:.4f}", flush=True)
        if res["macro_f1"] > best:
            best = res["macro_f1"]; best_res = res
            # per-subject on best
            subs = np.array([te[i][2] for i in I]); pred = P.argmax(1)
            per = {}
            for su in sorted(set(subs)):
                m = subs == su
                per[su] = dict(n=int(m.sum()),
                               seizure_f1=float(f1_score(Y[m], pred[m], pos_label=1, zero_division=0)),
                               macro_f1=float(f1_score(Y[m], pred[m], average="macro", zero_division=0)),
                               support=[int((Y[m] == 0).sum()), int((Y[m] == 1).sum())])
            best_res["per_subject"] = per
            torch.save({"model": model.state_dict(), "epoch": ep, "res": {k: v for k, v in best_res.items() if k != "per_subject"}},
                       Path(a.output) / f"best_{tag}.pt")
    return best_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="r2plus1d", choices=list(ARCH_FS.keys()))
    ap.add_argument("--frames", type=int, default=0, help="0 = arch default")
    ap.add_argument("--size", type=int, default=0, help="0 = arch default")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0, help="which fold to run (single)")
    ap.add_argument("--all_folds", action="store_true", help="run all K folds and average")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=ROOT, help="dataset root with 0/ and 1/ subdirs")
    ap.add_argument("--output", default="rodepil_out")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(a.output).mkdir(parents=True, exist_ok=True)

    df, ds = ARCH_FS[a.arch]
    if a.frames == 0: a.frames = df
    if a.size == 0: a.size = ds
    print(f"arch={a.arch}  frames={a.frames}  size={a.size}", flush=True)

    items = discover(a.root)
    subs = sorted({s for _, _, s in items})
    byc = Counter(y for _, y, _ in items)
    print(f"RodEpil: {len(items)} clips (normal {byc[0]}, seizure {byc[1]}), "
          f"{len(subs)} subjects, {a.folds}-fold subject-wise", flush=True)
    folds = subject_folds(subs, a.folds, a.seed)

    which = range(a.folds) if a.all_folds else [a.fold]
    allres = {}
    for fi in which:
        res = run_fold(items, folds[fi], a, dev, f"fold{fi}")
        allres[f"fold{fi}"] = res
        print(f"[fold{fi}] BEST  seizF1={res['seizure_f1']:.4f}  macroF1={res['macro_f1']:.4f}  "
              f"AUROC={res['auroc']:.4f}", flush=True)

    if len(allres) > 1:
        keys = ["seizure_f1", "macro_f1", "auroc", "balanced_accuracy", "accuracy"]
        fold_res = list(allres.values())
        allres["mean"] = {k: float(np.nanmean([r[k] for r in fold_res])) for k in keys}
        allres["std"] = {k: float(np.nanstd([r[k] for r in fold_res], ddof=1)) for k in keys}
        print("\n=== CV MEAN ===", {k: round(v, 4) for k, v in allres["mean"].items()})
        print("=== CV STD  ===", {k: round(v, 4) for k, v in allres["std"].items()})
    (Path(a.output) / "results.json").write_text(json.dumps(allres, indent=2))
    print(f"\nwrote {a.output}/results.json")


if __name__ == "__main__":
    main()
