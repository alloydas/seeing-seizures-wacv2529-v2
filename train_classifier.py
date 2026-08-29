"""
Seizure vs Non-Seizure Video Classifier
=======================================
Trains an R(2+1)D-18 (Kinetics-400 pretrained) on the paired
seizure_*/  and  clip_*_vs_seizure_*/  clips produced by
cut_seizure_clips.py + cut_non_seizure_clips.py.

Defaults expect data at /path/to/repo/Data/<session>/<clip_dir>/video.mp4
and write checkpoints + history to /path/to/repo/classifier_out/.

Smoke test (1 epoch, small subset):
    python train_classifier.py --epochs 1 --max_train_sessions 2 --epoch_subsample 32

Full run:
    python train_classifier.py --epochs 15 --batch_size 16
"""

import argparse, os, random, time, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm


KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/path/to/repo/Data")
    p.add_argument("--output",    type=str, default="/path/to/repo/classifier_out")
    p.add_argument("--frames",    type=int, default=16)
    p.add_argument("--size",      type=int, default=112)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs",    type=int, default=15)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--val_frac",  type=float, default=0.2)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="cuda:0")
    p.add_argument("--max_train_sessions", type=int, default=None,
                   help="Cap number of training sessions (smoke-test).")
    p.add_argument("--epoch_subsample", type=int, default=None,
                   help="Cap clips per epoch (smoke-test).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Data discovery + per-session split
# ─────────────────────────────────────────────────────────────

def discover_clips(data_root):
    """Return list of (video_path, label, session_name).
    Positive: <session>/seizure_NN_<label>_<ts>/video.mp4
    Negative: <session>/clip_NN_vs_seizure_NN_<label>_<ts>/video.mp4
    """
    items = []
    root = Path(data_root)
    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub in session.iterdir():
            if not sub.is_dir():
                continue
            mp4 = sub / "video.mp4"
            if not mp4.exists():
                continue
            name = sub.name
            if name.startswith("seizure_"):
                items.append((str(mp4), 1, session.name))
            elif name.startswith("clip_"):
                items.append((str(mp4), 0, session.name))
    return items


def split_by_session(items, val_frac, seed):
    sessions = sorted({s for _, _, s in items})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val = max(1, int(round(len(sessions) * val_frac)))
    val_set = set(sessions[:n_val])
    train = [it for it in items if it[2] not in val_set]
    val   = [it for it in items if it[2] in val_set]
    return train, val, val_set


# ─────────────────────────────────────────────────────────────
#  Frame loading
# ─────────────────────────────────────────────────────────────

def sample_frame_indices(n_total, n_sample):
    if n_total <= 0:
        return None
    if n_total >= n_sample:
        return np.linspace(0, n_total - 1, n_sample).round().astype(int)
    idx = np.arange(n_total)
    pad = np.full(n_sample - n_total, n_total - 1, dtype=int)
    return np.concatenate([idx, pad])


def load_clip(path, n_frames, size, raw=False):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(n_total, n_frames)
    if indices is None:
        cap.release()
        return None
    target = set(int(x) for x in indices)
    grabbed = {}
    # Sequential grab() avoids costly seeks on copy-mode mp4s with sparse keyframes.
    i = 0
    while i < n_total and len(grabbed) < len(target):
        if not cap.grab():
            break
        if i in target:
            ok, frame = cap.retrieve()
            if ok:
                grabbed[i] = frame
        i += 1
    cap.release()
    if not grabbed:
        return None
    keys_sorted = sorted(grabbed.keys())
    out = []
    for idx in indices:
        f = grabbed.get(int(idx))
        if f is None:
            nearest = min(keys_sorted, key=lambda k: abs(k - int(idx)))
            f = grabbed[nearest]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        out.append(f)
    arr = np.stack(out, axis=0)                               # (T, H, W, C) uint8
    if raw:
        # Callers that normalise on the GPU take the frames unscaled: the uint8 array is
        # 4x smaller across a DataLoader worker->main handoff. Default stays float32 so
        # every existing caller is unchanged.
        return arr.transpose(3, 0, 1, 2)                       # (C, T, H, W) uint8
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - KINETICS_MEAN) / KINETICS_STD
    arr = arr.transpose(3, 0, 1, 2)                           # (C, T, H, W)
    return arr


class ClipDataset(Dataset):
    def __init__(self, items, n_frames, size, augment=False):
        self.items = items
        self.n_frames = n_frames
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        arr = load_clip(path, self.n_frames, self.size)
        if arr is None:
            arr = np.zeros((3, self.n_frames, self.size, self.size), dtype=np.float32)
        if self.augment and random.random() < 0.5:
            arr = arr[:, :, :, ::-1].copy()
        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
#  Model + training loop
# ─────────────────────────────────────────────────────────────

def build_model(num_classes=2):
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, desc):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_labels = [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            prob_pos = torch.softmax(logits.float(), dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
            all_probs.append(prob_pos.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/max(total,1):.3f}")
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")
    return loss_sum / max(total, 1), correct / max(total, 1), auc, probs, labels


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    items = discover_clips(args.data_root)
    n_pos = sum(1 for _, l, _ in items if l == 1)
    n_neg = sum(1 for _, l, _ in items if l == 0)
    n_sessions = len({s for _, _, s in items})
    print(f"Discovered {len(items)} clips across {n_sessions} sessions  "
          f"(pos={n_pos}, neg={n_neg})")
    if not items:
        raise SystemExit(f"No clips found under {args.data_root}")

    train_items, val_items, val_sessions = split_by_session(items, args.val_frac, args.seed)
    print(f"Train: {len(train_items):>5} clips / "
          f"{len({s for _,_,s in train_items}):>3} sessions")
    print(f"Val:   {len(val_items):>5} clips / {len(val_sessions):>3} sessions")

    if args.max_train_sessions:
        train_sess = sorted({s for _,_,s in train_items})[:args.max_train_sessions]
        train_items = [it for it in train_items if it[2] in train_sess]
        print(f"  smoke cap: {len(train_items)} train clips / {len(train_sess)} sessions")
    if args.epoch_subsample:
        train_items = random.sample(train_items, min(args.epoch_subsample, len(train_items)))
        val_items   = random.sample(val_items,   min(args.epoch_subsample, len(val_items)))
        print(f"  smoke subsample: train={len(train_items)} val={len(val_items)}")

    train_ds = ClipDataset(train_items, args.frames, args.size, augment=True)
    val_ds   = ClipDataset(val_items,   args.frames, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    model = build_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    history, best_auc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            train=True, desc=f"ep{epoch} train")
        scheduler.step()
        val_loss, val_acc, val_auc, vp, vl = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler,
            train=False, desc=f"ep{epoch} val  ")
        cm = confusion_matrix(vl, (vp >= 0.5).astype(int), labels=[0, 1]).tolist()
        rec = dict(epoch=epoch,
                   train_loss=tr_loss, train_acc=tr_acc, train_auc=tr_auc,
                   val_loss=val_loss, val_acc=val_acc, val_auc=val_auc,
                   val_cm=cm, secs=round(time.time() - t0, 1))
        history.append(rec)
        print(f"[ep{epoch}] tr loss={tr_loss:.4f} acc={tr_acc:.3f} auc={tr_auc:.3f}  "
              f"|  val loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f}  "
              f"|  cm={cm}  |  {rec['secs']}s")
        with open(os.path.join(args.output, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_auc": val_auc, "val_acc": val_acc,
                "args": vars(args),
                "val_sessions": sorted(val_sessions),
            }, os.path.join(args.output, "best.pt"))
            print(f"  saved best.pt  (val_auc={val_auc:.3f})")

    torch.save({
        "epoch": args.epochs, "model": model.state_dict(),
        "args": vars(args),
        "val_sessions": sorted(val_sessions),
    }, os.path.join(args.output, "last.pt"))
    print(f"Done. Best val AUROC = {best_auc:.3f}. Outputs in {args.output}")


if __name__ == "__main__":
    main()
