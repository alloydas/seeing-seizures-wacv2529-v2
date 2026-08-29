"""
Pooled multi-subject seizure severity classifier on EEG (GRU), all 10 subjects.

EEG counterpart to train_pooled.py. Loads stage_segments_pooled.npz (windows from
every subject's clip EDFs), uses the SAME session-disjoint seed-search split logic
as train_pooled.py so the pooled EEG and pooled video numbers are comparable.

Trains at window level, evaluates at clip level (mean-pool). --group2 / --group3 /
default 5-class. Always prints a per-subject breakdown.
"""
import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             balanced_accuracy_score, roc_auc_score, accuracy_score)

NAMES5 = ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"]
NAMES3 = ["non-seizure", "mild (S2-3)", "severe (S4-5)"]
NAMES2 = ["non-seizure", "seizure"]
GROUP3 = np.array([0, 1, 1, 2, 2]); GROUP2 = np.array([0, 1, 1, 1, 1])
NAMES, NC = NAMES5, 5


class GRUModel(nn.Module):
    def __init__(s, i=1, h=128, o=5, n=2, d=0.3):
        super().__init__(); s.gru = nn.GRU(i, h, n, batch_first=True, dropout=d); s.fc = nn.Linear(h, o)
    def forward(s, x):
        _, hn = s.gru(x); return s.fc(hn[-1])


class LSTMModel(nn.Module):
    def __init__(s, i=1, h=128, o=5, n=2, d=0.3):
        super().__init__(); s.lstm = nn.LSTM(i, h, n, batch_first=True, dropout=d); s.fc = nn.Linear(h, o)
    def forward(s, x):
        _, (hn, _) = s.lstm(x); return s.fc(hn[-1])


# ---- EEGNet (Lawhern et al. 2018), single-channel (C=1) adaptation ----------
# Input to every model is (B, T, 1): one EEG channel, T=750 samples @125 Hz.
class EEGNet(nn.Module):
    def __init__(s, nc=5, C=1, F1=8, D=2, F2=16, kern=64, p=0.3):
        super().__init__()
        s.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1))
        s.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p))
        s.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        s.head = nn.LazyLinear(nc)

    def forward(s, x):                       # (B, T, 1)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C=1, T)
        x = s.separable(s.depthwise(s.firstconv(x)))
        return s.head(x.flatten(1))


# ---- EEG Conformer (Song et al. 2022): conv patch-embed + transformer -------
class EEGConformer(nn.Module):
    def __init__(s, nc=5, C=1, k=40, tk=25, pool=15, depth=4, heads=4, p=0.3):
        super().__init__()
        s.embed = nn.Sequential(
            nn.Conv2d(1, k, (1, tk), padding=(0, tk // 2)),
            nn.Conv2d(k, k, (C, 1)),
            nn.BatchNorm2d(k), nn.ELU(),
            nn.AvgPool2d((1, pool), (1, pool // 2)),   # temporal tokens
            nn.Dropout(p))
        enc = nn.TransformerEncoderLayer(k, heads, k * 4, dropout=p,
                                         activation="gelu", batch_first=True)
        s.transformer = nn.TransformerEncoder(enc, depth)
        s.head = nn.Sequential(nn.LayerNorm(k), nn.Linear(k, nc))

    def forward(s, x):                       # (B, T, 1)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, T)
        x = s.embed(x).squeeze(2).permute(0, 2, 1)   # (B, Ttok, k)
        return s.head(s.transformer(x).mean(1))


# ---- Temporal Convolutional Network (Bai et al. 2018) -----------------------
class _Chomp(nn.Module):
    def __init__(s, c): super().__init__(); s.c = c
    def forward(s, x): return x[:, :, :-s.c].contiguous() if s.c > 0 else x


class _TCNBlock(nn.Module):
    def __init__(s, ci, co, k, dil, p=0.2):
        super().__init__(); pad = (k - 1) * dil
        s.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=dil), _Chomp(pad), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(co, co, k, padding=pad, dilation=dil), _Chomp(pad), nn.ReLU(), nn.Dropout(p))
        s.down = nn.Conv1d(ci, co, 1) if ci != co else None
        s.relu = nn.ReLU()

    def forward(s, x):
        o = s.net(x); r = x if s.down is None else s.down(x); return s.relu(o + r)


class TCN(nn.Module):
    def __init__(s, nc=5, ci=1, ch=(64, 64, 64, 64, 64), k=7, p=0.2):
        super().__init__()
        layers, prev = [], ci
        for i, c in enumerate(ch):
            layers.append(_TCNBlock(prev, c, k, 2 ** i, p)); prev = c
        s.tcn = nn.Sequential(*layers); s.head = nn.Linear(prev, nc)

    def forward(s, x):                # (B, T, 1)
        x = s.tcn(x.permute(0, 2, 1)) # (B, C, T)
        return s.head(x.mean(-1))


def build_model(arch, nc, hidden):
    if arch == "gru":       return GRUModel(1, hidden, nc)
    if arch == "lstm":      return LSTMModel(1, hidden, nc)
    if arch == "eegnet":    return EEGNet(nc)
    if arch == "conformer": return EEGConformer(nc)
    if arch == "tcn":       return TCN(nc)
    raise SystemExit(f"unknown --arch {arch}")


def split_sessions(sess_of_clip, lab_of_clip, seed, val_frac=0.2):
    """Session-disjoint; search seeds from `seed` so every class lands in both halves.
    Mirrors train_pooled.split_sessions (which starts at seed 49)."""
    sessions = sorted(set(sess_of_clip))
    for s in range(seed, seed + 500):
        rng = random.Random(s); ss = sessions[:]; rng.shuffle(ss)
        val = set(ss[:max(1, round(len(ss) * val_frac))])
        vlab = [l for l, se in zip(lab_of_clip, sess_of_clip) if se in val]
        tlab = [l for l, se in zip(lab_of_clip, sess_of_clip) if se not in val]
        cv, ct = Counter(vlab), Counter(tlab)
        if all(ct[c] > 0 for c in range(NC)) and all(cv[c] > 0 for c in range(NC)):
            return val, s
    raise SystemExit("no split with all classes in both halves")


def split_subjects(csess, seed, fold=0, n_folds=5):
    """Subject-disjoint k-fold for EEG: hold out whole subjects (RodEpil-style).
    Returns the set of val *sessions* (those of held-out subjects) so the caller's
    existing per-clip `s in val_sess` logic is unchanged, plus the held-out subject
    set for logging."""
    subs = sorted(set(s.split("/")[0] for s in csess))
    rng = random.Random(seed); rng.shuffle(subs)
    groups = [subs[k::n_folds] for k in range(n_folds)]
    valsub = set(groups[fold % n_folds])
    val_sess = set(s for s in csess if s.split("/")[0] in valsub)
    return val_sess, valsub


def report(y, pred, prob, tag):
    print(f"\n=== {tag} ===")
    print(f"  accuracy          {accuracy_score(y, pred):.4f}")
    print(f"  balanced accuracy {balanced_accuracy_score(y, pred):.4f}")
    print(f"  macro F1          {f1_score(y, pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted F1       {f1_score(y, pred, average='weighted', zero_division=0):.4f}")
    try:
        if NC == 2:
            print(f"  AUROC             {roc_auc_score(y, prob[:, 1]):.4f}")
        else:
            pres = sorted(set(y.tolist())); p = prob[:, pres]; p = p / p.sum(1, keepdims=True)
            print(f"  macro AUROC (ovr) {roc_auc_score(y, p, multi_class='ovr', average='macro', labels=pres):.4f}")
    except Exception as e:
        print(f"  AUROC             n/a ({e})")
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(NC), zero_division=0)
    print(f"\n  {'class':<14} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for i in range(NC):
        print(f"  {NAMES[i]:<14} {pr[i]:>6.3f} {rc[i]:>7.3f} {f1[i]:>6.3f} {sup[i]:>8d}")
    cm = confusion_matrix(y, pred, labels=range(NC))
    print("\n  confusion (rows=true):")
    print(f"  {'':<14}" + "".join(f"{n[:8]:>10}" for n in NAMES))
    for i in range(NC):
        print(f"  {NAMES[i]:<14}" + "".join(f"{cm[i][j]:>10d}" for j in range(NC)))
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                per_class={NAMES[i]: dict(precision=float(pr[i]), recall=float(rc[i]),
                                          f1=float(f1[i]), support=int(sup[i])) for i in range(NC)},
                confusion=cm.tolist())


def per_subject(y, pred, subs):
    print("\n=== PER-SUBJECT (clip level) ===")
    print(f"  {'subj':<8}{'n':>5}{'acc':>8}{'macroF1':>9}   support")
    out = {}
    for s in sorted(set(subs)):
        m = subs == s
        ys, ps = y[m], pred[m]
        acc = accuracy_score(ys, ps); mf1 = f1_score(ys, ps, average="macro", zero_division=0)
        sup = np.bincount(ys, minlength=NC).tolist()
        print(f"  {s:<8}{int(m.sum()):>5}{acc:>8.3f}{mf1:>9.3f}   {sup}")
        out[s] = dict(n=int(m.sum()), accuracy=float(acc), macro_f1=float(mf1), support=sup)
    return out


def aggregate_clip(P, cid_va, val_clips, how="mean", topk_frac=0.25, temp=1.0):
    """Pool window-level probabilities P into one probability vector per clip.

    Plan item #8 asks for clip-aggregation alternatives to the mean pool. All options
    here are parameter-free and applied at evaluation time only, so they change no
    training dynamics and can be re-swept post hoc from val_window_preds.npz.

      mean    - average over the clip's windows (original behaviour, the default)
      max     - per-class maximum; a single confident window carries the clip
      topk    - mean over the most confident topk_frac of windows (>=1)
      logmean - geometric mean (mean of log-probs); penalises any low-probability window
      attn    - confidence-weighted average, weights = softmax(max-prob / temp) over
                windows. NOTE: this is a *parameter-free* stand-in, not a learned
                attention pool -- a learned one needs clip-level training and is a
                separate change.
    """
    out = []
    for c in val_clips:
        p = P[cid_va == c]
        if len(p) == 0:
            out.append(np.zeros(P.shape[1], P.dtype)); continue
        if how == "mean":
            q = p.mean(0)
        elif how == "max":
            q = p.max(0)
        elif how == "topk":
            k = max(1, int(round(topk_frac * len(p))))
            idx = np.argsort(p.max(1))[-k:]
            q = p[idx].mean(0)
        elif how == "logmean":
            q = np.exp(np.log(np.clip(p, 1e-9, 1.0)).mean(0))
        elif how == "attn":
            w = p.max(1) / max(temp, 1e-6)
            w = np.exp(w - w.max()); w = w / w.sum()
            q = (p * w[:, None]).sum(0)
        else:
            raise SystemExit(f"unknown --agg {how}")
        s = q.sum()
        out.append(q / s if s > 0 else q)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--arch", default="gru", choices=["gru", "lstm", "eegnet", "conformer", "tcn"])
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["session", "subject"], default="session")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--group3", action="store_true")
    ap.add_argument("--group2", action="store_true")
    ap.add_argument("--cache", default="stage_segments_pooled.npz")
    ap.add_argument("--subject", default=None, help="keep only this subject (e.g. RN197)")
    ap.add_argument("--agg", default="mean",
                    choices=["mean", "max", "topk", "logmean", "attn"],
                    help="clip-level pooling of window probabilities (plan item #8)")
    ap.add_argument("--topk_frac", type=float, default=0.25)
    ap.add_argument("--agg_temp", type=float, default=1.0)
    ap.add_argument("--output", default="eeg_pooled_out")
    a = ap.parse_args()

    global NAMES, NC
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.cache, allow_pickle=True)
    segs, wlab, cid, clab, csess = d["segs"], d["wlab"], d["clip_id"], d["clip_lab"], d["clip_sess"]
    # clip_path is present only in caches built after 2026-08-10; older caches omit it
    cpath = d["clip_path"] if "clip_path" in d.files else None
    if a.subject:
        keep = [i for i, s in enumerate(csess) if s.split("/")[0] == a.subject]
        remap = {old: new for new, old in enumerate(keep)}
        wm = np.array([c in remap for c in cid])
        segs, wlab = segs[wm], wlab[wm]
        cid = np.array([remap[c] for c in cid[wm]])
        clab, csess = clab[keep], csess[keep]
        print(f"*** filtered to {a.subject}: {len(keep)} clips, {len(segs)} windows ***")
    if a.group2:
        wlab, clab = GROUP2[wlab], GROUP2[clab]; NAMES, NC = NAMES2, 2; print("*** pooled EEG 2-class ***")
    elif a.group3:
        wlab, clab = GROUP3[wlab], GROUP3[clab]; NAMES, NC = NAMES3, 3; print("*** pooled EEG 3-class ***")
    else:
        print("*** pooled EEG 5-class ***")

    m = segs.mean(1, keepdims=True); sd = segs.std(1, keepdims=True); sd[sd == 0] = 1.0
    segs = ((segs - m) / sd).astype(np.float32)

    # csess / clab are already per-clip (indexed by clip id); cid is per-window
    sess_of_clip = list(csess)
    lab_of_clip = [int(x) for x in clab]
    if a.split == "subject":
        val_sess, valsub = split_subjects(csess, a.split_seed, a.fold, a.n_folds)
        used_seed = a.split_seed
        print(f"*** SUBJECT-DISJOINT EEG: fold {a.fold}/{a.n_folds}, held-out={sorted(valsub)} ***")
    else:
        val_sess, used_seed = split_sessions(sess_of_clip, lab_of_clip, a.split_seed)
    clip_is_val = np.array([s in val_sess for s in csess])  # per-clip, indexed by clip id
    win_is_val = clip_is_val[cid]                            # per-window

    Xtr, ytr = segs[~win_is_val], wlab[~win_is_val]
    print(f"subjects: {len(set(s.split('/')[0] for s in csess))}  clips: train={int((~clip_is_val).sum())} "
          f"val={int(clip_is_val.sum())}  val_sessions={len(val_sess)}  seed={used_seed}")
    print(f"windows: train={len(Xtr)} val={int(win_is_val.sum())}")
    print(f"  train window labels: {np.bincount(ytr, minlength=NC).tolist()}")

    dtr = DataLoader(TensorDataset(torch.from_numpy(Xtr).unsqueeze(-1), torch.from_numpy(ytr)),
                     batch_size=a.batch_size, shuffle=True, drop_last=True)
    dva = DataLoader(TensorDataset(torch.from_numpy(segs[win_is_val]).unsqueeze(-1),
                                   torch.from_numpy(wlab[win_is_val])), batch_size=512, shuffle=False)

    model = build_model(a.arch, NC, a.hidden).to(dev)
    with torch.no_grad():                       # materialize any LazyLinear before optimizer
        model(torch.zeros(2, segs.shape[1], 1, device=dev))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"arch: {a.arch}  params={n_par/1e3:.1f}k", flush=True)
    freq = np.array([max(int((ytr == i).sum()), 1) for i in range(NC)], float)
    w = 1.0 / freq; w = w / w.mean()
    print("class weights: " + "  ".join(f"{NAMES[i]}={w[i]:.2f}" for i in range(NC)), flush=True)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    cid_va = cid[win_is_val]
    val_clips = np.unique(cid_va)
    cy = np.array([int(clab[c]) for c in val_clips])
    csub = np.array([csess[c].split("/")[0] for c in val_clips])
    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for x, y in dtr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        model.eval(); P = []
        with torch.no_grad():
            for x, _ in dva:
                P.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
        P = np.concatenate(P)
        cp = aggregate_clip(P, cid_va, val_clips, a.agg, a.topk_frac, a.agg_temp)
        cpred = cp.argmax(1)
        mf1 = f1_score(cy, cpred, average="macro", zero_division=0)
        acc = accuracy_score(cy, cpred)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), clip_acc=float(acc), clip_macro_f1=float(mf1)))
        print(f"ep{ep:>2}  loss={tot/max(n,1):.4f}  clip_acc={acc:.4f}  clip_macroF1={mf1:.4f}", flush=True)
        if mf1 > best:
            best = mf1
            torch.save({"model": model.state_dict(), "epoch": ep, "macro_f1": mf1}, out / "best.pt")
            _extra = {} if cpath is None else {"path": np.asarray([cpath[c] for c in val_clips])}
            np.savez(out / "val_clip_preds.npz", y=cy, probs=cp, pred=cpred, sub=csub,
                     **_extra)
            # window-level probs let the aggregation axis of #8 be re-swept
            # without retraining (see sweep_eeg_agg.py)
            np.savez(out / "val_window_preds.npz", probs=P, cid=cid_va,
                     clips=np.asarray(val_clips), y=cy, sub=csub)
            print(f"   saved best.pt (macro_f1={mf1:.4f})", flush=True)
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    z = np.load(out / "val_clip_preds.npz", allow_pickle=True)
    res = report(z["y"], z["pred"], z["probs"], f"POOLED EEG ({len(set(csub))} subjects), clip level")
    res["per_subject"] = per_subject(z["y"], z["pred"], z["sub"])
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
