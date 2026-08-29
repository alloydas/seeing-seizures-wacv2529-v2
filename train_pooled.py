"""
Pooled multi-subject seizure severity classifier (video).

Trains on ALL animal-cropped subjects at once. Each root is cropped to its OWN cage
(geometry verified per camera in _crop_all.sh / crop_verify/), so pooling is valid.

Modes:
  default    5-class: non-seizure / Stage2 / Stage3 / Stage4 / Stage5
  --group3   3-class: non-seizure / mild (S2-3) / severe (S4-5)

Splits are SESSION-disjoint. Because seizure burden is wildly uneven across animals
(RN197 + RN213 are ~45% of all seizure clips, RN216 has 19), a single pooled score is
misleading -- so we ALWAYS print a per-subject breakdown as well.

Usage:
  python train_pooled.py --group3 --epochs 12 --output classifier_pooled3
"""
import argparse, glob, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             balanced_accuracy_score, accuracy_score, roc_auc_score)
sys.path.insert(0, "/path/to/repo")
from train_classifier import build_model, load_clip, KINETICS_MEAN, KINETICS_STD


def build_video_model(arch, nc, pretrained=True):
    """Return (model, frames, size). All torchvision video nets take (B,C,T,H,W)
    -- the same layout load_clip() produces -- so only the head, frame count and
    spatial size differ per backbone. Frame/size are the values each net was
    pretrained at on Kinetics-400. pretrained=False -> random init (for the
    pretraining ablation); currently wired for r2plus1d."""
    if arch == "r2plus1d":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m, 16, 112
    if arch == "swin":
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        m = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        m.head = nn.Linear(m.head.in_features, nc)
        return m, 32, 224
    if arch == "mvit":
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        m.head[-1] = nn.Linear(m.head[-1].in_features, nc)
        return m, 16, 224
    if arch == "swin_s":
        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        m = swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1)
        m.head = nn.Linear(m.head.in_features, nc)
        return m, 32, 224
    if arch == "mvit_v1":
        from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
        m = mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1)
        m.head[-1] = nn.Linear(m.head[-1].in_features, nc)
        return m, 16, 224
    if arch == "s3d":
        from torchvision.models.video import s3d, S3D_Weights
        m = s3d(weights=S3D_Weights.KINETICS400_V1)
        m.classifier[1] = nn.Conv3d(1024, nc, kernel_size=1, stride=1)
        return m, 16, 224
    if arch == "x3d":                        # pytorchvideo, single-pathway
        from pytorchvideo.models.hub import x3d_m
        m = x3d_m(pretrained=True)
        m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, nc)
        return m, 16, 224
    if arch == "slowfast":                   # pytorchvideo, dual-pathway (wrapped)
        from pytorchvideo.models.hub import slowfast_r50
        net = slowfast_r50(pretrained=True)
        net.blocks[-1].proj = nn.Linear(net.blocks[-1].proj.in_features, nc)
        return _SlowFastWrap(net), 32, 224
    raise SystemExit(f"unknown --arch {arch}")


class _SlowFastWrap(nn.Module):
    """Feed a single (B,C,T,H,W) clip to SlowFast: the fast pathway is the full
    clip, the slow pathway is temporally subsampled by `alpha`. Keeps the
    existing single-tensor Clips dataset usable for SlowFast."""
    def __init__(s, net, alpha=4):
        super().__init__(); s.net = net; s.alpha = alpha
    def forward(s, x):
        idx = torch.linspace(0, x.shape[2] - 1, x.shape[2] // s.alpha).long().to(x.device)
        return s.net([x.index_select(2, idx), x])

# every cropped root, keyed by subject
ROOTS = {
    # RoomC
    "RN197": "data/Data_RN197_cropped",
    "RN199": "data/Data_RN199_cropped",
    "RN204": "data/Data_RN204_cropped",
    "RN213": "data/Data_RN213_cropped",
    "RN216": "data/Data_RN216_cropped",
    "RN222": "data/Data_RN222_cropped",
    "RN235": "data/Data_RN235_cropped",
    "RN237": "data/Data_RN237_cropped",
    "RN238": "data/Data_RN238_cropped",
    "RN245": "data/Data_RN245_cropped",
    # RoomD (added 2026-07-22; each cropped to its own cage)
    "RN208": "data/Data_RN208_cropped",
    "RN210": "data/Data_RN210_cropped",
    "RN215": "data/Data_RN215_cropped",
    "RN219": "data/Data_RN219_cropped",
    "RN223": "data/Data_RN223_cropped",
    "RN224": "data/Data_RN224_cropped",
    "RN227": "data/Data_RN227_cropped",
    "RN229": "data/Data_RN229_cropped",
    "RN242": "data/Data_RN242_cropped",
    "RN244": "data/Data_RN244_cropped",
}
STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}
NAMES5 = ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"]
NAMES3 = ["non-seizure", "mild (S2-3)", "severe (S4-5)"]
GROUP3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
GROUP2 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
NAMES2 = ["non-seizure", "seizure"]
NAMES, NC = NAMES5, 5


def discover():
    items = []
    for subj, root in ROOTS.items():
        if not os.path.isdir(root):
            print(f"  [warn] missing root {root} (subject {subj}) -- skipped")
            continue
        for d in glob.glob(f"{root}/*/seizure_*") + glob.glob(f"{root}/*/clip_*_vs_seizure_*"):
            mp4 = os.path.join(d, "video.mp4")
            if not os.path.exists(mp4):
                continue
            b, sess = os.path.basename(d), os.path.basename(os.path.dirname(d))
            if b.startswith("seizure_"):
                m = re.search(r"Stage_[0-9]+", b)
                if not m or m.group() not in STAGE:
                    continue
                y = STAGE[m.group()]
            else:
                y = 0
            items.append((mp4, y, f"{subj}/{sess}", subj))
    return items


def split_sessions(items, seed, val_frac=0.2, need_all=True):
    """Session-disjoint. Search seeds so every class lands in both halves."""
    sessions = sorted({i[2] for i in items})
    for s in range(seed, seed + 500):
        rng = random.Random(s); ss = sessions[:]; rng.shuffle(ss)
        val = set(ss[:max(1, round(len(ss) * val_frac))])
        tr = [i for i in items if i[2] not in val]
        va = [i for i in items if i[2] in val]
        ctr, cva = Counter(i[1] for i in tr), Counter(i[1] for i in va)
        if not need_all or (all(ctr[c] > 0 for c in range(NC)) and all(cva[c] > 0 for c in range(NC))):
            return tr, va, val, s
    raise SystemExit("no split found with all classes in both halves")


def split_subjects(items, seed, fold=0, n_folds=5):
    """Subject-disjoint k-fold: whole subjects are held out for validation, so
    the model is tested on animals it never saw in training (the strict RodEpil-
    style protocol). Round-robin partition on a seeded subject shuffle; the
    `fold`-th group is validation. Rare stages may be absent from some folds --
    that is inherent to subject-wise evaluation, so we do NOT enforce full class
    coverage here (unlike session-disjoint)."""
    subs = sorted({i[3] for i in items})
    rng = random.Random(seed); rng.shuffle(subs)
    groups = [subs[k::n_folds] for k in range(n_folds)]
    val = set(groups[fold % n_folds])
    tr = [i for i in items if i[3] not in val]
    va = [i for i in items if i[3] in val]
    return tr, va, val, seed


class FrameCache:
    """Pre-decoded frames for one (frames, size) geometry; see build_frame_cache.py.

    Holds the post-resize RGB uint8 frames, i.e. the `out` list inside load_clip()
    before scaling and normalisation, so reconstruction here is bit-identical to
    decoding. fetch() returns those frames still uint8; norm_batch() scales and
    normalises on the GPU. Falls back to load_clip(raw=True) for any clip absent
    from the index.
    """
    _open = {}

    def __init__(s, d, frames, size):
        import json as _json
        meta = _json.load(open(os.path.join(d, "index.json")))
        assert meta["frames"] == frames and meta["size"] == size, \
            f"cache {d} is {meta['frames']}x{meta['size']}, need {frames}x{size}"
        s.row = {p: i for i, p in enumerate(meta["paths"])}
        s.bad = set(meta.get("unreadable", []))
        s.shape = (meta["n"], frames, size, size, 3)
        s.path = os.path.join(d, "frames.u8")
        s.mm = None

    @classmethod
    def get(cls, d, frames, size):
        if d not in cls._open:
            cls._open[d] = FrameCache(d, frames, size)
        return cls._open[d]

    def fetch(s, p):
        i = s.row.get(p)
        if i is None or p in s.bad:
            return None
        if s.mm is None:                       # opened lazily, per DataLoader worker
            s.mm = np.memmap(s.path, dtype=np.uint8, mode="r", shape=s.shape)
        # uint8 (T,H,W,C) -> (C,T,H,W) view. Scaling and Kinetics normalisation moved
        # to norm_batch() on the GPU, so what crosses the worker->main shared-memory
        # boundary is 4.8 MB per 32x224 sample instead of 19.3 MB. Same arithmetic in
        # the same order, so the normalised tensor is bit-identical either way.
        return s.mm[i].transpose(3, 0, 1, 2)


FILL_U8 = np.round(KINETICS_MEAN * 255).astype(np.uint8)

_NORM = {}


def norm_batch(x, dev):
    """uint8 (B,C,T,H,W) on `dev` -> Kinetics-normalised float32.

    Same three float32 ops in the same order as the old worker-side path: divide by
    255, subtract mean, divide by std. 255 is held as a 0-dim *device* tensor, not a
    Python scalar -- ATen's CUDA div_ takes a reciprocal-multiply fast path for cpu
    scalars (a * (1/255)), which disagrees with numpy true division on 126 of the 256
    uint8 levels by 1 ULP (5.96e-08 pre-norm, 7.15e-07 after /std). Going through a
    device tensor keeps the exact division kernel, so the result stays bit-identical
    to the pre-refactor worker-side numpy path.
    """
    ms = _NORM.get(dev)
    if ms is None:
        ms = (torch.as_tensor(KINETICS_MEAN, device=dev).view(1, 3, 1, 1, 1),
              torch.as_tensor(KINETICS_STD, device=dev).view(1, 3, 1, 1, 1),
              torch.as_tensor(255.0, device=dev))
        _NORM[dev] = ms
    return x.float().div_(ms[2]).sub_(ms[0]).div_(ms[1])


class Clips(Dataset):
    def __init__(s, items, frames, size, cache_dir=None):
        s.items, s.frames, s.size = items, frames, size
        s.cache = FrameCache.get(cache_dir, frames, size) if cache_dir else None
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i][0], s.items[i][1]
        a = s.cache.fetch(p) if s.cache else None
        if a is None:
            a = load_clip(p, s.frames, s.size, raw=True)
        if a is None:
            # Unreadable clip. The pre-GPU-normalisation code filled with np.zeros in
            # *normalised* space, i.e. exactly the Kinetics mean; the nearest uint8 is
            # round(mean*255) = (110, 101, 96), which normalises to
            # (-3.45e-3, +6.38e-3, +9.49e-5) rather than exactly 0. This is the one
            # place the refactor is not bit-identical, and it
            # is dead code for the current caches -- f32s224 reports unreadable=0 and
            # all 24497 pooled items resolve in the index.
            a = np.empty((3, s.frames, s.size, s.size), np.uint8)
            a[:] = FILL_U8[:, None, None, None]
        return torch.from_numpy(np.ascontiguousarray(a)), y, i


def evaluate(model, dl, dev):
    model.eval(); P, Y, I = [], [], []
    with torch.no_grad():
        for x, y, i in dl:
            x = norm_batch(x.to(dev, non_blocking=True), dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x)
            P.append(torch.softmax(lg.float(), 1).cpu().numpy()); Y.append(y.numpy()); I.append(i.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(I)


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
            pres = sorted(set(y.tolist()))
            p = prob[:, pres]; p = p / p.sum(1, keepdims=True)
            print(f"  macro AUROC (ovr) {roc_auc_score(y, p, multi_class='ovr', average='macro', labels=pres):.4f}")
    except Exception as e:
        print(f"  macro AUROC       n/a ({e})")
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(NC), zero_division=0)
    print(f"\n  {'class':<14} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for i in range(NC):
        print(f"  {NAMES[i]:<14} {pr[i]:>6.3f} {rc[i]:>7.3f} {f1[i]:>6.3f} {sup[i]:>8d}")
    cm = confusion_matrix(y, pred, labels=range(NC))
    print(f"\n  confusion (rows=true):")
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
    print(f"\n=== PER-SUBJECT breakdown (held-out val) ===")
    print(f"  {'subject':<8} {'n':>5} {'acc':>7} {'bal_acc':>8} {'macroF1':>8}   class support")
    out = {}
    for s in sorted(set(subs)):
        m = subs == s
        if m.sum() == 0: continue
        ys, ps = y[m], pred[m]
        sup = np.bincount(ys, minlength=NC).tolist()
        acc = accuracy_score(ys, ps)
        bacc = balanced_accuracy_score(ys, ps) if len(set(ys.tolist())) > 1 else float("nan")
        mf1 = f1_score(ys, ps, average="macro", zero_division=0)
        print(f"  {s:<8} {int(m.sum()):>5} {acc:>7.3f} {bacc:>8.3f} {mf1:>8.3f}   {sup}")
        out[s] = dict(n=int(m.sum()), accuracy=float(acc), macro_f1=float(mf1), support=sup)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frames", type=int, default=0, help="0 = use arch default")
    ap.add_argument("--size", type=int, default=0, help="0 = use arch default")
    ap.add_argument("--arch", default="r2plus1d",
                    choices=["r2plus1d", "swin", "mvit", "swin_s", "mvit_v1", "s3d",
                             "x3d", "slowfast"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cache_dir", default=None,
                    help="pre-decoded frame cache from build_frame_cache.py")
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--seed", type=int, default=42, help="init/data-order seed (for seed-variance runs)")
    ap.add_argument("--deterministic", action="store_true",
                    help="force bit-reproducible training (see the note by the seeding "
                         "block). Costs throughput; NOT the default, because every "
                         "published number in this repo was produced without it.")
    ap.add_argument("--split", choices=["session", "subject"], default="session",
                    help="session-disjoint (default) or subject-disjoint k-fold eval")
    ap.add_argument("--fold", type=int, default=0, help="subject-split: which fold is val")
    ap.add_argument("--n_folds", type=int, default=5, help="subject-split: number of folds")
    ap.add_argument("--group3", action="store_true")
    ap.add_argument("--group2", action="store_true")
    ap.add_argument("--output", default="classifier_pooled")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="data-efficiency: keep this fraction of TRAIN subjects (seeded)")
    ap.add_argument("--scratch", action="store_true",
                    help="pretraining ablation: random init instead of Kinetics weights")
    a = ap.parse_args()

    global NAMES, NC
    # Seeding alone does NOT make training reproducible on this box. Measured
    # 2026-08-26: re-running mvit_g5_s5 at its original --seed 5 gave macro-F1 0.6532
    # against the recorded 0.6825, with the two runs already differing in epoch 1
    # (train_loss 0.843649 vs 0.839866) -- before any accumulation could explain it.
    # The same retrain of x3d_g3_s2 reproduced EXACTLY (+0.0000) and slowfast_bin_s1 to
    # +0.0005, so it is not the data, the cache or the seed: it is MViT. torchvision's
    # mvit_v2_s carries 49 Dropout/StochasticDepth modules and 16 attention blocks, and
    # the nondeterministic reduction kernels those use perturb gradients enough to
    # diverge within a single epoch.
    #
    # --deterministic pins that down. It is opt-in rather than default because every
    # number already published from this repo was produced WITHOUT it; flipping the
    # default would silently make new runs incomparable with the existing tables.
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by cuBLAS
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)     # warn: some attention
        print("*** deterministic mode: reproducible, slower ***", flush=True)  # kernels lack
    # No else-branch: the non-deterministic path deliberately leaves cudnn.benchmark at
    # its PyTorch default (False). Turning it on here would change the behaviour of every
    # existing run configuration, which is the opposite of an opt-in flag.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    items = discover()
    if a.group2:
        items = [(p, GROUP2[y], s, sub) for p, y, s, sub in items]
        NAMES, NC = NAMES2, 2
        print("*** 2-class detection mode ***")
    elif a.group3:
        items = [(p, GROUP3[y], s, sub) for p, y, s, sub in items]
        NAMES, NC = NAMES3, 3
        print("*** 3-class severity mode ***")
    if a.split == "subject":
        tr, va, valsess, seed = split_subjects(items, a.split_seed, a.fold, a.n_folds)
        print(f"*** SUBJECT-DISJOINT split: fold {a.fold}/{a.n_folds}, "
              f"held-out subjects = {sorted(valsess)} ***")
    else:
        tr, va, valsess, seed = split_sessions(items, a.split_seed)

    # data-efficiency ablation: keep only a seeded fraction of TRAIN subjects
    if a.frac < 1.0:
        tsubs = sorted({i[3] for i in tr})
        rng = random.Random(1000 + a.split_seed); rng.shuffle(tsubs)
        keep = set(tsubs[:max(1, round(len(tsubs) * a.frac))])
        tr = [i for i in tr if i[3] in keep]
        print(f"*** DATA-FRACTION {a.frac}: kept {len(keep)}/{len(tsubs)} train subjects, "
              f"{len(tr)} train clips ***")

    ctr, cva = Counter(i[1] for i in tr), Counter(i[1] for i in va)

    bysub = Counter(i[3] for i in items)
    print(f"\npooled {len(items)} clips / {len({i[2] for i in items})} sessions / {len(bysub)} subjects")
    print("  per subject: " + "  ".join(f"{k}={v}" for k, v in sorted(bysub.items())))
    print(f"train {len(tr)} / val {len(va)}  ({len(valsess)} val groups, split={a.split}, seed={seed})")
    print(f"  {'class':<14} {'train':>7} {'val':>6}")
    for i in range(NC):
        print(f"  {NAMES[i]:<14} {ctr[i]:>7d} {cva[i]:>6d}")

    ARCH_FS = {"r2plus1d": (16, 112), "swin": (32, 224), "mvit": (16, 224),
               "swin_s": (32, 224), "mvit_v1": (16, 224), "s3d": (16, 224),
               "x3d": (16, 224), "slowfast": (32, 224)}
    df, ds = ARCH_FS[a.arch]
    if a.frames == 0: a.frames = df
    if a.size == 0: a.size = ds
    print(f"arch={a.arch}  frames={a.frames}  size={a.size}", flush=True)

    dtr = DataLoader(Clips(tr, a.frames, a.size, a.cache_dir), batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    dva = DataLoader(Clips(va, a.frames, a.size, a.cache_dir), batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=True)

    model, _, _ = build_video_model(a.arch, NC, pretrained=not a.scratch)
    if a.scratch: print("*** SCRATCH: random init (no Kinetics pretraining) ***", flush=True)
    model = model.to(dev)
    freq = np.array([max(ctr[i], 1) for i in range(NC)], float)
    w = 1.0 / freq; w = w / w.mean()
    print("class weights: " + "  ".join(f"{NAMES[i]}={w[i]:.2f}" for i in range(NC)), flush=True)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for x, y, _ in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            x = norm_batch(x, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        Y, P, I = evaluate(model, dva, dev)
        pred = P.argmax(1)
        mf1 = f1_score(Y, pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(Y, pred); acc = accuracy_score(Y, pred)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_acc=float(acc),
                         val_bal_acc=float(bacc), val_macro_f1=float(mf1)))
        print(f"ep{ep:>2}  loss={tot/max(n,1):.4f}  val_acc={acc:.4f}  "
              f"val_bal_acc={bacc:.4f}  val_macroF1={mf1:.4f}", flush=True)
        if mf1 > best:
            best = mf1
            torch.save({"model": model.state_dict(), "epoch": ep, "macro_f1": mf1,
                        "classes": NAMES}, out / "best.pt")
            # clip paths make video predictions pairable with EEG per clip;
            # `idx` alone is only a position in this run's own item list
            np.savez(out / "val_preds.npz", y=Y, probs=P, pred=pred, idx=I,
                     path=np.array([va[i][0] for i in I]))
            print(f"   saved best.pt (macro_f1={mf1:.4f})", flush=True)
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    z = np.load(out / "val_preds.npz")
    Y, P, pred, I = z["y"], z["probs"], z["pred"], z["idx"]
    subs = np.array([va[i][3] for i in I])
    res = report(Y, pred, P, f"POOLED ({len(ROOTS)} subjects) -- held-out val")
    res["per_subject"] = per_subject(Y, pred, subs)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
