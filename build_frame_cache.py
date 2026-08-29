#!/usr/bin/env python3
"""
Pre-decodes every clip's sampled frames into a uint8 memmap, so training reads an array
instead of re-decoding H.264 on every epoch.

This is only sound because sample_frame_indices() is deterministic -- plain
np.linspace over the clip, with no per-epoch temporal jitter -- so a clip yields the
identical frames on every pass. Across the 120-run seeding sweep the same decode was
being repeated ~1,400 times per clip.

The cache stores frames AFTER BGR->RGB conversion and resize but BEFORE scaling and
Kinetics normalisation, i.e. exactly the `out` list inside load_clip(). Reconstruction is
therefore bit-identical:  arr.astype(float32)/255 -> normalise -> transpose.

Layout:
  <dir>/frames.u8    memmap, shape (N, T, S, S, 3), uint8, C-order
  <dir>/index.json   {"paths": [...], "frames": T, "size": S, "n": N}

Usage:
  python3 build_frame_cache.py --frames 16 --size 112 --out cache_frames/f16s112 --workers 14
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import sample_frame_indices

_G = {}


def _init(path, shape, frames, size):
    _G["mm"] = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
    _G["frames"], _G["size"] = frames, size


def _one(job):
    row, path = job
    T, S = _G["frames"], _G["size"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return row, False
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(n_total, T)
    if idxs is None:
        cap.release()
        return row, False
    target = set(int(x) for x in idxs)
    grabbed, i = {}, 0
    # sequential grab(), matching load_clip: seeks are expensive on copy-mode mp4s
    while i < n_total and len(grabbed) < len(target):
        if not cap.grab():
            break
        if i in target:
            ok, fr = cap.retrieve()
            if ok:
                grabbed[i] = fr
        i += 1
    cap.release()
    if not grabbed:
        return row, False
    keys = sorted(grabbed)
    out = []
    for ix in idxs:
        f = grabbed.get(int(ix))
        if f is None:
            f = grabbed[min(keys, key=lambda k: abs(k - int(ix)))]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA))
    _G["mm"][row] = np.stack(out, axis=0)
    return row, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N clips")
    a = ap.parse_args()

    import train_pooled as tp
    items = tp.discover()
    paths = [it[0] for it in items]
    if a.limit:
        paths = paths[:a.limit]
    n = len(paths)
    shape = (n, a.frames, a.size, a.size, 3)
    gb = np.prod(shape) / 2**30
    os.makedirs(a.out, exist_ok=True)
    binp = os.path.join(a.out, "frames.u8")
    print(f"{n} clips -> {shape}  ({gb:.1f} GB) at {binp}")

    mm = np.memmap(binp, dtype=np.uint8, mode="w+", shape=shape)
    del mm

    bad = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(binp, shape, a.frames, a.size)) as ex:
        for k, (row, ok) in enumerate(ex.map(_one, list(enumerate(paths)), chunksize=8)):
            if not ok:
                bad.append(paths[row])
            if (k + 1) % 2000 == 0:
                print(f"  {k+1}/{n}  ({len(bad)} unreadable)", flush=True)

    json.dump({"paths": paths, "frames": a.frames, "size": a.size, "n": n,
               "unreadable": bad},
              open(os.path.join(a.out, "index.json"), "w"))
    print(f"done: {n - len(bad)} cached, {len(bad)} unreadable")
    print(f"wrote {a.out}/index.json")


if __name__ == "__main__":
    main()
