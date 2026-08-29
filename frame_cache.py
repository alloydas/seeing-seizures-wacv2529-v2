"""
Shared reader for the pre-decoded frame caches written by build_frame_cache.py.

Returns the raw post-resize RGB uint8 frames (T, S, S, 3). Each trainer applies its own
normalisation on top, because they differ: train_pooled.py uses Kinetics statistics and
(C,T,H,W), VideoMAE uses ImageNet statistics and (T,C,H,W), TimeSformer uses 0.45/0.225
and (T,C,H,W). The frame *extraction* is identical across all three -- same
sample_frame_indices, same sequential grab, same nearest-frame fallback, same INTER_AREA
resize -- which is what makes one cache per (frames, size) geometry sufficient.
"""
import json
import os

import numpy as np

_OPEN = {}


class RawFrameCache:
    def __init__(self, d, frames, size):
        meta = json.load(open(os.path.join(d, "index.json")))
        if meta["frames"] != frames or meta["size"] != size:
            raise ValueError(f"cache {d} is {meta['frames']}x{meta['size']}, "
                             f"caller wants {frames}x{size}")
        self.row = {p: i for i, p in enumerate(meta["paths"])}
        self.bad = set(meta.get("unreadable", []))
        self.shape = (meta["n"], frames, size, size, 3)
        self.path = os.path.join(d, "frames.u8")
        self.mm = None                      # opened lazily so DataLoader forks are safe

    def raw(self, p):
        """uint8 (T, S, S, 3), or None if this clip is not cached."""
        i = self.row.get(p)
        if i is None or p in self.bad:
            return None
        if self.mm is None:
            self.mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=self.shape)
        return self.mm[i]


def get(d, frames, size):
    if d is None:
        return None
    key = (d, frames, size)
    if key not in _OPEN:
        _OPEN[key] = RawFrameCache(d, frames, size)
    return _OPEN[key]
