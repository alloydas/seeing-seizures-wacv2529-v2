# Model weights — Seeing Seizures (anonymous WACV 2027 submission)

Best checkpoint per task. Verify downloads with `sha256sum -c SHA256SUMS`.

| Task | Backbone | File | Macro-F1 (this run) | Macro-F1 (5 seeds) |
|---|---|---|---|---|
| Detection (2-class) | SlowFast-R50 | `slowfast_bin_s1.pt` | 0.977 | 0.973 ± 0.003 |
| 3-class severity | X3D-M | `x3d_g3_s2.pt` | 0.801 | 0.794 ± 0.005 |
| 5-class severity | MViTv2-S | `mvit_g5_s5.pt` | 0.653 | 0.660 ± 0.016 |

```python
import torch
from train_pooled import build_video_model   # code repository root on sys.path
model, frames, size = build_video_model("slowfast", 2, pretrained=False)
model.load_state_dict(torch.load("slowfast_bin_s1.pt", map_location="cpu")["model"])
model.eval()
```

Architectures/classes: `("slowfast", 2)`, `("x3d", 3)`, `("mvit", 5)`. Kinetics-normalized
float32 clips `(B,C,T,H,W)`; SlowFast 32 frames @ 224 px, X3D-M and MViTv2-S 16 @ 224.
Use `torch==2.4.x` (2.6 flips `torch.load(weights_only=)` and rejects these files).

Caveats: the pytorchvideo X3D head contains a Softmax, so this model outputs
probabilities, not logits — do not double-softmax, and strip `blocks[5].activation`
before gradient attribution. Cite the MViT checkpoint by its seed distribution
(0.660 ± 0.016): MViT training is non-deterministic at fixed seed, so any single
number is one draw.

## EEG weights (single-channel biopotential)

Retrained on the post-repair cache; every cell reproduces its published seed-42 value
to within ±0.004 macro-F1.

| Task | Model | File | Macro-F1 (this run / published) |
|---|---|---|---|
| Detection | TCN (best EEG) | `eeg_tcn_bin.pt` | 0.978 / 0.978 |
| 3-class | TCN | `eeg_tcn_g3.pt` | 0.767 / 0.767 |
| 5-class | TCN | `eeg_tcn_g5.pt` | 0.567 / 0.563 |
| Detection | GRU (reference) | `eeg_gru_bin.pt` | 0.970 / 0.974 |
| 3-class | GRU | `eeg_gru_g3.pt` | 0.750 / 0.750 |
| 5-class | GRU | `eeg_gru_g5.pt` | 0.562 / 0.560 |

```python
from train_pooled_eeg import build_model
model = build_model("tcn", 2, 128)          # ("gru"|"tcn", n_classes, hidden)
model.load_state_dict(torch.load("eeg_tcn_bin.pt", map_location="cpu")["model"])
```

Inputs are z-scored 6 s windows at 125 Hz, shape `(B, 750, 1)`, from the first
recorded channel; clip decisions pool per-window posteriors with log-mean. Caveat:
19 of 20 cohort subjects carry a single ECG lead, so these are single-lead
biopotential models, not multichannel EEG.
