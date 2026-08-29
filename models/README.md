# Model weights — Seeing Seizures (anonymous WACV 2027 submission)

Best checkpoint per task, hosted at the anonymous folder:
**[https://drive.google.com/drive/folders/1mAclhqRjE9wAIW_B6Jh2yF-nfW54Df5L?usp=drive_link](https://drive.google.com/drive/folders/1mAclhqRjE9wAIW_B6Jh2yF-nfW54Df5L?usp=drive_link)**

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
