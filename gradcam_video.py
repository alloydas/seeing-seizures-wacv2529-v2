"""
Grad-CAM for the R(2+1)D-18 seizure detector: where in the frame (and when) does
the model look to decide 'seizure'? Hooks the last conv block (layer4), backprops
the seizure-class logit, and overlays the spatiotemporal CAM on the input frames.

  python gradcam_video.py --ckpt classifiers/classifier_pooled2_v2/best.pt \
      --clip data/Data_RN197_cropped/.../seizure_.. --out gradcam_out/foo.png
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from train_classifier import build_model, load_clip, KINETICS_MEAN, KINETICS_STD

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradcam(model, x, cls):
    acts, grads = {}, {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    logit = model(x)
    prob = torch.softmax(logit, 1)[0, cls].item()
    logit[0, cls].backward()
    h1.remove(); h2.remove()
    A, G = acts["a"], grads["g"]                       # (1,C,T',H',W')
    w = G.mean(dim=(2, 3, 4), keepdim=True)            # channel weights
    cam = F.relu((w * A).sum(1, keepdim=True))         # (1,1,T',H',W')
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), prob            # (T,H,W)


def denorm(x):  # (C,T,H,W) normalized -> (T,H,W,3) uint8 RGB
    a = x.cpu().numpy().transpose(1, 2, 3, 0)
    a = (a * KINETICS_STD + KINETICS_MEAN)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def write_overlay_video(mp4, cam, prob, out_path, alpha=0.5, max_side=None):
    """Paint the (T,H,W) CAM onto every native-resolution frame and write an mp4.

    CAM frame t corresponds to source frame linspace(0, n_total-1, T)[t] (the same
    sampling load_clip uses), so we linearly interpolate between CAM frames to keep
    the heatmap smooth across all real frames.
    """
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        print(f"  [video] could not open {mp4}"); return
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h); w, h = int(w * s), int(h * s)
    T = cam.shape[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        # fractional CAM-time for this native frame
        p = 0.0 if n_total <= 1 else i / (n_total - 1) * (T - 1)
        t0 = int(np.floor(p)); t1 = min(t0 + 1, T - 1); fr = p - t0
        cam_i = (1 - fr) * cam[t0] + fr * cam[t1]          # (Hc,Wc)
        cam_i = cv2.resize(cam_i, (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((np.clip(cam_i, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)         # BGR
        blend = cv2.addWeighted(frame, 1 - alpha, heat, alpha, 0)
        txt = f"P(seizure)={prob:.3f}  attn={cam_i.mean():.2f}"
        cv2.putText(blend, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blend, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(blend)
        i += 1
    cap.release(); vw.release()
    print(f"  saved {out_path}  ({i} frames @ {fps:.1f}fps, {w}x{h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="classifiers/classifier_pooled2_v2/best.pt")
    ap.add_argument("--clip", required=True, help="clip dir or video.mp4")
    ap.add_argument("--nclass", type=int, default=2)
    ap.add_argument("--cls", type=int, default=1, help="class to explain (1=seizure)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", help="PNG panel output (optional)")
    ap.add_argument("--video", help="mp4 overlay output (optional)")
    ap.add_argument("--alpha", type=float, default=0.5, help="heatmap opacity for --video")
    ap.add_argument("--max_side", type=int, default=None,
                    help="cap the longer side of the overlay video (px); default = native")
    a = ap.parse_args()
    if not a.out and not a.video:
        ap.error("nothing to do: pass --out (PNG) and/or --video (mp4)")

    mp4 = a.clip if a.clip.endswith(".mp4") else str(Path(a.clip) / "video.mp4")
    arr = load_clip(mp4, a.frames, 112)
    if arr is None: sys.exit(f"could not read {mp4}")
    x = torch.from_numpy(arr).unsqueeze(0).to(DEV)

    model = build_model(a.nclass).to(DEV)
    sd = torch.load(a.ckpt, map_location=DEV, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    cam, prob = gradcam(model, x, a.cls)
    rgb = denorm(x[0])                                  # (T,H,W,3)

    T = a.frames
    name = Path(a.clip).name
    if a.out:
        show = list(range(0, T, max(1, T // 8)))[:8]
        fig, axes = plt.subplots(2, len(show), figsize=(2.0 * len(show), 4.4))
        for j, t in enumerate(show):
            axes[0, j].imshow(rgb[t]); axes[0, j].axis("off")
            axes[0, j].set_title(f"t={t}", fontsize=8)
            hm = cam[t]
            axes[1, j].imshow(rgb[t])
            axes[1, j].imshow(hm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[1, j].axis("off")
        axes[0, 0].set_ylabel("frame", fontsize=9)
        fig.suptitle(f"Grad-CAM  |  {name}  |  P(seizure)={prob:.3f}", fontsize=11)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(a.out, dpi=130, bbox_inches="tight"); plt.close()
        print(f"  saved {a.out}")

    if a.video:
        write_overlay_video(mp4, cam, prob, a.video, alpha=a.alpha, max_side=a.max_side)

    # temporal importance (mean CAM per frame) -> when does it look?
    tp = cam.reshape(T, -1).mean(1)
    print(f"{name}: P(seizure)={prob:.3f}  peak-attention frame={int(tp.argmax())}  "
          f"CAM temporal profile={np.round(tp,2).tolist()}")


if __name__ == "__main__":
    main()
