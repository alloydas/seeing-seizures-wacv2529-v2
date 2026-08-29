"""
Bulk Video Cropper
==================
Walks --input for .mp4 files and writes a cropped copy under --output,
mirroring the input directory structure.

Default crop is the **top-left 60% (width) × 80% (height)** region,
keeping the cropped pixels at native resolution (no resize). For an
800×600 source this yields a 480×480 clip.

The crop filter requires re-encoding (libx264, CRF 18 by default).
Per-clip sidecars (info.txt, eeg.edf) are copied alongside each
cropped video so the resulting tree is drop-in usable for retraining
train_classifier.py — just point --data_root at --output.

Examples:
    python crop_clips.py --input /path/to/repo/Data \
                         --output /path/to/repo/Data_cropped

    python crop_clips.py --input /path --output /path/out \
                         --workers 16 --threads 2 --crf 20 --preset fast

    python crop_clips.py --input ... --output ... --left_frac 0.5 --top_frac 0.8
"""

import argparse, os, shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Bulk-crop .mp4 clips with ffmpeg, in parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  required=True, help="Root folder to walk for .mp4 files")
    p.add_argument("--output", required=True, help="Mirror destination root")
    p.add_argument("--left_frac", type=float, default=0.60,
                   help="Width fraction of the source to KEEP, anchored at left (default 0.60)")
    p.add_argument("--top_frac",  type=float, default=0.80,
                   help="Height fraction of the source to KEEP, anchored at top (default 0.80)")
    p.add_argument("--x_frac", type=float, default=0.0,
                   help="Left offset as a width fraction (default 0.0 = anchor at left edge)")
    p.add_argument("--y_frac", type=float, default=0.0,
                   help="Top offset as a height fraction (default 0.0 = anchor at top edge)")
    p.add_argument("--crf",       type=int,   default=18,
                   help="x264 CRF (lower = better quality / larger file; default 18)")
    p.add_argument("--preset",    type=str,   default="fast",
                   choices=["ultrafast","superfast","veryfast","faster","fast",
                            "medium","slow","slower","veryslow"])
    p.add_argument("--workers",   type=int,   default=max(1, (os.cpu_count() or 4) // 2),
                   help="Number of parallel ffmpeg processes (default = cpu_count/2)")
    p.add_argument("--threads",   type=int,   default=2,
                   help="ffmpeg threads per worker (default 2)")
    p.add_argument("--no_sidecars", action="store_true",
                   help="Do not copy info.txt / eeg.edf next to each cropped video")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode even if the output file already exists")
    return p.parse_args()


def discover(input_root: Path):
    return sorted(input_root.rglob("*.mp4"))


def crop_one(job):
    (src, dst, left_frac, top_frac, x_frac, y_frac, crf, preset, threads,
     copy_sidecars, overwrite) = job
    src = Path(src); dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return str(src), "skip", None

    # trunc(...*2)*2 keeps the output even-sized (libx264 requires even W/H).
    vf = (f"crop=trunc(iw*{left_frac}/2)*2:"
          f"trunc(ih*{top_frac}/2)*2:"
          f"trunc(iw*{x_frac}/2)*2:"
          f"trunc(ih*{y_frac}/2)*2")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        "-threads", str(threads),
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if dst.exists():
            try: dst.unlink()
            except OSError: pass
        return str(src), "error", (e.stderr or "").strip()[:400]

    if copy_sidecars:
        for name in ("info.txt", "eeg.edf"):
            sib = src.parent / name
            if sib.exists():
                try:
                    shutil.copy2(sib, dst.parent / name)
                except Exception:
                    pass
    return str(src), "ok", None


def main():
    args = parse_args()
    input_root  = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        sys.exit(f"--input does not exist or is not a dir: {input_root}")
    if input_root == output_root:
        sys.exit("--output must differ from --input")
    if output_root in input_root.parents or input_root in output_root.parents:
        # allow but warn — would re-walk our own output otherwise if nested
        if output_root in input_root.parents:
            sys.exit(f"--output {output_root} is an ancestor of --input — refuse")

    sources = discover(input_root)
    if not sources:
        sys.exit(f"No .mp4 files found under {input_root}")

    jobs = []
    for src in sources:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        jobs.append((src, dst, args.left_frac, args.top_frac,
                     args.x_frac, args.y_frac, args.crf,
                     args.preset, args.threads,
                     not args.no_sidecars, args.overwrite))

    print(f"Cropping {len(jobs)} videos")
    print(f"  input : {input_root}")
    print(f"  output: {output_root}")
    print(f"  crop  : {args.left_frac*100:.0f}%w × {args.top_frac*100:.0f}%h "
          f"@ offset ({args.x_frac*100:.0f}%, {args.y_frac*100:.0f}%)  (no resize)")
    print(f"  codec : libx264  crf={args.crf}  preset={args.preset}  -an  yuv420p")
    print(f"  paral : workers={args.workers}, ffmpeg threads/worker={args.threads}  "
          f"(total ~{args.workers * args.threads} threads)")

    n_ok = n_skip = n_err = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(crop_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="vid"):
            src, status, msg = fut.result()
            if status == "ok":     n_ok += 1
            elif status == "skip": n_skip += 1
            else:
                n_err += 1
                errors.append((src, msg))
    print(f"\nDone. ok={n_ok}  skip={n_skip}  err={n_err}")
    if errors:
        print("\nFailures (first 10):")
        for src, msg in errors[:10]:
            print(f"  {src}\n    {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
