"""Autonomous RodEpil scheduler: launches the remaining backbones onto whichever
GPU has room, packing multiple per GPU, until all are running. Memory-aware so it
never OOMs; loops until every pending backbone is launched, then exits (the
training procs keep running independently)."""
import subprocess, time, os, torch

# (arch, script, batch, est_GB_needed)  -- heavy 32-frame nets get small batches
PENDING = [
    ("mvit",     "train_rodepil.py",    6, 8),
    ("videomae", "train_rodepil_hf.py", 4, 8),
    ("slowfast", "train_rodepil.py",    6, 9),
    ("swin",     "train_rodepil.py",    4, 11),
    ("swin_s",   "train_rodepil.py",    3, 11),
]
BASE = "--all_folds --root RodEpil_std --epochs 8 --workers 6".split()

def done(a):    return os.path.exists(f"output/rodepil_{a}/results.json")
def running(a): return bool(subprocess.run(["pgrep","-f",f"output/rodepil_{a}"],
                                           capture_output=True).stdout.strip())
def free_gb(g): return torch.cuda.mem_get_info(g)[0] / 1e9

pending = list(PENDING)
print(f"[scheduler] managing {len(pending)} RodEpil backbones", flush=True)
while pending:
    launched_this_pass = False
    for g in (0, 1):
        if not pending:
            break
        arch, script, batch, est = pending[0]
        if done(arch) or running(arch):
            pending.pop(0); continue
        if free_gb(g) > est + 2.0:
            cmd = ["python3", script, "--arch", arch, *BASE,
                   "--batch_size", str(batch), "--output", f"output/rodepil_{arch}"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            with open(f"logs/rodepil/{arch}.log", "w") as log:
                subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[scheduler] launched {arch} on gpu{g} "
                  f"(free {free_gb(g):.1f}GB, batch {batch})", flush=True)
            pending.pop(0); launched_this_pass = True
            time.sleep(90)   # let it allocate before re-checking memory
    time.sleep(60)
print("[scheduler] all RodEpil backbones launched", flush=True)
