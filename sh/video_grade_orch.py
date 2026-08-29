"""Autonomous video-grading seed orchestrator.

Waits until the 4 video DETECTION seeds (vid_s1..s4) finish, then runs the video
GRADING seed runs on the freed GPU0 -- max 2 concurrent (video decode is CPU-bound
on 32 cores; EEG grading runs light on GPU1 via its own gated launcher). Retries the
intermittent NVML-init assert. Fire-and-forget: launch with nohup, it self-terminates
when every grading seed has a results.json.

Grading seeds: 3-class (--group3) and 5-class (no flag), seeds {1,2}. With the seed-42
baselines (classifier_pooled20_g3 / _g5) that gives 3 points per cell for mean+/-std.
"""
import subprocess, time, os

ROOT = "/path/to/repo"
LOG  = f"{ROOT}/logs/seed_runs"
DET  = ["vid_s1", "vid_s2", "vid_s3", "vid_s4"]          # detection seeds to wait on
# (name, extra-args)  -- run on GPU0 after detection frees it
GRADE = [
    ("vid_g3_s1", ["--group3", "--seed", "1"]),
    ("vid_g3_s2", ["--group3", "--seed", "2"]),
    ("vid_g5_s1", ["--seed", "1"]),
    ("vid_g5_s2", ["--seed", "2"]),
]
MAXCONC = 2          # concurrent video trainers (CPU-decode cap)
GPU = "0"            # detection frees GPU0; EEG grading owns GPU1

def done(name):    return os.path.exists(f"{ROOT}/output/seed_runs/{name}/results.json")
def running(name): return bool(subprocess.run(["pgrep","-f",f"seed_runs/{name}"],
                                               capture_output=True).stdout.strip())
def log(m):        print(f"[{time.strftime('%F %T')}] {m}", flush=True)

# ---- phase 1: wait for detection to finish -------------------------------------
log(f"orchestrator up; waiting for detection seeds {DET} to finish")
while not all(done(d) for d in DET):
    time.sleep(120)
log("all detection seeds done -> starting video grading on GPU0")

# ---- phase 2: run grading seeds, <=MAXCONC at a time, retry NVML ----------------
attempts = {name: 0 for name, _ in GRADE}
pending  = [g for g in GRADE if not done(g[0])]
while pending:
    active = [g for g in GRADE if running(g[0])]
    if len(active) < MAXCONC:
        for g in list(pending):
            name, args = g
            if done(name):
                pending.remove(g); continue
            if running(name):
                continue
            if attempts[name] >= 4:
                log(f"{name} GAVE UP after {attempts[name]} tries"); pending.remove(g); continue
            attempts[name] += 1
            cmd = ["python3", "train_pooled.py", "--arch", "r2plus1d", *args,
                   "--epochs", "12", "--batch_size", "16", "--workers", "8",
                   "--split_seed", "49", "--output", f"output/seed_runs/{name}"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False")
            with open(f"{LOG}/{name}.log", "w") as f:
                subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            log(f"launched {name} (try {attempts[name]}) on gpu{GPU}")
            time.sleep(120)   # let it get past init before counting the slot
            break
    # reap: drop finished, requeue silent crashes (no results, not running)
    for g in list(pending):
        name, _ = g
        if done(name):
            log(f"{name} finished OK"); pending.remove(g)
    time.sleep(60)
log("video grading orchestrator done")
