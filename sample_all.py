import subprocess, os, sys
from tqdm import tqdm
os.makedirs("logs", exist_ok=True)

# Sample hand/body/face sequentially. Edit CONFIG or override on the CLI
# (CLI wins). See the README "Sampling" section for what each key does.
CONFIG = {
    "eval.note": "CSLDaily",
    "eval.ode_stepnum": 20,
    "eval.split": "test",
    "test_data": "[CSL-Daily]",
    "eval.candidate_num": 1,
    "eval.batch_size": 64,
    "eval.max_samples": 100,   # -1 = whole split
    "eval.seed": 123,
    "flip_left_hand": True,
}

# Per-stream checkpoints at $SIGNSPARK_CKPT_DIR/<stream>/ema_0.9999_<iter>.pt.
# Set CHECKPOINT_ITER=None to use each yaml's default model_path.
CHECKPOINT_BASE = os.getenv("SIGNSPARK_CKPT_DIR", "checkpoints")
CHECKPOINT_ITER = 200000
CHECKPOINTS = {
    name: f"{CHECKPOINT_BASE}/{name}/ema_0.9999_{CHECKPOINT_ITER}.pt"
    for name in ["hand", "body", "face"]
} if CHECKPOINT_ITER is not None else {}

cli_args = sys.argv[1:]
cli_keys = {a.split("=", 1)[0] for a in cli_args if "=" in a}
config_args = [f"{k}={v}" for k, v in CONFIG.items() if k not in cli_keys]
all_args = config_args + cli_args

note = next((a.split("=", 1)[1] for a in all_args if a.startswith("eval.note=")), None)
log_suffix = f"_{note}" if note else ""
subprocess_args = all_args

# Sequential (not parallel) to keep peak VRAM at a single stream's footprint.
for name in tqdm(["hand", "body", "face"], desc="Evaluating"):
    ckpt_arg = (
        [f"eval.model_path={CHECKPOINTS[name]}"]
        if name in CHECKPOINTS and "eval.model_path" not in cli_keys
        else []
    )
    cmd = ["python", "sample.py", f"data_v2={name}", "data_v2/common@common=inference"] + ckpt_arg + subprocess_args
    with open(f"logs/eval_{name}{log_suffix}.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        proc.wait()

for name in ["hand", "body", "face"]:
    with open(f"logs/eval_{name}{log_suffix}.log", "r") as f:
        for line in f:
            if "### Written the decoded output to" in line:
                print(f"Results for {name}{log_suffix}:")
                print(line.strip())
    os.remove(f"logs/eval_{name}{log_suffix}.log")

os.rmdir("logs")
    