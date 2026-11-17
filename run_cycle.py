"""Run the full experiment cycle into runs/<name>/ so no run overwrites another.

  1. training, memory off     -> train_off.csv
  2. reflect on its failures  -> lessons.csv + lessons.faiss (in the run dir, not the repo root)
  3. training, memory on      -> train_on.csv
  4. held-out, memory off     -> holdout_off.csv
  5. held-out, memory on      -> holdout_on.csv
  6. paired fixed/broke comparison for both sets -> *_pairs.csv

Each step is a subprocess because memory.py reads its file paths from env at import time.

  python run_cycle.py --name gemma3-4b --backend local
"""

import argparse
import os
import subprocess
import sys

from compare import compare


def run(step, args, env):
    print(f"\n##### {step} #####", flush=True)
    subprocess.run([sys.executable, *args], env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run directory name under runs/")
    p.add_argument("--backend", default=os.getenv("LLM_BACKEND", "api"), choices=["api", "local"])
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    args = p.parse_args()

    d = os.path.join("runs", args.name)
    os.makedirs(d, exist_ok=True)
    env = {
        **os.environ,
        "LLM_BACKEND": args.backend,
        "ATTEMPTS_CSV": os.path.join(d, "attempts.csv"),
        "LESSONS_CSV": os.path.join(d, "lessons.csv"),
        "LESSONS_FAISS": os.path.join(d, "lessons.faiss"),
    }
    if args.backend == "local":
        env["HF_HUB_OFFLINE"] = "1"  # model is cached; don't hit the network
    lim = ["--limit", str(args.limit)] if args.limit else []
    out = lambda f: ["--out", os.path.join(d, f)]

    run("1/5 training, memory OFF", ["evaluate.py", "--reset", "--no-memory", *lim, *out("train_off.csv")], env)
    run("2/5 reflect on failures", ["reflect.py", "--reset"], env)
    run("3/5 training, memory ON", ["evaluate.py", "--reset", *lim, *out("train_on.csv")], env)
    holdout = ["--questions", "questions_holdout.csv"]
    run("4/5 held-out, memory OFF", ["evaluate.py", "--reset", "--no-memory", *holdout, *lim, *out("holdout_off.csv")], env)
    run("5/5 held-out, memory ON", ["evaluate.py", "--reset", *holdout, *lim, *out("holdout_on.csv")], env)

    print("\n##### comparison #####")
    for split in ("train", "holdout"):
        compare(os.path.join(d, f"{split}_off.csv"), os.path.join(d, f"{split}_on.csv"),
                out=os.path.join(d, f"{split}_pairs.csv"), label=f"{args.name} / {split}")
        print()


if __name__ == "__main__":
    main()
