"""Phase 3: two-track memory — lessons (rules) vs corrections (facts) vs both. See RESULTS.md for
the pre-registered design and predictions.

Reuses the phase 2 run (same model, same 129 lessons): its memory-off and lessons-only runs are
copied, not re-run — the prompt with no corrections is byte-identical, decoding is greedy. The new
work is corrections (built from the phase 2 training failures, no model call), the corrections and
both arms on training + held-out, and all four conditions on the paraphrase set.

Every evaluation is skipped if its output already exists, so a killed run resumes where it stopped.

  python run_twotrack.py --source runs/gemma3-4b-trivia --name gemma3-4b-twotrack
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

from compare import compare

SETS = {
    "train": "questions_trivia.csv",
    "paraphrase": "questions_trivia_paraphrase.csv",
    "holdout": "questions_trivia_holdout.csv",
}
ARMS = ("lessons", "corrections", "both")


def run(step, args, env):
    print(f"\n##### {step} #####", flush=True)
    subprocess.run([sys.executable, *args], env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="runs/gemma3-4b-trivia", help="phase 2 run to reuse")
    p.add_argument("--name", default="gemma3-4b-twotrack")
    args = p.parse_args()

    d = os.path.join("runs", args.name)
    os.makedirs(d, exist_ok=True)
    env = {
        **os.environ,
        "LLM_BACKEND": "local",
        "HF_HUB_OFFLINE": "1",
        "ATTEMPTS_CSV": os.path.join(d, "attempts.csv"),
        "LESSONS_CSV": os.path.join(d, "lessons.csv"),
        "LESSONS_FAISS": os.path.join(d, "lessons.faiss"),
        "CORRECTIONS_CSV": os.path.join(d, "corrections.csv"),
        "CORRECTIONS_FAISS": os.path.join(d, "corrections.faiss"),
    }

    # Reuse phase 2: lessons + the runs whose prompts are unchanged.
    reuse = {"lessons.csv": "lessons.csv", "lessons.faiss": "lessons.faiss",
             "train_off.csv": "train_off.csv", "train_on.csv": "train_lessons.csv",
             "holdout_off.csv": "holdout_off.csv", "holdout_on.csv": "holdout_lessons.csv"}
    for src, dst in reuse.items():
        if not os.path.exists(os.path.join(d, dst)):
            shutil.copy(os.path.join(args.source, src), os.path.join(d, dst))

    # Corrections from the phase 2 training failures. Those attempt rows predate the canonical-first
    # alias order, so refresh expected_answer from the question file (same alias set, reordered).
    if not os.path.exists(os.path.join(d, "corrections.csv")):
        attempts = pd.read_csv(os.path.join(d, "train_off.csv"))
        canon = pd.read_csv(SETS["train"]).set_index("question")["expected_answer"]
        attempts["expected_answer"] = attempts["question"].map(canon)
        attempts.drop(columns=["category"], errors="ignore").to_csv(env["ATTEMPTS_CSV"], index=False)
        run("build corrections from training failures", ["reflect.py", "--reset", "--corrections-only"], env)

    for split, qfile in SETS.items():
        conditions = [("off", ["--no-memory"])] + [(arm, ["--memory-mode", arm]) for arm in ARMS]
        for cond, flags in conditions:
            out = os.path.join(d, f"{split}_{cond}.csv")
            if os.path.exists(out):
                continue
            run(f"{split} / {cond}", ["evaluate.py", "--reset", "--questions", qfile, *flags, "--out", out], env)

    print("\n##### comparison #####")
    for split in SETS:
        for arm in ARMS:
            compare(os.path.join(d, f"{split}_off.csv"), os.path.join(d, f"{split}_{arm}.csv"),
                    out=os.path.join(d, f"{split}_{arm}_pairs.csv"), label=f"{split} / {arm}")
            print()


if __name__ == "__main__":
    main()
