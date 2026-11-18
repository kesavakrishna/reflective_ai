"""Phase 4: does the 0.7 similarity gate hold on fresh questions, and is off-topic harm just noise?
See RESULTS.md ("Phase 4 design") for the pre-registered predictions.

  B. placebo — the phase 3 held-out set with the corrections block replaced by lorem-ipsum filler of
     the same length, or by 3 random corrections. Uses the phase 3 correction store; no API.
  A. gate    — a fresh 250/250 TriviaQA split (seed 1) plus paraphrases of its training questions.
     Corrections come from the fresh training failures. Each set runs off / corrections / gated.

Paraphrases are generated over the API in the background while the GPU runs B. Every evaluation is
skipped if its output exists, so a killed run resumes where it stopped (same as run_twotrack.py).

  python run_phase4.py
"""

import os
import shutil
import subprocess
import sys

import pandas as pd

from compare import compare

GATE = "0.7"  # fixed before the run (phase 3 finding 15); do not tune on these results
PHASE3 = os.path.join("runs", "gemma3-4b-twotrack")
GATE_DIR = os.path.join("runs", "gemma3-4b-phase4")
PLACEBO_DIR = os.path.join("runs", "gemma3-4b-placebo")
FRESH = {
    "train": "questions_trivia_v2.csv",
    "paraphrase": "questions_trivia_v2_paraphrase.csv",
    "holdout": "questions_trivia_v2_holdout.csv",
}
CONDITIONS = {
    "off": ["--no-memory"],
    "corrections": ["--memory-mode", "corrections"],
    "gated": ["--memory-mode", "corrections", "--gate", GATE],
}


def make_env(d, corrections_dir):
    return {
        **os.environ,
        "LLM_BACKEND": "local",
        "HF_HUB_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
        "ATTEMPTS_CSV": os.path.join(d, "attempts.csv"),
        "LESSONS_CSV": os.path.join(d, "lessons.csv"),  # unused (no lessons arm); kept out of the repo root
        "LESSONS_FAISS": os.path.join(d, "lessons.faiss"),
        "CORRECTIONS_CSV": os.path.join(corrections_dir, "corrections.csv"),
        "CORRECTIONS_FAISS": os.path.join(corrections_dir, "corrections.faiss"),
    }


def run(step, args, env):
    print(f"\n##### {step} #####", flush=True)
    subprocess.run([sys.executable, *args], env=env, check=True)


def evaluate(d, split, qfile, cond, flags, env):
    out = os.path.join(d, f"{split}_{cond}.csv")
    if not os.path.exists(out):
        run(f"{split} / {cond}", ["evaluate.py", "--reset", "--questions", qfile, *flags, "--out", out], env)
    return out


def main():
    os.makedirs(GATE_DIR, exist_ok=True)
    os.makedirs(PLACEBO_DIR, exist_ok=True)

    para = None
    if not os.path.exists(FRESH["paraphrase"]):
        print("##### paraphrases (API, background) #####", flush=True)
        log = open(os.path.join(GATE_DIR, "paraphrase.log"), "w", encoding="utf-8")
        para = subprocess.Popen(
            [sys.executable, "build_paraphrases.py", "--questions", FRESH["train"], "--out", FRESH["paraphrase"]],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=log, stderr=subprocess.STDOUT)

    # B. placebo on the phase 3 held-out set, against the phase 3 correction store.
    env_b = make_env(PLACEBO_DIR, PHASE3)
    for name in ("holdout_off.csv", "holdout_corrections.csv"):
        if not os.path.exists(os.path.join(PLACEBO_DIR, name)):
            shutil.copy(os.path.join(PHASE3, name), os.path.join(PLACEBO_DIR, name))
    for arm in ("placebo_lorem", "placebo_random"):
        evaluate(PLACEBO_DIR, "holdout", "questions_trivia_holdout.csv", arm, ["--memory-mode", arm], env_b)

    # A. gate on the fresh split. Corrections are built from the fresh training failures first.
    env_a = make_env(GATE_DIR, GATE_DIR)
    train_off = evaluate(GATE_DIR, "train", FRESH["train"], "off", CONDITIONS["off"], env_a)
    if not os.path.exists(os.path.join(GATE_DIR, "corrections.csv")):
        pd.read_csv(train_off).drop(columns=["category"], errors="ignore").to_csv(env_a["ATTEMPTS_CSV"], index=False)
        run("build corrections from fresh training failures", ["reflect.py", "--reset", "--corrections-only"], env_a)
    for split in ("train", "holdout"):
        for cond, flags in CONDITIONS.items():
            evaluate(GATE_DIR, split, FRESH[split], cond, flags, env_a)
    if para is not None:
        print("\n##### waiting for paraphrases #####", flush=True)
        if para.wait() != 0:
            raise SystemExit(f"build_paraphrases.py failed; see {GATE_DIR}/paraphrase.log")
    for cond, flags in CONDITIONS.items():
        evaluate(GATE_DIR, "paraphrase", FRESH["paraphrase"], cond, flags, env_a)

    print("\n##### comparison #####")
    for arm in ("placebo_lorem", "placebo_random", "corrections"):
        compare(os.path.join(PLACEBO_DIR, "holdout_off.csv"), os.path.join(PLACEBO_DIR, f"holdout_{arm}.csv"),
                out=os.path.join(PLACEBO_DIR, f"holdout_{arm}_pairs.csv"), label=f"placebo / holdout / {arm}")
        print()
    for split in FRESH:
        for arm in ("corrections", "gated"):
            compare(os.path.join(GATE_DIR, f"{split}_off.csv"), os.path.join(GATE_DIR, f"{split}_{arm}.csv"),
                    out=os.path.join(GATE_DIR, f"{split}_{arm}_pairs.csv"), label=f"fresh / {split} / {arm}")
            print()


if __name__ == "__main__":
    main()
