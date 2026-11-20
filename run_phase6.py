"""Phase 6: can the model *use* a corrected fact, or only repeat it? See RESULTS.md ("Phase 6 design").

Questions come from build_twohop.py: for each phase 4 correction, a two-hop question that needs the
corrected answer as an intermediate step, plus a direct version that names it (solvability check).

  retrieval (int8 pipeline, evaluate.py, phase 4 correction store):
      direct / off, twohop / off, corrections (ungated), gated (0.7), oracle (retrieval by the
      two-hop question's own source question, so the right correction is always in the prompt)
  weights (dequantized pipeline, weight_edit.py test, the phase 5 config and cached test deltas):
      direct + twohop / off, edited, placebo

Every evaluation is skipped if its output exists, so a killed run resumes.

  python run_phase6.py --layer L --lam X      # the phase 5 selected config
"""

import argparse
import os
import subprocess
import sys

from compare import compare
from run_phase4 import make_env

GATE = "0.7"
PHASE4 = os.path.join("runs", "gemma3-4b-phase4")
D = os.path.join("runs", "gemma3-4b-twohop")
SETS = {"twohop": "questions_twohop.csv", "direct": "questions_twohop_direct.csv"}
RETRIEVAL = {
    "off": ["--no-memory"],
    "corrections": ["--memory-mode", "corrections"],
    "gated": ["--memory-mode", "corrections", "--gate", GATE],
    "oracle": ["--memory-mode", "corrections", "--retrieve-by", "source_question"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--lam", type=float, required=True)
    args = p.parse_args()
    os.makedirs(D, exist_ok=True)
    env = make_env(D, PHASE4)

    runs = [("direct", "off")] + [("twohop", c) for c in RETRIEVAL]
    for split, cond in runs:
        out = os.path.join(D, f"{split}_{cond}.csv")
        if not os.path.exists(out):
            print(f"\n##### retrieval: {split} / {cond} #####", flush=True)
            subprocess.run([sys.executable, "evaluate.py", "--reset", "--questions", SETS[split],
                            *RETRIEVAL[cond], "--out", out], env=env, check=True)

    print("\n##### weights: off / edited / placebo #####", flush=True)
    subprocess.run([sys.executable, "weight_edit.py", "test", "--layer", str(args.layer), "--lam", str(args.lam),
                    "--name", "gemma3-4b-twohop-edit", "--sets", *[f"{k}={v}" for k, v in SETS.items()]],
                   env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}, check=True)

    print("\n##### comparison (retrieval) #####")
    for cond in ("corrections", "gated", "oracle"):
        compare(os.path.join(D, "twohop_off.csv"), os.path.join(D, f"twohop_{cond}.csv"),
                out=os.path.join(D, f"twohop_{cond}_pairs.csv"), label=f"twohop / {cond}")
        print()


if __name__ == "__main__":
    main()
