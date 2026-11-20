"""Phase 7 (chain prompt on the two-hop set) and phase 8 (near-duplicate gate stress test).
See RESULTS.md ("Phase 7 design", "Phase 8 design"). int8 pipeline, phase 4 correction store.
Every evaluation is skipped if its output exists, so a killed run resumes.

  python run_phase78.py            # needs questions_twohop.csv; questions_neardup.csv for phase 8
"""

import os
import subprocess
import sys

from compare import compare
from run_phase4 import make_env

GATE = "0.7"
PHASE4 = os.path.join("runs", "gemma3-4b-phase4")
P7 = os.path.join("runs", "gemma3-4b-chain")
P8 = os.path.join("runs", "gemma3-4b-neardup")
CHAIN = {
    "off_chain": ["--no-memory", "--style", "chain"],
    "gated_chain": ["--memory-mode", "corrections", "--gate", GATE, "--style", "chain"],
    "oracle_chain": ["--memory-mode", "corrections", "--retrieve-by", "source_question", "--style", "chain"],
}
NEARDUP = {
    "off": ["--no-memory"],
    "corrections": ["--memory-mode", "corrections"],
    "gated": ["--memory-mode", "corrections", "--gate", GATE],
}


def run_all(d, qfile, conds, prefix):
    os.makedirs(d, exist_ok=True)
    env = make_env(d, PHASE4)
    for cond, flags in conds.items():
        out = os.path.join(d, f"{prefix}_{cond}.csv")
        if not os.path.exists(out):
            print(f"\n##### {d}: {cond} #####", flush=True)
            subprocess.run([sys.executable, "evaluate.py", "--reset", "--questions", qfile, *flags, "--out", out],
                           env=env, check=True)


def main():
    run_all(P7, "questions_twohop.csv", CHAIN, "twohop")
    if not os.path.exists("questions_neardup.csv"):
        sys.exit("questions_neardup.csv missing; run build_neardup.py, then rerun this (phase 7 is done)")
    run_all(P8, "questions_neardup.csv", NEARDUP, "neardup")
    for cond in ("corrections", "gated"):
        compare(os.path.join(P8, "neardup_off.csv"), os.path.join(P8, f"neardup_{cond}.csv"),
                out=os.path.join(P8, f"neardup_{cond}_pairs.csv"), label=f"neardup / {cond}")
        print()


if __name__ == "__main__":
    main()
