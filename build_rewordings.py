"""Phase 9: automatic rewordings of each corrected question, used as extra keys by the memit_aug arm.

These play the role of the embedding model in retrieval: a cheap, automatic way to cover more than
one wording of a stored question. They are not the test paraphrases. They come from a different
prompt, and any rewording identical to the test paraphrase of the same question, or containing an
accepted answer, is rejected. NOTES.md reports how similar they turn out to be to the test paraphrases.

  python build_rewordings.py --corrections runs/gemma3-4b-twotrack/corrections.csv \\
      --train questions_trivia.csv --paraphrases questions_trivia_paraphrase.csv --out rewordings_trivia.csv
  python build_rewordings.py      # phase 4 defaults -> rewordings_trivia_v2.csv
"""

import argparse
import json
import os
import re

os.environ["LLM_BACKEND"] = "api"

import pandas as pd  # noqa: E402

from build_paraphrases import PARAPHRASE_MODEL  # noqa: E402
from grader import grade  # noqa: E402
from llm_client import generate  # noqa: E402

PROMPT = """Here is a trivia question:

{question}

Write {n} different ways a quiz host might ask for the same answer. Keep every clue needed to identify the answer, but vary the wording and the order of the clues. Do not include or hint at the answer.

Reply with ONLY a JSON list of {n} strings, no markdown."""
N = 2
MAX_TRIES = 3


def parse_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        xs = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).replace("*", "").strip().strip('"') for x in xs if isinstance(x, str) and x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corrections", default="runs/gemma3-4b-phase4/corrections.csv")
    p.add_argument("--train", default="questions_trivia_v2.csv")
    p.add_argument("--paraphrases", default="questions_trivia_v2_paraphrase.csv")
    p.add_argument("--out", default="rewordings_trivia_v2.csv")
    p.add_argument("--model", default=PARAPHRASE_MODEL)
    args = p.parse_args()
    corr = pd.read_csv(args.corrections)
    tr = pd.read_csv(args.train)
    aliases = dict(zip(tr["question"], tr["expected_answer"]))
    para = pd.read_csv(args.paraphrases)
    test_para = dict(zip(para["source_question"], para["question"].str.strip().str.lower()))
    part = args.out + ".partial"  # appended after every question, so a stopped build resumes
    done = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame(columns=["source_question", "rewording"])
    rows, short = done.dropna().to_dict("records"), 0
    seen = set(done["source_question"])
    for i, r in enumerate(corr.itertuples()):
        if r.question in seen:
            continue
        keep = []
        for _ in range(MAX_TRIES):
            for x in parse_list(generate(PROMPT.format(question=r.question, n=N), model=args.model)):
                low = x.lower()
                if (low != r.question.strip().lower() and low != test_para.get(r.question)
                        and not grade(r.question, x, aliases[r.question])[0]
                        and low not in {k.lower() for k in keep}):
                    keep.append(x)
            if len(keep) >= N:
                break
        short += len(keep) < N
        new = [{"source_question": r.question, "rewording": x} for x in keep[:N]] or [{"source_question": r.question, "rewording": None}]
        pd.DataFrame(new).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
        rows += [x for x in new if x["rewording"] is not None]
        if i % 20 == 0:
            print(f"[{i}] {r.question[:70]}\n   -> {keep[:1]}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n{len(rows)} rewordings for {len(corr)} questions -> {args.out}; {short} questions got fewer than {N}")


if __name__ == "__main__":
    main()
