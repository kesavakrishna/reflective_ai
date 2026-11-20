"""Phase 6: build fact-use (two-hop) questions from the phase 4 corrections.

Phases 3–5 only ever asked for the corrected answer itself, so a "fix" could be pure copying. Here each
correction (question Q, corrected answer A) becomes a question that needs A as an intermediate step
but asks for a *different* fact about A, referring to A only through Q's description:

  Q:       "Which 19th century American poet wrote over 1,000 poems ... published after her death?"  A: Emily Dickinson
  two-hop: "In which Massachusetts town was the 19th century American poet who wrote over 1,000 poems
            ... published after her death born?"                                       answer: Amherst
  direct:  "In which Massachusetts town was Emily Dickinson born?"                     answer: Amherst

The direct question checks that Gemma knows the second hop at all (solvability). Questions are written
by a different model family over the API (same reason as build_paraphrases.py). An item is rejected if:
the two-hop question names A or its own answer, the direct question doesn't name A, the new answer is
A, or the reply isn't valid JSON (3 tries).

  python build_twohop.py      # -> questions_twohop.csv, questions_twohop_direct.csv
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

PROMPT = """You are writing a test question for a language-model experiment.

Original trivia question: {question}
Its answer: {answer}

Write ONE new trivia question that needs the answer above as an intermediate step but asks for a DIFFERENT, well-known fact about "{answer}" (for example where it was born or is located, its nationality, its capital, a year, its creator, or another person or place closely tied to it).
Rules:
- Refer to "{answer}" ONLY through the description in the original question. Reuse enough of that description that a reader could identify it. Never name "{answer}" and never hint at the name.
- The new answer must be short (a name, place, number or year), unambiguous, and well established. It must not be "{answer}".
- Also write the "direct" version: the same new question, but naming "{answer}" explicitly instead of describing it.

Reply with ONLY a JSON object, no markdown:
{{"two_hop": "...", "direct": "...", "answer": "...", "aliases": ["other accepted spellings of the new answer"]}}"""
MAX_TRIES = 3


def parse_json(text, keys):
    """The first {...} in text as a dict with non-empty string `keys` and a cleaned "aliases" list."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(d.get(k), str) and d[k].strip() for k in keys):
        return None
    aliases = d.get("aliases", [])
    aliases = [str(a).strip() for a in (aliases if isinstance(aliases, list) else []) if str(a).strip() and "|" not in str(a)]
    out = {k: d[k].strip() for k in keys}
    if "answer" in out:
        out["answer"] = out["answer"].replace("|", "/")
    return {**out, "aliases": aliases}


def parse(text):
    return parse_json(text, ("two_hop", "direct", "answer"))


def check(item, bridge_aliases):
    """Reason the item is rejected, or None if it passes."""
    new = "|".join(dict.fromkeys([item["answer"], *item["aliases"]]))
    if grade("", item["two_hop"], bridge_aliases)[0]:
        return "two-hop names the bridge entity"
    if grade("", item["two_hop"], new)[0]:
        return "two-hop contains its own answer"
    if not grade("", item["direct"], bridge_aliases)[0]:
        return "direct question doesn't name the bridge entity"
    if grade("", item["answer"], bridge_aliases)[0]:
        return "new answer is the bridge entity"
    return None


def build(corrections, train, model):
    aliases = dict(zip(train["question"], train["expected_answer"]))
    rows, rejected = [], []
    for i, r in enumerate(corrections.itertuples()):
        bridge = aliases[r.question]
        reason = "no valid JSON"
        for _ in range(MAX_TRIES):
            item = parse(generate(PROMPT.format(question=r.question, answer=r.correct_answer), model=model))
            if item is None:
                continue
            reason = check(item, bridge)
            if reason is None:
                break
        if reason is not None:
            rejected.append({"source_question": r.question, "reason": reason})
            print(f"[{i}] REJECTED ({reason}): {r.question[:60]}", flush=True)
            continue
        rows.append({"two_hop": item["two_hop"], "direct": item["direct"],
                     "expected_answer": "|".join(dict.fromkeys([item["answer"], *item["aliases"]])),
                     "source_question": r.question, "bridge_answer": r.correct_answer, "bridge_aliases": bridge})
        if i % 20 == 0:
            print(f"[{i}] {r.correct_answer} -> {item['answer']}\n   {item['two_hop'][:100]}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corrections", default="runs/gemma3-4b-phase4/corrections.csv")
    p.add_argument("--train", default="questions_trivia_v2.csv")
    p.add_argument("--out", default="questions_twohop.csv")
    p.add_argument("--direct-out", default="questions_twohop_direct.csv")
    p.add_argument("--model", default=PARAPHRASE_MODEL)
    args = p.parse_args()
    corr = pd.read_csv(args.corrections)
    items, rejected = build(corr, pd.read_csv(args.train), args.model)
    keep = ["expected_answer", "source_question", "bridge_answer", "bridge_aliases"]
    two = items.rename(columns={"two_hop": "question"})[["question", *keep]].assign(category="twohop")
    direct = items.rename(columns={"direct": "question"})[["question", *keep]].assign(category="twohop_direct")
    two.to_csv(args.out, index=False)
    direct.to_csv(args.direct_out, index=False)
    rejected.to_csv(os.path.splitext(args.out)[0] + "_rejected.csv", index=False)
    print(f"\n{len(items)} items -> {args.out}, {args.direct_out}; {len(rejected)} rejected")


if __name__ == "__main__":
    main()
