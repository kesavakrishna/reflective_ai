"""Phase 8: near-duplicate questions with *different* answers, to stress the 0.7 similarity gate.

For each phase 4 correction (question Q, corrected answer A), a question that reuses as much of Q's
wording as possible but whose correct answer is a different entity B:

  Q:  "Which 19th century American poet wrote over 1,000 poems which were published ... after her death?"  A: Emily Dickinson
  Q': "Which 19th century American poet wrote 'Leaves of Grass', first published in 1855?"                B: Walt Whitman

If the gate lets Q's correction into Q''s prompt and the model copies A, a correction has caused harm
on a question it doesn't apply to. Written by Nemotron over the API (as build_paraphrases.py).
Rejected: B matches an alias of A, Q' contains B, Q' equals Q, or no valid JSON in 3 tries.

  python build_neardup.py      # -> questions_neardup.csv
"""

import argparse
import os

os.environ["LLM_BACKEND"] = "api"

import pandas as pd  # noqa: E402

from build_paraphrases import PARAPHRASE_MODEL  # noqa: E402
from build_twohop import parse_json  # noqa: E402
from grader import grade  # noqa: E402
from llm_client import generate  # noqa: E402

PROMPT = """You are writing a test question for a language-model experiment.

Original trivia question: {question}
Its answer: {answer}

Write ONE new trivia question that reuses as much of the original wording as possible (change as few words as you can), but whose correct answer is a DIFFERENT, well-known person, place or thing of the same kind, not "{answer}".
Rules:
- The new question must have exactly one well-established correct answer, and it must not be "{answer}".
- Do not include the new answer in the question.

Reply with ONLY a JSON object, no markdown:
{{"question": "...", "answer": "...", "aliases": ["other accepted spellings of the new answer"]}}"""
MAX_TRIES = 3


def check(item, question, bridge_aliases):
    new = "|".join(dict.fromkeys([item["answer"], *item["aliases"]]))
    if item["question"].strip().lower() == question.strip().lower():
        return "same as the original"
    if grade("", item["answer"], bridge_aliases)[0] or any(grade("", a, new)[0] for a in bridge_aliases.split("|")):
        return "answer is the original answer"
    if grade("", item["question"], new)[0]:
        return "question contains its own answer"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corrections", default="runs/gemma3-4b-phase4/corrections.csv")
    p.add_argument("--train", default="questions_trivia_v2.csv")
    p.add_argument("--out", default="questions_neardup.csv")
    p.add_argument("--model", default=PARAPHRASE_MODEL)
    args = p.parse_args()
    corr = pd.read_csv(args.corrections)
    aliases = dict(zip(pd.read_csv(args.train)["question"], pd.read_csv(args.train)["expected_answer"]))
    part = args.out + ".partial"  # one row per finished item, so a stopped build resumes
    done = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame(columns=["source_question", "status"])
    rows = done[done["status"] == "ok"].drop(columns="status").to_dict("records")
    rejected = done[done["status"] != "ok"][["source_question", "status"]].rename(columns={"status": "reason"}).to_dict("records")
    seen = set(done["source_question"]) if len(done) else set()
    for i, r in enumerate(corr.itertuples()):
        if r.question in seen:
            continue
        orig = aliases[r.question]
        reason = "no valid JSON"
        for _ in range(MAX_TRIES):
            item = parse_json(generate(PROMPT.format(question=r.question, answer=r.correct_answer), model=args.model),
                              ("question", "answer"))
            if item is None:
                continue
            reason = check(item, r.question, orig)
            if reason is None:
                break
        if reason is not None:
            rejected.append({"source_question": r.question, "reason": reason})
            pd.DataFrame([{"source_question": r.question, "status": reason}]).to_csv(
                part, mode="a", header=not os.path.exists(part), index=False)
            print(f"[{i}] REJECTED ({reason}): {r.question[:60]}", flush=True)
            continue
        rows.append({"question": item["question"],
                     "expected_answer": "|".join(dict.fromkeys([item["answer"], *item["aliases"]])),
                     "category": "neardup", "source_question": r.question,
                     "source_answer": r.correct_answer, "source_aliases": orig})
        pd.DataFrame([{**rows[-1], "status": "ok"}]).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
        if i % 20 == 0:
            print(f"[{i}] {r.correct_answer} -> {item['answer']}\n   {item['question'][:100]}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rejected).to_csv(os.path.splitext(args.out)[0] + "_rejected.csv", index=False)
    print(f"\n{len(rows)} near-duplicates -> {args.out}; {len(rejected)} rejected")


if __name__ == "__main__":
    main()
