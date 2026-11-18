"""Reword every training question so it asks for the same answer in different words (phase 3).

A correction retrieved for the exact question it came from is a lookup table. A correction that
fixes a *reworded* question is closer to having learned the fact. This builds that test set.

Paraphrases come from a different model family over the API (LLM_BACKEND=api; default Mistral
Large 4 — Llama 3.3 70B was retired from NVIDIA on 2026-08-26), not from Gemma — the model under
test shouldn't write its own test. A paraphrase is rejected if it contains any accepted
answer alias (the grader would mark the question itself as "answered") or equals the original.

  python build_paraphrases.py
"""

import argparse
import os

os.environ["LLM_BACKEND"] = "api"

import pandas as pd  # noqa: E402

from grader import grade  # noqa: E402
from llm_client import generate  # noqa: E402

PARAPHRASE_PROMPT = (
    "Rewrite this trivia question so that it asks for exactly the same answer, but with clearly "
    "different wording and sentence structure. Keep every fact needed to identify the answer. "
    "Do not include or hint at the answer itself.\n\n"
    "Question: {question}\n\n"
    "Reply with ONLY the rewritten question."
)
MAX_TRIES = 3
PARAPHRASE_MODEL = "mistralai/mistral-large-4-1t-a52b"


def paraphrase(question, expected, model):
    for _ in range(MAX_TRIES):
        text = generate(PARAPHRASE_PROMPT.format(question=question), model=model).strip().strip('"').strip()
        leaked, _ = grade(question, text, expected)
        if text and text.lower() != question.lower() and not leaked:
            return text
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="questions_trivia.csv")
    p.add_argument("--out", default="questions_trivia_paraphrase.csv")
    p.add_argument("--model", default=PARAPHRASE_MODEL)
    args = p.parse_args()

    df = pd.read_csv(args.questions)
    rows, rejected = [], 0
    for i, r in df.iterrows():
        text = paraphrase(r["question"], r["expected_answer"], args.model)
        if text is None:
            rejected += 1
            print(f"[{i}] REJECTED: {r['question'][:70]}")
            continue
        rows.append({"question": text, "expected_answer": r["expected_answer"],
                     "category": r["category"], "source_question": r["question"]})
        if i % 25 == 0:
            print(f"[{i}] {r['question'][:60]}\n   -> {text[:60]}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n{len(rows)} paraphrases saved to {args.out}; {rejected} rejected")


if __name__ == "__main__":
    main()
