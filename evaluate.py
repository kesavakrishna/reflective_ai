"""Run agent over questions.csv, grade each answer, report accuracy by category. See NOTES.md."""

import argparse
import os
import time

import pandas as pd

from agent import answer
from grader import grade
from memory import record_attempt_outcome, ATTEMPTS_CSV, ATTEMPTS_COLS

QUESTIONS_CSV = "questions.csv"
THROTTLE_SECS = int(os.getenv("EVAL_THROTTLE_SECS", "0"))


def _reset_attempts():
    pd.DataFrame(columns=ATTEMPTS_COLS).to_csv(ATTEMPTS_CSV, index=False)


def evaluate(limit=None, reset=False, questions_path=QUESTIONS_CSV, use_memory=True):
    if reset:
        _reset_attempts()
        # Re-import to pick up the empty file (module-level _attempts was loaded at import).
        import memory
        memory._attempts = pd.read_csv(ATTEMPTS_CSV)

    questions = pd.read_csv(questions_path)
    if limit:
        questions = questions.head(limit)

    results = []
    for i, (_, row) in enumerate(questions.iterrows()):
        if i > 0 and THROTTLE_SECS:
            time.sleep(THROTTLE_SECS)
        q, expected, category = row["question"], row["expected_answer"], row["category"]
        reply, attempt_id = answer(q, use_memory=use_memory)
        correct, note = grade(q, reply, expected)
        record_attempt_outcome(attempt_id, correct, expected, notes=note)
        results.append({"category": category, "correct": int(correct)})
        flag = "+" if correct else "-"
        print(f"[{flag}] {category:11s} | {q[:55]:55s} -> {reply[:40]}")

    df = pd.DataFrame(results)
    print("\n=== Accuracy ===")
    print(f"Overall: {df['correct'].mean():.1%}  ({df['correct'].sum()}/{len(df)})")
    print("\nBy category:")
    by_cat = df.groupby("category")["correct"].agg(["sum", "count", "mean"])
    by_cat.columns = ["correct", "total", "accuracy"]
    by_cat["accuracy"] = (by_cat["accuracy"] * 100).round(1).astype(str) + "%"
    print(by_cat.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--reset", action="store_true", help="wipe attempts.csv before running")
    parser.add_argument("--questions", default=QUESTIONS_CSV, help="CSV of questions to evaluate")
    parser.add_argument("--no-memory", action="store_true", help="disable lesson retrieval (memory-off run)")
    args = parser.parse_args()
    evaluate(limit=args.limit, reset=args.reset, questions_path=args.questions, use_memory=not args.no_memory)
