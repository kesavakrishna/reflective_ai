"""Step 4: turn failed attempts into lessons (and, phase 3, corrections). See NOTES.md.

Reads failures (correct == 0) from attempts.csv, asks the model to write itself a
transferable lesson for each, and stores them via memory.record_lesson. Each failure also
becomes a correction (question, wrong answer, right answer) via memory.record_correction —
no model call. Retrieval is NOT enabled here — that's step 5. Run this on the baseline run,
then read the lessons by hand before turning memory on.
"""

import argparse

import pandas as pd

import memory
from memory import record_lesson, record_correction, ATTEMPTS_CSV
from llm_client import generate

# v2: force diagnosis before the rule, demand a self-applicable strategy, and explicitly
# ban "be careful / check sources" platitudes (v1 produced only those). The diagnosis-first
# structure is the crux — if the model can't name the failure mode, the rule won't transfer.
REFLECT_PROMPT = (
    "You answered a question incorrectly.\n\n"
    "Question: {question}\n"
    "Your answer: {answer}\n"
    "Correct answer: {expected}\n\n"
    "First, in a few words, identify the SPECIFIC type of error you made "
    "(e.g. confused which event came earlier; gave the value in the wrong unit; miscounted). "
    "Then write ONE actionable rule that would catch this error on similar questions in the "
    "future, using ONLY your own reasoning — you cannot look anything up or use any tools.\n\n"
    "Do NOT give generic advice like \"double-check\", \"be careful\", or \"consult reliable "
    "sources\". Be specific to the kind of question.\n\n"
    "Format: <error type> -> <rule>"
)


def reflect(reset=False, corrections_only=False):
    if reset:
        memory.reset_corrections()
        if not corrections_only:
            memory.reset_lessons()

    attempts = pd.read_csv(ATTEMPTS_CSV)
    failures = attempts[attempts["correct"] == 0]
    print(f"{len(failures)} failures to reflect on\n")

    for _, row in failures.iterrows():
        # TriviaQA expected answers are "canonical|alias|alias…". Phase 2 passed the whole list into
        # the reflection prompt; use the canonical answer only.
        correct = str(row["expected_answer"]).split("|")[0]
        cid = record_correction(row["question"], row["answer"], correct, source_attempt_id=int(row["id"]))
        if corrections_only:
            continue
        prompt = REFLECT_PROMPT.format(
            question=row["question"],
            answer=row["answer"],
            expected=correct,
        )
        lesson = generate(prompt).strip()
        lid = record_lesson(lesson, source_attempt_id=int(row["id"]))
        print(f"[lesson {lid} / correction {cid}] from attempt {int(row['id'])}")
        print(f"  Q:      {row['question'][:72]}")
        print(f"  wrong:  {str(row['answer'])[:40]}   correct: {correct}")
        print(f"  lesson: {lesson}\n")

    if corrections_only:
        print(f"Recorded {len(failures)} corrections (lessons left as they are)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="wipe existing lessons and corrections first")
    p.add_argument("--corrections-only", action="store_true",
                   help="record corrections only; keep existing lessons (reuse a previous run's)")
    args = p.parse_args()
    reflect(reset=args.reset, corrections_only=args.corrections_only)
