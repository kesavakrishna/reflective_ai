"""Answer a question with lesson-aware retrieval and constrained output. See NOTES.md."""

import os

from llm_client import generate
from memory import record_attempt, retrieve_lessons, retrieve_corrections

TOP_K_LESSONS = 3  # selective retrieval — with ~6 lessons, top-5 would inject nearly all of them
TOP_K_CORRECTIONS = 3  # same budget as lessons so the arms are comparable
# Which memory tracks to retrieve (phase 3): "lessons" (rules — phase 1/2 behavior), "corrections"
# (facts from past failures), or "both".
MEMORY_MODE = os.getenv("MEMORY_MODE", "lessons")
MEMORY_MODES = ("lessons", "corrections", "both")


def build_prompt(question, lessons, corrections=None):
    # With no corrections this is byte-identical to the phase 1/2 prompt, so those runs stay comparable.
    correction_block = ""
    if corrections:
        correction_block = (
            "Corrections to questions you answered wrong before, which may apply:\n"
            + "\n".join(
                f"- Q: {c['question']}\n  Your wrong answer: {c['wrong_answer']}. Correct answer: {c['correct_answer']}."
                for c in corrections
            )
            + "\n\n"
        )
    if lessons:
        lesson_block = (
            "Lessons from your past mistakes that may apply:\n"
            + "\n".join(f"- {l['text']}" for l in lessons)
            + "\n\n"
        )
    else:
        lesson_block = ""

    return (
        "Answer the question below. Reply with ONLY the answer itself — "
        "no preamble, no reasoning shown, no commentary on your process.\n\n"
        f"{correction_block}"
        f"{lesson_block}"
        f"Question: {question}\n"
        "Answer:"
    )


def answer(question, use_memory=True, memory_mode=None):
    mode = memory_mode or MEMORY_MODE
    if mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {MEMORY_MODES}, got {mode!r}")
    lessons = retrieve_lessons(question, top_k=TOP_K_LESSONS) if use_memory and mode != "corrections" else []
    corrections = retrieve_corrections(question, top_k=TOP_K_CORRECTIONS) if use_memory and mode != "lessons" else []
    prompt = build_prompt(question, lessons, corrections)
    reply = generate(prompt).strip()
    attempt_id = record_attempt(
        question=question,
        answer=reply,
        retrieved_lesson_ids=[l["id"] for l in lessons],
        retrieved_correction_ids=[c["id"] for c in corrections],
    )
    return reply, attempt_id
