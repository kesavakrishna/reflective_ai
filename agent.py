"""Answer a question with lesson-aware retrieval and constrained output. See NOTES.md."""

import os

from llm_client import generate
from memory import record_attempt, retrieve_lessons, retrieve_corrections, random_corrections

TOP_K_LESSONS = 3  # selective retrieval — with ~6 lessons, top-5 would inject nearly all of them
TOP_K_CORRECTIONS = 3  # same budget as lessons so the arms are comparable
# Which memory tracks to retrieve (phase 3): "lessons" (rules — phase 1/2 behavior), "corrections"
# (facts from past failures), or "both".
# Phase 4 placebos replace the corrections block: "placebo_lorem" (filler of the same length, no
# facts) and "placebo_random" (3 random corrections instead of the 3 nearest).
MEMORY_MODE = os.getenv("MEMORY_MODE", "lessons")
MEMORY_MODES = ("lessons", "corrections", "both", "placebo_lorem", "placebo_random")
# Phase 4 gate: use corrections only if the nearest stored question has cosine similarity >= this.
# None = no gate. A gated-out question gets the memory-off prompt exactly.
CORRECTION_GATE = float(os.environ["CORRECTION_GATE"]) if os.getenv("CORRECTION_GATE") else None

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
    "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
)


def _correction_block(corrections):
    if not corrections:
        return ""
    return (
        "Corrections to questions you answered wrong before, which may apply:\n"
        + "\n".join(
            f"- Q: {c['question']}\n  Your wrong answer: {c['wrong_answer']}. Correct answer: {c['correct_answer']}."
            for c in corrections
        )
        + "\n\n"
    )


def lorem_block(length):
    """Fact-free filler: "Notes:" + lorem ipsum, cut at a word boundary to `length` characters."""
    head, tail = "Notes:\n", "\n\n"
    body = (LOREM * (length // len(LOREM) + 2))[: max(length - len(head) - len(tail), 0)]
    body = body.rsplit(" ", 1)[0] if " " in body else body
    return head + body + tail


CHAIN_INSTRUCTION = (  # phase 7: name the described entity first, then answer what is asked
    "Answer the question below. The question may describe a person, place or thing without naming it. "
    "First name what it describes, then answer the question that is actually asked. "
    "Reply in exactly this format, and nothing else:\n"
    "Entity: <name>\nAnswer: <short answer>\n\n"
)


def final_answer(reply):
    """The part after the last "Answer:" (phase 7 chain replies); the whole reply if there is none."""
    i = reply.rfind("Answer:")
    return reply[i + len("Answer:"):].strip() if i >= 0 else reply.strip()


def build_prompt(question, lessons, corrections=None, correction_block=None, style=None):
    # With no corrections this is byte-identical to the phase 1/2 prompt, so those runs stay comparable.
    # correction_block overrides the formatted corrections (the lorem placebo uses this slot).
    if correction_block is None:
        correction_block = _correction_block(corrections)
    if lessons:
        lesson_block = (
            "Lessons from your past mistakes that may apply:\n"
            + "\n".join(f"- {l['text']}" for l in lessons)
            + "\n\n"
        )
    else:
        lesson_block = ""

    if style == "chain":
        return f"{CHAIN_INSTRUCTION}{correction_block}{lesson_block}Question: {question}\n"
    return (
        "Answer the question below. Reply with ONLY the answer itself — "
        "no preamble, no reasoning shown, no commentary on your process.\n\n"
        f"{correction_block}"
        f"{lesson_block}"
        f"Question: {question}\n"
        "Answer:"
    )


def answer(question, use_memory=True, memory_mode=None, gate=None, query=None, style=None):
    """query: retrieve corrections with this text instead of the question (phase 6 "oracle" retrieval:
    the two-hop question's own source question, so retrieval always finds the right correction)."""
    mode = memory_mode or MEMORY_MODE
    gate = CORRECTION_GATE if gate is None else gate
    if mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {MEMORY_MODES}, got {mode!r}")
    lessons, corrections, block = [], [], None
    if use_memory:
        if mode in ("lessons", "both"):
            lessons = retrieve_lessons(question, top_k=TOP_K_LESSONS)
        if mode in ("corrections", "both"):
            corrections = retrieve_corrections(query or question, top_k=TOP_K_CORRECTIONS)
            if gate is not None and corrections and corrections[0]["sim"] < gate:
                corrections = []
        elif mode == "placebo_random":
            corrections = random_corrections(question, k=TOP_K_CORRECTIONS)
        elif mode == "placebo_lorem":
            real = _correction_block(retrieve_corrections(question, top_k=TOP_K_CORRECTIONS))
            block = lorem_block(len(real)) if real else ""
    prompt = build_prompt(question, lessons, corrections, correction_block=block, style=style)
    reply = generate(prompt).strip()
    attempt_id = record_attempt(
        question=question,
        answer=reply,
        retrieved_lesson_ids=[l["id"] for l in lessons],
        retrieved_correction_ids=[c["id"] for c in corrections],
    )
    return reply, attempt_id
