"""Answer a question with lesson-aware retrieval and constrained output. See NOTES.md."""

from llm_client import generate
from memory import record_attempt, retrieve_lessons

TOP_K_LESSONS = 3  # selective retrieval — with ~6 lessons, top-5 would inject nearly all of them


def build_prompt(question, lessons):
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
        f"{lesson_block}"
        f"Question: {question}\n"
        "Answer:"
    )


def answer(question, use_memory=True):
    lessons = retrieve_lessons(question, top_k=TOP_K_LESSONS) if use_memory else []
    prompt = build_prompt(question, lessons)
    reply = generate(prompt).strip()
    attempt_id = record_attempt(
        question=question,
        answer=reply,
        retrieved_lesson_ids=[l["id"] for l in lessons],
    )
    return reply, attempt_id
