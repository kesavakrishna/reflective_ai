"""Answer a question with lesson-aware retrieval and constrained output. See NOTES.md."""

from gemini_client import generate
from memory import record_attempt, retrieve_lessons

TOP_K_LESSONS = 5


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


def answer(question):
    lessons = retrieve_lessons(question, top_k=TOP_K_LESSONS)
    prompt = build_prompt(question, lessons)
    response = generate(prompt)
    reply = response.text.strip()
    attempt_id = record_attempt(
        question=question,
        answer=reply,
        retrieved_lesson_ids=[l["id"] for l in lessons],
    )
    return reply, attempt_id
