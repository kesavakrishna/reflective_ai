"""Two-tier grader: normalized string match where safe, LLM judge otherwise. See NOTES.md."""

import re
import unicodedata

from gemini_client import generate


def _normalize(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _string_match(answer, expected):
    return _normalize(expected) in _normalize(answer)


def _expected_appears_in_question(question, expected):
    return _normalize(expected) in _normalize(question)


def _llm_judge(question, answer, expected):
    prompt = (
        "You are grading an answer. The expected answer is the ground truth — "
        "do not override it based on outside knowledge.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Model's answer: {answer}\n\n"
        "Does the model's answer convey the same factual content as the expected answer? "
        "Reply with exactly one word: yes or no."
    )
    verdict = generate(prompt).text.strip().lower()
    return verdict.startswith("yes")


def grade(question, answer, expected):
    """Return (correct: bool, note: str). note is 'string_match' or 'llm_judge'."""
    if not _expected_appears_in_question(question, expected):
        if _string_match(answer, expected):
            return True, "string_match"
    verdict = _llm_judge(question, answer, expected)
    return verdict, "llm_judge"
