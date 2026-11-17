"""Deterministic grader. Numeric comparison for numeric answers, normalized string
containment for entity answers. No LLM judge — see NOTES.md §Step 3 addendum 3 for why."""

import re
import unicodedata


def _normalize(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUMBER_WORDS = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def _parse_number(s):
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    if m:
        return float(m.group().replace(",", ""))
    # Gemma answers small counts in words ("Nine") — digits-only parsing marked those wrong.
    w = re.search(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", str(s).lower())
    return float(_NUMBER_WORDS[w.group(1)]) if w else None


def grade(question, answer, expected):
    """Return (correct: bool, note: str). Fully deterministic, no API calls."""
    exp_num = _parse_number(expected)
    if exp_num is not None:
        ans_num = _parse_number(answer)
        if ans_num is None:
            return False, "numeric"
        if "." in str(expected):  # decimal expected → allow rounding slack
            return abs(ans_num - exp_num) <= 0.05, "numeric"
        return round(ans_num) == round(exp_num), "numeric"  # integer → exact
    return _normalize(expected) in _normalize(answer), "string_match"
