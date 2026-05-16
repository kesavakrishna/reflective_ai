"""Attempts + lessons + persisted FAISS index. See NOTES.md for design rationale."""

import os
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd

from gemini_client import embed as _embed_call

EMBED_DIM = 768
ATTEMPTS_CSV = os.getenv("ATTEMPTS_CSV", "attempts.csv")
LESSONS_CSV = os.getenv("LESSONS_CSV", "lessons.csv")
LESSONS_FAISS = os.getenv("LESSONS_FAISS", "lessons.faiss")

ATTEMPTS_COLS = [
    "id", "timestamp", "question", "answer",
    "correct", "expected_answer", "retrieved_lesson_ids", "notes",
]
LESSONS_COLS = ["id", "timestamp", "text", "source_attempt_id"]


def _load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return df


def _load_index():
    if os.path.exists(LESSONS_FAISS):
        return faiss.read_index(LESSONS_FAISS)
    return faiss.IndexFlatL2(EMBED_DIM)


_attempts = _load_csv(ATTEMPTS_CSV, ATTEMPTS_COLS)
_lessons = _load_csv(LESSONS_CSV, LESSONS_COLS)
_lessons_index = _load_index()


def _embed(text):
    return np.array(_embed_call(text), dtype="float32")


def _ensure_aligned():
    """Rebuild the FAISS index if it doesn't match lessons.csv row count."""
    global _lessons_index
    if _lessons_index.ntotal == len(_lessons):
        return
    _lessons_index = faiss.IndexFlatL2(EMBED_DIM)
    for txt in _lessons["text"]:
        _lessons_index.add(np.expand_dims(_embed(txt), 0))
    faiss.write_index(_lessons_index, LESSONS_FAISS)


_ensure_aligned()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _next_id(df):
    return int(df["id"].max()) + 1 if not df.empty else 1


def record_attempt(question, answer, retrieved_lesson_ids=None, notes=""):
    global _attempts
    row = {
        "id": _next_id(_attempts),
        "timestamp": _now(),
        "question": question,
        "answer": answer,
        "correct": None,
        "expected_answer": None,
        "retrieved_lesson_ids": ";".join(str(i) for i in (retrieved_lesson_ids or [])),
        "notes": notes,
    }
    _attempts = pd.concat([_attempts, pd.DataFrame([row])], ignore_index=True)
    _attempts.to_csv(ATTEMPTS_CSV, index=False)
    return row["id"]


def record_attempt_outcome(attempt_id, correct, expected_answer, notes=None):
    """Write grading result back to an existing attempt row. Stores correct as 1/0."""
    global _attempts
    mask = _attempts["id"] == attempt_id
    if not mask.any():
        raise ValueError(f"no attempt with id={attempt_id}")
    _attempts.loc[mask, "correct"] = int(bool(correct))
    _attempts.loc[mask, "expected_answer"] = expected_answer
    if notes is not None:
        _attempts.loc[mask, "notes"] = notes
    _attempts.to_csv(ATTEMPTS_CSV, index=False)


def record_lesson(text, source_attempt_id=None):
    global _lessons
    row = {
        "id": _next_id(_lessons),
        "timestamp": _now(),
        "text": text,
        "source_attempt_id": source_attempt_id,
    }
    _lessons = pd.concat([_lessons, pd.DataFrame([row])], ignore_index=True)
    _lessons.to_csv(LESSONS_CSV, index=False)
    _lessons_index.add(np.expand_dims(_embed(text), 0))
    faiss.write_index(_lessons_index, LESSONS_FAISS)
    return row["id"]


def retrieve_lessons(query, top_k=5):
    if _lessons_index.ntotal == 0:
        return []
    q = _embed(query)
    k = min(top_k, _lessons_index.ntotal)
    _, idxs = _lessons_index.search(np.expand_dims(q, 0), k)
    rows = _lessons.iloc[idxs[0]]
    return [{"id": int(r["id"]), "text": r["text"]} for _, r in rows.iterrows()]