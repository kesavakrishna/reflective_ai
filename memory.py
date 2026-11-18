"""Attempts + lessons + corrections, each memory track with a persisted FAISS index.
Lessons are rules ("compare actual years"), indexed by their text. Corrections are facts (a failed
question with its right answer), indexed by the question — phase 3's two-track memory.
See NOTES.md for design rationale."""

import os
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd

from llm_client import embed as _embed_call

EMBED_DIM = 384  # all-MiniLM-L6-v2 (local). Must match the embedding model in llm_client.py.
ATTEMPTS_CSV = os.getenv("ATTEMPTS_CSV", "attempts.csv")
LESSONS_CSV = os.getenv("LESSONS_CSV", "lessons.csv")
LESSONS_FAISS = os.getenv("LESSONS_FAISS", "lessons.faiss")
CORRECTIONS_CSV = os.getenv("CORRECTIONS_CSV", "corrections.csv")
CORRECTIONS_FAISS = os.getenv("CORRECTIONS_FAISS", "corrections.faiss")

ATTEMPTS_COLS = [
    "id", "timestamp", "question", "answer",
    "correct", "expected_answer", "retrieved_lesson_ids", "retrieved_correction_ids", "notes",
]
LESSONS_COLS = ["id", "timestamp", "text", "source_attempt_id"]
CORRECTIONS_COLS = ["id", "timestamp", "question", "wrong_answer", "correct_answer", "source_attempt_id"]


def _load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return df


def _load_index(path):
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatL2(EMBED_DIM)


_attempts = _load_csv(ATTEMPTS_CSV, ATTEMPTS_COLS)
_lessons = _load_csv(LESSONS_CSV, LESSONS_COLS)
_lessons_index = _load_index(LESSONS_FAISS)
_corrections = _load_csv(CORRECTIONS_CSV, CORRECTIONS_COLS)
_corrections_index = _load_index(CORRECTIONS_FAISS)


def _embed(text):
    return np.array(_embed_call(text), dtype="float32")


def _aligned(index, df, key_col, path):
    """Return index, rebuilt from df[key_col] if it doesn't match the CSV's row count."""
    if index.ntotal == len(df):
        return index
    index = faiss.IndexFlatL2(EMBED_DIM)
    for txt in df[key_col]:
        index.add(np.expand_dims(_embed(txt), 0))
    faiss.write_index(index, path)
    return index


_lessons_index = _aligned(_lessons_index, _lessons, "text", LESSONS_FAISS)
_corrections_index = _aligned(_corrections_index, _corrections, "question", CORRECTIONS_FAISS)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _next_id(df):
    return int(df["id"].max()) + 1 if not df.empty else 1


def record_attempt(question, answer, retrieved_lesson_ids=None, retrieved_correction_ids=None, notes=""):
    global _attempts
    row = {
        "id": _next_id(_attempts),
        "timestamp": _now(),
        "question": question,
        "answer": answer,
        "correct": None,
        "expected_answer": None,
        "retrieved_lesson_ids": ";".join(str(i) for i in (retrieved_lesson_ids or [])),
        "retrieved_correction_ids": ";".join(str(i) for i in (retrieved_correction_ids or [])),
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


def reset_lessons():
    """Wipe all lessons + the FAISS index. Lets us regenerate cleanly while tuning."""
    global _lessons, _lessons_index
    _lessons = pd.DataFrame(columns=LESSONS_COLS)
    _lessons.to_csv(LESSONS_CSV, index=False)
    _lessons_index = faiss.IndexFlatL2(EMBED_DIM)
    if os.path.exists(LESSONS_FAISS):
        os.remove(LESSONS_FAISS)


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


def reset_corrections():
    global _corrections, _corrections_index
    _corrections = pd.DataFrame(columns=CORRECTIONS_COLS)
    _corrections.to_csv(CORRECTIONS_CSV, index=False)
    _corrections_index = faiss.IndexFlatL2(EMBED_DIM)
    if os.path.exists(CORRECTIONS_FAISS):
        os.remove(CORRECTIONS_FAISS)


def record_correction(question, wrong_answer, correct_answer, source_attempt_id=None):
    """Store the fact behind a failure. Indexed by the question, so a new phrasing of the same
    question should retrieve it — that's what the paraphrase set tests."""
    global _corrections
    row = {
        "id": _next_id(_corrections),
        "timestamp": _now(),
        "question": question,
        "wrong_answer": wrong_answer,
        "correct_answer": correct_answer,
        "source_attempt_id": source_attempt_id,
    }
    _corrections = pd.concat([_corrections, pd.DataFrame([row])], ignore_index=True)
    _corrections.to_csv(CORRECTIONS_CSV, index=False)
    _corrections_index.add(np.expand_dims(_embed(question), 0))
    faiss.write_index(_corrections_index, CORRECTIONS_FAISS)
    return row["id"]


def retrieve_corrections(query, top_k=3):
    if _corrections_index.ntotal == 0:
        return []
    q = _embed(query)
    k = min(top_k, _corrections_index.ntotal)
    _, idxs = _corrections_index.search(np.expand_dims(q, 0), k)
    rows = _corrections.iloc[idxs[0]]
    return [
        {"id": int(r["id"]), "question": r["question"], "wrong_answer": r["wrong_answer"],
         "correct_answer": r["correct_answer"]}
        for _, r in rows.iterrows()
    ]