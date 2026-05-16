"""Thin Gemini SDK wrapper with rate-limit retry. See NOTES.md §Step 3 addendum."""

import os
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
RETRY_DELAYS = [30, 60, 90]  # seconds; worst-case ~3min before surfacing the error


def _with_retry(fn, label):
    for delay in RETRY_DELAYS:
        try:
            return fn()
        except ResourceExhausted:
            print(f"  [rate-limited on {label}, sleeping {delay}s]")
            time.sleep(delay)
    return fn()  # final attempt — let it raise if still rate-limited


def generate(prompt, model_name=None):
    model = genai.GenerativeModel(model_name or MODEL_NAME)
    return _with_retry(lambda: model.generate_content(prompt), label="generate")


def embed(text, model_name=None):
    resp = _with_retry(
        lambda: genai.embed_content(
            model=model_name or EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
        ),
        label="embed",
    )
    return resp["embedding"]
