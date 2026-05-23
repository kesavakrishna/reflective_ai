"""LLM client. Generation via OpenAI-compatible endpoint (NVIDIA NIM default, Ollama-swappable);
embeddings via a local sentence-transformers model — no API, no rate limits, never deprecated.
See NOTES.md for the provider rationale."""

import os
import time

from openai import (
    OpenAI,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Generation: any OpenAI-compatible endpoint. NVIDIA by default; for Ollama set
# LLM_BASE_URL=http://localhost:11434/v1 and LLM_MODEL=<local model>, key is ignored.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/phi-4-multimodal-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "ollama"

# Embeddings: local, CPU-friendly. all-MiniLM-L6-v2 is 384-dim (must match EMBED_DIM in memory.py).
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
_embed_model = SentenceTransformer(EMBED_MODEL)
RETRY_DELAYS = [15, 30, 60]


def _with_retry(fn, label, exc):
    for delay in RETRY_DELAYS:
        try:
            return fn()
        except exc:
            print(f"  [rate-limited on {label}, sleeping {delay}s]")
            time.sleep(delay)
    return fn()  # final attempt — let it raise if still blocked


def generate(prompt, model=None):
    def _call():
        resp = _client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic — the experiment needs reproducible answers
        )
        return resp.choices[0].message.content or ""
    # 429s plus transient 5xx / network blips (NVIDIA threw a 504 mid-run) are all retryable.
    retryable = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)
    return _with_retry(_call, label="generate", exc=retryable)


def embed(text):
    return _embed_model.encode(text).tolist()
