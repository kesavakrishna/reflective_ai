"""LLM client. Generation via OpenAI-compatible endpoint (NVIDIA NIM default, Ollama-swappable)
or a local Hugging Face model (LLM_BACKEND=local) whose activations we can inspect;
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
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "ollama"

# Local backend: runs the model in-process so interpretability code can reach its activations.
# 8-bit because Gemma 3 4B in bf16 (~8.6GB) doesn't fit the 8GB RTX 4060.
LLM_BACKEND = os.getenv("LLM_BACKEND", "api")  # "api" or "local"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/gemma-3-4b-it")
LOCAL_8BIT = os.getenv("LOCAL_8BIT", "1") == "1"
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))  # reflect.py lessons need room

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


_local = None  # (model, tokenizer), loaded on first use so the API path never imports CUDA stuff


def load_local():
    """Load the local model once. Public so interpretability code can share the same instance."""
    global _local
    if _local is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # per-tensor loading bar floods eval logs
        quant = BitsAndBytesConfig(load_in_8bit=True) if LOCAL_8BIT else None
        model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_MODEL,
            quantization_config=quant,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        _local = (model, AutoTokenizer.from_pretrained(LOCAL_MODEL))
    return _local


def _generate_local(prompt):
    import torch

    model, tok = load_local()
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=LOCAL_MAX_NEW_TOKENS, do_sample=False)  # greedy = temp 0
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate(prompt, model=None):
    if LLM_BACKEND == "local":
        return _generate_local(prompt)

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
