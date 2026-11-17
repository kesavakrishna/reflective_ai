"""Capture Gemma's internal state on every fixed/broke/stayed pair from a run. See NOTES.md.

For each question, two forward passes (no generation) — the memory-OFF prompt and the memory-ON
prompt with the same lessons that run retrieved. At the last prompt token (where the answer's first
token is predicted) we save, per decoder layer:
  - the residual stream (pre-final-norm)  → probe input
  - logit lens: logit(first token of the OFF answer) − logit(first token of the ON answer)
    → where in depth the lesson flips Maastricht to Lisbon
plus the model's actual probability on the OFF answer's first token (the "confidence" baseline the
probe has to beat).

  python interp_capture.py runs/gemma3-4b-trivia
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/gemma3-4b-trivia"
# memory.py (imported via agent) loads its files at import — point it at the run's lessons, not the repo root's.
os.environ.setdefault("LESSONS_CSV", os.path.join(RUN_DIR, "lessons.csv"))
os.environ.setdefault("LESSONS_FAISS", os.path.join(RUN_DIR, "lessons.faiss"))
os.environ.setdefault("ATTEMPTS_CSV", os.path.join(RUN_DIR, "attempts.csv"))
os.environ["LLM_BACKEND"] = "local"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from agent import build_prompt  # noqa: E402
from llm_client import load_local  # noqa: E402


def first_token(tok, text):
    ids = tok(str(text).strip(), add_special_tokens=False)["input_ids"]
    return ids[0] if ids else tok.unk_token_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    pairs = pd.concat(
        [pd.read_csv(os.path.join(args.run_dir, f"{s}_pairs.csv")).assign(split=s) for s in ("train", "holdout")],
        ignore_index=True,
    )
    if args.limit:
        pairs = pairs.head(args.limit)
    lessons = pd.read_csv(os.path.join(args.run_dir, "lessons.csv")).set_index("id")["text"]

    model, tok = load_local()
    lm = model.model.language_model
    layers, final_norm, lm_head = lm.layers, lm.norm, model.lm_head
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    captured = {}
    hooks = [
        layer.register_forward_hook(lambda m, i, o, idx=idx: captured.__setitem__(idx, (o[0] if isinstance(o, tuple) else o)[0, -1]))
        for idx, layer in enumerate(layers)
    ]

    def run(prompt, tok_a, tok_b):
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        captured.clear()
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
            resid = torch.stack([captured[i] for i in range(len(layers))])  # [L, d]
            lens = lm_head(final_norm(resid)).float()
            if softcap:
                lens = torch.tanh(lens / softcap) * softcap
        diff = (lens[:, tok_a] - lens[:, tok_b]).cpu().numpy()
        # Sanity: the lens at the top layer must reproduce the real logits.
        top_err = (lens[-1] - logits).abs().max().item()
        prob_a = torch.softmax(logits, -1)[tok_a].item()
        # float32, not float16: Gemma has a massive-activation channel (dim 443, layers 28–32) past fp16's 65504.
        return resid.float().cpu().numpy(), diff, prob_a, top_err

    n, L = len(pairs), len(layers)
    acts_off = np.zeros((n, L, lm.config.hidden_size), np.float32)
    acts_on = np.zeros_like(acts_off)
    lens_off, lens_on = np.zeros((n, L), np.float32), np.zeros((n, L), np.float32)
    conf_off, max_err = np.zeros(n, np.float32), 0.0

    for i, (_, r) in enumerate(pairs.iterrows()):
        tok_a, tok_b = first_token(tok, r["answer_off"]), first_token(tok, r["answer_on"])
        retrieved = [{"text": lessons[int(x)]} for x in str(r["retrieved_lesson_ids"]).split(";") if x.strip().isdigit()]
        acts_off[i], lens_off[i], conf_off[i], e1 = run(build_prompt(r["question"], []), tok_a, tok_b)
        acts_on[i], lens_on[i], _, e2 = run(build_prompt(r["question"], retrieved), tok_a, tok_b)
        max_err = max(max_err, e1, e2)
        if i % 50 == 0:
            print(f"[{i}/{n}] top-layer lens error so far: {max_err:.3f}", flush=True)

    for h in hooks:
        h.remove()
    out = os.path.join(args.run_dir, "activations.npz")
    np.savez_compressed(out, acts_off=acts_off, acts_on=acts_on, lens_off=lens_off, lens_on=lens_on, conf_off=conf_off)
    pairs.to_csv(os.path.join(args.run_dir, "activations_meta.csv"), index=False)
    print(f"Saved {n} pairs x {L} layers to {out}; max top-layer lens error {max_err:.3f}")


if __name__ == "__main__":
    main()
