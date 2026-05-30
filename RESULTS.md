# RESULTS — reflective_ai

A controlled small-N experiment testing whether a frozen LLM can improve on novel questions by applying lessons extracted from its own past mistakes — without any fine-tuning. Reflexion-style memory, deterministic grader, hand-designed training/held-out split.

## Question

Can a frozen LLM, given a notebook of lessons extracted from its own past failures, do better at the moment of answering a *new* question? And does any improvement come from genuine **transfer** of a generalizable principle, or from **memorization** of specific facts?

## Setup

- **Model:** `meta/llama-3.3-70b-instruct` via NVIDIA NIM, `temperature=0` (deterministic — the same input produces the same answer across runs).
- **Embeddings:** `all-MiniLM-L6-v2`, local, 384-dim (no API).
- **Grader:** deterministic. Numeric comparison for numeric answers (±0.05 for decimals, exact for integers); accent-folded normalized string containment for entity answers. **No LLM judge** — the LLM judge we initially built rubber-stamped wrong answers as correct (see NOTES addendum 3) and was replaced.
- **Pipeline:** `agent.py` asks → `grader.py` scores → `reflect.py` writes a one-sentence diagnosis-first lesson for each failure → lessons embedded into FAISS → at attempt time, the top-3 most similar lessons are injected into the prompt.
- **Task design:** 40 training questions + 24 held-out questions, both grouped into 8 categories chosen to target LLM failure modes (sequencing, near-name disambiguation, units, negation, etc.). Each held-out question is **factually distinct from its 5 training siblings but shares the category's failure mode.** This is the load-bearing design decision: if lessons transfer, they should transfer within a category.

## Results

| Set | Memory OFF | Memory ON | Δ |
|---|---|---|---|
| **Training (40 Q)** | 92.5% (37/40) | 97.5% (39/40) | **+2** |
| **Held-out (24 Q)** | 100% (24/24) | 95.8% (23/24) | **−1** |
| Combined (64 Q) | 95.3% | 96.9% | +1 net |

Data: `lessons.csv` (the 3 Llama-generated lessons); `results_holdout_baseline.csv` / `results_holdout_memory.csv` (held-out runs). The Llama seen-40 attempt files were overwritten by the held-out runs — numbers above are taken from logged stdout. Re-run `python evaluate.py --reset` and `python evaluate.py --reset` after a `reflect.py` pass to repopulate them.

## Findings

The +2 on the training set decomposes into three distinct cases, each illuminating a different limit of reflection.

**1. Real lesson-driven fix (telephone / electric light bulb).** Llama's lesson: *"create a mental timeline of major events and their corresponding years to ensure the correct order of occurrence."* On re-run, Llama applied it, recalled telephone 1876 / bulb 1879 *correctly*, and answered correctly. This is reflection working as designed — on a question whose underlying knowledge was already correct, the lesson just gave it a reasoning hook.

**2. Memorization fix (paper ignites).** The lesson generated from this question was retrieved for this exact question. The "improvement" is closer to a lookup table than learning. The contamination caveat flagged from the start, now visible in the data.

**3. Lesson + wrong recall = still wrong (Microsoft / Apple).** The lesson here was sharp: *"recall the founding year of each company and compare directly, not by which feels older."* Llama followed the instruction — explicitly recalled years — and **recalled wrong**, producing more confident wrong answers than baseline. **A well-formed reflective lesson cannot fix a knowledge error.** The lesson tells the model *how to think*; it does not fix *what it knows*.

The held-out adds two more findings:

**4. Failure modes don't cluster categorically in a strong model.** Llama got 24/24 with memory off — *none* of the held-out questions tripped the failure modes their categories were designed around. The training-set failures (MS/Apple, telephone/bulb, paper-ignite precision) aren't category-level reasoning weaknesses; they're **idiosyncratic factual blind spots**. So lessons keyed to "the category's failure mode" have little transferable surface to attach to.

**5. Memory can actively hurt good answers (Hoover Dam vs Empire State Building).** Memory off: Empire State (correct — finished 1931, vs Hoover Dam 1936). Memory on: Hoover Dam (wrong). Same mechanism as finding 3, inverted in sign: the retrieved "compare years" lesson prompted Llama to recall years for both, **its recall was wrong**, and structured reasoning over wrong recall produced a wrong answer where gut intuition produced a right one. **The lesson made the model worse by making it think harder with bad facts.**

## What this means

The +2/+1-net "improvement" headline is misleading on this evidence. Honestly read:

- Memory had **no net positive effect on novel (held-out) questions**, and actively degraded one.
- The training-set gain is half memorization (paper ignites) and half real-but-narrow (telephone/bulb).
- The cleanest finding is the **failure mechanism**: a process-shaped lesson can only help when the underlying knowledge it operates over is correct. When recall is wrong, the same lesson actively hurts — by reaching for more explicit reasoning over the wrong facts.

This is a **limit result** for Reflexion-style reflection on a strong frozen instruct model. It does not say reflection is useless — it says the conditions under which it helps are narrower than the framing assumes, and the same mechanism that helps can hurt.

For the broader research framing: "can an LLM improve by applying lessons from past mistakes without fine-tuning" gets a qualified yes — *sometimes, when failures have a transferable shape and underlying knowledge is correct.* Most of the time, on a capable model, it doesn't move the needle, and occasionally it makes things worse.

## Caveats

- **n = 64 total questions, one model, one reflection prompt, single seed (temp 0).** Suggestive, not statistical.
- **One model.** A weaker model would fail more on the held-out (giving real signal to measure); a stronger model would fail less. Conclusions are conditional on Llama 3.3 70B specifically.
- **Question-design assumption challenged by the result.** The held-out assumed failures cluster by category, which finding 4 partially contradicts. A better design would pre-screen candidate held-out questions against the target model and keep only those it fails — focusing measurement where it can actually move.
- **One prompt iteration.** v1 of the reflection prompt produced platitudes ("double-check your facts"); v2 (diagnosis-first, ban-on-platitudes) produced the lessons used here. Better prompt designs may produce sharper lessons.
- **Same-question contamination on training.** Each training question could retrieve the lesson generated from itself. The held-out doesn't have this issue, which is why the held-out number is the more meaningful one.

## Files

- `agent.py`, `grader.py`, `memory.py`, `llm_client.py`, `evaluate.py`, `reflect.py` — pipeline
- `questions.csv` (40 training), `questions_holdout.csv` (24 held-out)
- `lessons.csv` — the 3 Llama-generated lessons
- `results_holdout_baseline.csv`, `results_holdout_memory.csv` — held-out results, both conditions
- `NOTES.md` — full working log including dead ends, model swaps, design decisions
