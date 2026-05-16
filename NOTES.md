# reflective_ai — design notes

A running log of design changes, intent, and what's next. Append as we go.

## Current state (2026-05-16)

Pivoting from "chatbot with RAG over its own chat log" to "agent that reflects on its own task failures." The original reflection loop in `scheduler.py` had no grounding signal — it reflected on its own prior reflections. New design follows Reflexion (Shinn et al., 2023) and Voyager (Wang et al., 2023): task → attempt → grade → lesson on failure → retrieve relevant lessons before the next attempt.

The broader research question: **can an LLM, given its own past failures as context, do better than it would cold?** That's the closest functional analog to "learning from mistakes" with current tech. Phenomenal feeling / consciousness is out of scope.

## The plan (6 steps)

1. **Pick a task with verifiable outcomes.** ← done (see Step 1 section below)
2. **Stop polluting memory + restructure schema.** ← done
3. **Add a grader.** ← done (see Step 3 section below)
4. **Failure-triggered reflection.** On a wrong answer, ask the model "what general lesson could prevent this next time" and store it as a lesson.
5. **Lesson-aware retrieval at attempt time.** Inject relevant past lessons into the prompt (already wired — just empty until step 4 starts producing lessons).
6. **Evaluate.** Accuracy with vs without memory, on seen vs unseen questions. Spot-read 20 lessons for quality.

---

## Step 1 — Task selection (2026-05-17)

### Why this matters more than it looks

A random trivia set (TriviaQA, HotpotQA, etc.) is wrong for this experiment. If question N fails because of failure mode X, but question M is about something totally unrelated, then a lesson extracted from N's failure ("watch out for X") has nowhere to apply on M. The reflection loop would generate lessons that never get retrieved usefully — and we'd conclude "reflection doesn't help" when really we just chose a non-transferable task.

The fix: group questions by **shared failure mode**. If the agent fails Q1 in a category and writes a lesson, that lesson should plausibly help on Q2–Q5 in the same category. Different categories should NOT share lessons. This gives us a clean signal: improvement within category = reflection working; cross-category bleed = lessons too generic.

### The 40-question set (`questions.csv`)

8 categories × 5 questions. Schema: `question, expected_answer, category`.

| Category | Failure mode it targets |
|----------|------------------------|
| `authorship` | Confusing original creator with adapter/franchise owner |
| `sequencing` | Temporal ordering (which came first) — LLMs are weak at this |
| `near_name` | Two similarly-known people/things; picking the wrong one |
| `units` | Unit context (Celsius vs Fahrenheit, km vs miles) |
| `first_to` | "First X to Y" — specificity of the qualifier matters |
| `negation` | "Which of these is NOT…" — LLMs over-attend to the listed items |
| `counting` | Numeric facts with no easy interpolation |
| `geography` | Capital/largest/longest — well-known confusions (Brazil → Rio not Brasília) |

### What this set is deliberately NOT

- Not random trivia. Each question has 4 siblings sharing a failure mode.
- Not adversarial. The questions are answerable by a strong LLM most of the time. We want *some* failures, not most — too many failures and there's no baseline to improve from.
- Not enormous. 40 is enough to see category-level patterns, small enough to inspect by hand and iterate.

### Known gotchas

- **`units` and `counting` answers are numeric.** The string-match grader will fail on format variation ("100" vs "100°C" vs "100 degrees Celsius"). Plan: when the grader lands in step 3, use an LLM-judge fallback for these categories, or instruct the prompt to reply with bare numerals only.
- **Q27 (RGB primary).** "Yellow" is correct for additive RGB but a strong model might argue from subtractive/RYB. Acceptable — if it gets this wrong, the lesson should be "match the framework named in the question."
- **Geography Q37 has a non-ASCII character** (Brasília). The grader needs to be accent-insensitive.

### What's next

Step 3: grader. Need a `grade(question, answer, expected) -> (correct: bool, feedback: str)` that:
- Strips/lowercases/accent-folds for substring match
- Falls back to LLM-judge for numeric / fuzzy answers
- Updates the corresponding row in `attempts.csv` with `correct` and `expected_answer`
Plus a small driver script that loops over `questions.csv`, calls `agent.answer`, calls the grader, and prints accuracy by category.

---

## Step 2 — Memory schema + prompt fix (2026-05-16)

### Why

The old `memories.csv` mixed three things in one table (user / agent / reflection) keyed only by a `type` column. Two problems:

- The agent's verbose meta-narration (e.g. `**My Past Responses:** ...`) got stored as agent memory and later retrieved into prompts. The model was learning from its own scratchpad.
- "Reflection" rows were free-text blobs not linked to anything. There was nothing to ground reflection against, so reflections were always vague self-narration ("I should be more helpful").

For reflection-on-mistakes to work, the store needs to distinguish **attempts** (what happened, with grading info) from **lessons** (what was learned). Different access patterns: attempts are scored numerically; lessons are retrieved by similarity.

### What changed

- **`memories.csv` → `memories_archive.csv`.** Kept for archeology; not read by new code.
- **New `attempts.csv`.** Columns: `id, timestamp, question, answer, correct, expected_answer, retrieved_lesson_ids, notes`. `correct` and `expected_answer` start empty — the grader (step 3) fills them in.
- **New `lessons.csv`.** Columns: `id, timestamp, text, source_attempt_id`. Empty until step 4 produces lessons.
- **New `lessons.faiss`.** FAISS index persisted to disk. The old code re-embedded the entire CSV on every startup (slow + paid API calls). Now the index is loaded from disk; an alignment check rebuilds it only if it's out of sync with `lessons.csv`.
- **`memory.py`** rewritten around the new schema. Public API: `record_attempt`, `record_lesson`, `retrieve_lessons`. Old `store_memory` / `retrieve_memories` removed.
- **`agent.py`** rewritten. Entry point: `answer(question)`. Prompt explicitly constrains the model to reply with only the answer (no preamble, no reasoning, no meta-narration). Retrieves lessons (currently always empty) and records the attempt.
- **`main.py`** — route renamed `/chat` → `/ask` to match the new framing.
- **`scheduler.py`** deleted. Step 4 will reintroduce reflection on a different trigger (failure, not a 5-turn timer) — the old version doesn't survive.

### Files touched

- `NOTES.md` (new)
- `memory.py` (rewrite)
- `agent.py` (rewrite)
- `main.py` (small)
- `scheduler.py` (deleted)
- `memories.csv` → `memories_archive.csv` (renamed)

### Sanity check

After this step you should be able to:

1. Start the server: `uvicorn main:app --reload`
2. Hit it: `curl "http://localhost:8000/ask/?question=what is the capital of france"`
3. See a new row in `attempts.csv` with empty `correct` / `expected_answer` / `retrieved_lesson_ids`
4. Confirm `lessons.csv` is empty and `lessons.faiss` exists (created on first import of `memory`)
5. Crucially: the `answer` column should be **just the answer** (e.g. "Paris"), not a multi-paragraph monologue

If step 5 fails — model still generates preamble — tighten the prompt in `agent.py:build_prompt`.

### Known gotcha for step 3

`attempts.csv`'s `correct` column will round-trip as the strings `"True"` / `"False"` through pandas, not booleans. When the grader lands, either store as `1` / `0` from the start or parse explicitly on read. Easy to fix; flagging it now so future-you doesn't get a silent bug.

### What's next

Step 3: write the grader. Needs a question set first — pick or create one (30–50 trivia Q&A pairs is plenty). Once that exists, the grader is ~10 lines: string-match for trivia, LLM-judge for fuzzier answers.

---

## Step 3 — Grader + evaluation driver (2026-05-17)

### Why two-tier grading

The grader has to survive three kinds of model output:

1. **Clean direct answer** ("Paris") — normalized substring match works fine.
2. **Format variance** ("100°C" vs "100" vs "one hundred degrees") — substring match catches the easy case; LLM judge catches the rest.
3. **Verbose wrong answers that echo the expected token** — e.g. "Microsoft, founded in 1975, predates Apple by one year." For sequencing/near-name/negation categories, the expected answer is *literally inside the question* as one of the offered options. A naive substring match would mark "the answer contains the word 'Apple'" as correct.

The rule used by `grader.grade`:

> If the expected answer appears in the question text (= it's one of the multiple options being asked about), skip string-match and go straight to LLM judge. Otherwise, try string-match first; on miss, fall back to LLM judge.

This keeps grading deterministic on ~half the questions (authorship, units, first_to, counting, geography) and only uses the non-deterministic judge where it actually needs to (sequencing, near_name, negation, plus any format-mismatch fallthrough).

### What changed

- **`grader.py` (new)** — `grade(question, answer, expected) -> (correct, note)`. The `note` is either `"string_match"` or `"llm_judge"` and gets written into the attempt row's `notes` column, so you can audit which grading path was used for each item.
- **`evaluate.py` (new)** — the driver. Loops `questions.csv`, calls `agent.answer`, grades, records the outcome, prints per-category accuracy at the end. Supports `--limit N` for smoke tests.
- **`memory.py`** — added `record_attempt_outcome(attempt_id, correct, expected_answer, notes=None)`. Critical detail: `correct` is stored as int `1`/`0`, **not** Python bool. This resolves the gotcha flagged in step 2 (booleans round-trip as strings `"True"`/`"False"` through CSV and break truthiness checks on reload).
- **`agent.py`** — `answer(question)` now returns `(reply, attempt_id)` so the eval driver can write the outcome back to the correct row.
- **`main.py`** — unpacks the tuple; the HTTP route still returns just the reply.

### Files touched
- `grader.py` (new)
- `evaluate.py` (new)
- `memory.py` (added function)
- `agent.py` (signature change: returns tuple)
- `main.py` (unpack)

### How to run

```powershell
.\venv\Scripts\Activate.ps1
python evaluate.py --limit 5     # smoke test, ~5 questions
python evaluate.py               # full 40-question run
```

After a full run, inspect `attempts.csv`:
- `correct` is `1` or `0` for every row
- `notes` shows which grader path (`string_match` or `llm_judge`) graded it
- Sort by `category` and eyeball the failures — those are the seeds for step 4's lessons

### What to read into the baseline

This run is the **memory-off baseline**: `lessons.csv` is still empty, the agent has nothing to retrieve, the lesson block in the prompt is absent. Whatever accuracy you see is the model's raw capability without reflection.

Expected pattern (informed guess, will be wrong in interesting ways):
- High on `authorship`, `geography`, `first_to`
- Mid on `units`, `counting` (numeric format may cost some)
- Lower on `sequencing`, `near_name`, `negation` — the categories the set deliberately targets

**Write the per-category numbers down before step 4.** They're the control condition for the whole experiment. Without them, "did reflection help?" isn't answerable.

### Known limits

- **Model name is now env-driven.** All Gemini calls go through `gemini_client.py`, which reads `GEMINI_MODEL` from env (default `gemini-2.5-flash`). Reason: `gemini-2.0-flash` was moved off Google's free tier (`limit: 0` quota errors). When the current default ages out too, swap via `.env` instead of editing source. Try `gemini-flash-latest` as an alias if specific versions stop working.

### Addendum (2026-05-17): rate limits + a ground-truth bug

**Two issues surfaced on the first attempted full run.**

**(a) Test-set bug — caught by the agent.**
Sequencing question 1 had expected answer `"Apple"` for *"Which was founded first, Microsoft or Apple?"* Microsoft (1975) is actually older than Apple (1976). The model answered "Microsoft" — correctly — and the grader marked it wrong against bad ground truth.

This is the kind of failure that matters disproportionately in a reflection-based system: if it had slipped past, step 4 would have extracted a lesson ("Apple was founded before Microsoft") and step 5 would have *poisoned* retrieval on future related questions. Bad test data makes the agent actively worse over time. Spot-check expected answers; don't trust your own ground truth without verification. The other 39 questions were audited and look correct.

Fix: `questions.csv` row patched to `"Microsoft"`.

**(b) Free-tier `gemini-2.5-flash` is 5 RPM.**
At ~1–2 API calls per question (answer + LLM-judge when needed), the eval triggers the cap by question 6–7. Two changes:

- **`gemini_client.py` (new).** Single wrapper around `genai.generate_content` and `genai.embed_content`. Catches `ResourceExhausted`, sleeps 30s / 60s / 90s on successive failures, then surfaces the error if still blocked. `agent.py`, `grader.py`, `memory.py` all route through it now — no more duplicated `genai.configure` / model-string boilerplate.
- **`evaluate.py` throttles between questions.** `EVAL_THROTTLE_SECS` env var, default 20s. With ~2 calls/question this keeps us comfortably under 5 RPM. Set to `0` if you upgrade to paid tier.
- **`evaluate.py --reset` flag.** Wipes `attempts.csv` before running. Needed for clean baseline runs — without it, repeated runs append duplicates and confuse per-category stats.

### Estimated full-run time on free tier

- 40 questions × 20s throttle ≈ 13 min
- Plus retry sleeps if a rate-limit miss happens anyway
- If this is too slow once step 4 doubles the API call count, swap to `gemini-2.5-flash-lite` (typically higher RPM on free tier) or enable billing on the GCP project.
- **LLM judge is non-deterministic.** Re-running can flip 1–2 borderline calls. For step 6's eval, run each condition 2–3 times and average — don't read into single-run deltas of <5%.

### What's next

Step 4: failure-triggered reflection. When `grade(...)` returns `correct=False`, hand the (question, model_answer, expected_answer) triple to the model and ask: *"In one sentence, what general lesson would prevent this mistake in the future?"* Store via `record_lesson` with `source_attempt_id` linking back. Do NOT enable lesson retrieval during this pass — we want to accumulate lessons from a clean baseline pass first. Step 5 turns retrieval on and re-runs.
