# reflective_ai — design notes

A running log of design changes, intent, and what's next. Append as we go.

## Current state (2026-05-16)

Pivoting from "chatbot with RAG over its own chat log" to "agent that reflects on its own task failures." The original reflection loop in `scheduler.py` had no grounding signal — it reflected on its own prior reflections. New design follows Reflexion (Shinn et al., 2023) and Voyager (Wang et al., 2023): task → attempt → grade → lesson on failure → retrieve relevant lessons before the next attempt.

The broader research question: **can an LLM, given its own past failures as context, do better than it would cold?** That's the closest functional analog to "learning from mistakes" with current tech. Phenomenal feeling / consciousness is out of scope.

## The plan (6 steps)

1. **Pick a task with verifiable outcomes.** ← done (see Step 1 section below)
2. **Stop polluting memory + restructure schema.** ← done
3. **Add a grader.** ← done (see Step 3 section below)
4. **Failure-triggered reflection.** On a wrong answer, ask the model for a lesson; store it. ← v1 done — naive prompt produces vacuous "be more careful" platitudes (see Step 4 section). Tuning the prompt next.
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

### Addendum 2 (2026-05-17): moved generation off Gemini

Gemini free tier turned out to be unusable: not just 5 RPM but a **daily cap of 20 `generate_content` requests** per model (`GenerateRequestsPerDayPerProjectPerModel`). At ~2 calls/question we can't finish even one 40-question run. Throttle/retry can't help — a daily cap resets in 24h, not seconds. (Partial baseline before hitting the wall: 14/14 correct across authorship + sequencing + near_name.)

**Generation now goes through an OpenAI-compatible endpoint (`llm_client.py`), defaulting to NVIDIA NIM.** Both NVIDIA and Ollama speak the OpenAI protocol, so switching providers is purely an `.env` change — no code edits.

- **`gemini_client.py` → `llm_client.py`.** `generate()` now uses the `openai` SDK against `LLM_BASE_URL`. Returns the text string directly (callers updated: `generate(prompt).strip()`, no more `.text`). `temperature=0` for reproducibility. Retry catches `openai.RateLimitError`.
- **Embeddings stayed on Gemini.** They're low-volume (only when lessons exist) and use a *separate* quota from generation, so they don't hit the daily generate cap. `embed()` still calls Gemini, retry catches `ResourceExhausted`. `EMBED_DIM` unchanged at 768. Migrate to NVIDIA embeddings later only if Gemini's embed quota becomes a problem.
- **`EVAL_THROTTLE_SECS` default → 0.** NVIDIA's 40 RPM is plenty; retry covers the rare burst.

**Required `.env` settings** (generation provider):

```
# NVIDIA (default)
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=microsoft/phi-4-multimodal-instruct
LLM_API_KEY=nvapi-...          # from build.nvidia.com  <-- the one thing you MUST set

# --- or Ollama fallback (offline / reproducible) ---
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.1:8b
# LLM_API_KEY=ollama           # ignored by Ollama but must be non-empty

# still needed for embeddings:
GOOGLE_API_KEY=...
```

`LLM_BASE_URL` and `LLM_MODEL` now default to NVIDIA + Phi-4 in `llm_client.py`, so in practice only `LLM_API_KEY` and `GOOGLE_API_KEY` must be present in `.env`.

### Model selection (2026-05-17)

Spent several rounds picking a generation model. The decision rule, in priority order:

1. **No reasoning models.** A model that thinks before answering (MiniMax M2, DeepSeek-R1, QwQ, Seed-OSS with thinking on) confounds the experiment: chain-of-thought is itself a mechanism that fixes the reasoning/instruction traps we're testing, so we couldn't attribute any accuracy gain to *memory* vs. the model's own reasoning. The whole experiment is "does reflection-from-memory help?" — a second improvement mechanism in the room makes that unanswerable.
   - *MiniMax M2*: rejected — thinking is baked in, can't disable cleanly; vendor recommends against it.
   - *Seed-OSS 36B*: usable in principle (`thinking_budget: 0` disables thinking) but no advantage over a plain instruct model, and 36B risks acing the set.
2. **Plain instruct, `temperature=0`.** Reproducibility — we're measuring small before/after deltas; sampling noise would swamp them.
3. **Capable enough to know the facts, weak enough to make instruction/disambiguation slips.** We want failures that a *lesson* can fix (misreading "NOT", grabbing the wrong near-name option), not pure knowledge gaps (memorizing a fact doesn't generalize to its category siblings).

**Chosen: `microsoft/phi-4-multimodal-instruct`.** Non-reasoning, ~14B, knows most facts → failures should concentrate on the lesson-correctable traps. Weaker than the Gemini 2.5 Flash that went 14/14, so we should actually see failures.

**Fallback if Phi-4 aces it (>90%): `nvidia/nemotron-mini-4b-instruct`** (one `.env` line change). 4B guarantees more failures; downside is some will be knowledge gaps rather than reasoning slips.

### Signal concern to revisit

Partial Gemini run hit 14/14 before the cap. If the full Phi-4 run comes back >90% overall, the set is too easy and step 4 has too few failures to learn from. Two levers: switch to Nemotron-4B (more failures), or harden the set (trickier sequencing/negation, adversarial near-name pairs). Reflection on 2–3 failures won't show a measurable effect — we need a meaningful failure count before step 4 is worth building.

### Addendum 3 (2026-05-17): the LLM judge was unreliable — replaced with a deterministic grader

First full Phi-4 run reported **95%**. It was wrong. The LLM judge (Phi-4 grading its own answers) produced **false positives**: it marked "Apple" correct when the expected answer was "Microsoft", "electric light bulb" correct vs "telephone", and "DRC" correct vs "Algeria". Verified directly — the judge literally returns `"Yes"` to *"does 'Apple' convey the same factual content as 'Microsoft'?"*

**Root cause:** a model is not a reliable judge of ground truth it doesn't itself know. Asking a weak model "are these the same?" against facts it's shaky on yields rubber-stamping. This is a general methodology trap — don't use an LLM judge for factual correctness unless the judge is clearly stronger than the model under test, and even then, prefer deterministic checks.

**Fix:** `grader.py` is now fully deterministic, no API calls:
- Expected answer numeric → parse both sides, compare. Integers exact; decimals get ±0.05 slack (handles "42.195" vs "42.2"). This correctly fails the paper-ignition answer (model gave 451°F, question asked Celsius/233).
- Expected answer non-numeric → normalized string containment (accent-folded, punctuation-stripped). Every false positive above is fixed because the wrong answer doesn't *contain* the expected token ("Apple" ⊅ "Microsoft").

**Residual risk (noted, low):** containment can false-positive if the model gives a verbose answer that names the expected option while asserting the wrong one ("Apple, not Microsoft, came first"). The answer-only prompt constraint keeps answers short enough that this didn't occur. Revisit if a future model gets chatty.

Bonus: grading is now reproducible and free, so step 5's before/after comparison won't have judge noise in it.

## Baseline results — Phi-4, memory OFF (2026-05-17)

**Overall: 85% (34/40).** Saved to `results_baseline.csv`. This is the control condition.

| Category   | Accuracy | Notes |
|------------|----------|-------|
| authorship | 100% | |
| near_name  | 100% | |
| negation   | 100% | |
| first_to   | 100% | |
| units      | 80%  | gave 451°F when Celsius was asked |
| counting   | 80%  | "M" states: said 4 (it's 8) |
| geography  | 80%  | largest African country: said DRC (it's Algeria) |
| sequencing | **40%** | Microsoft/Apple, telephone/bulb, Eiffel/Statue all wrong |

**The 6 failures, classified by whether a lesson could transfer:**

- **Transferable (shared failure mode → the interesting cases):**
  - *sequencing* (3 failures): the model systematically misjudges temporal order. A lesson like "for founding/invention-order questions, recall the actual years instead of trusting which is more iconic" could plausibly help all 5 sequencing questions, including the 2 it currently gets right.
  - *units* (1 failure): answered in the wrong unit. A lesson "answer in the exact unit the question requests" could generalize across units questions.
- **Isolated (fact-specific → a lesson is just memorizing the answer, won't generalize):**
  - *counting* (M states), *geography* (largest African country).

**Verdict: keep Phi-4, proceed to step 4.** The failure profile is close to ideal — `sequencing` gives a concentrated, shared, plausibly-transferable failure mode, which is exactly what's needed to test "do reflection lessons generalize?" More failures from a weaker model (Nemotron-4B) would mostly be isolated knowledge gaps, which are *worse* for the experiment, not better. The real test in step 5 will be whether the reflection step (step 4) produces lessons abstract enough to transfer across sequencing questions, rather than memorizing individual founding dates.
- **LLM judge is non-deterministic.** Re-running can flip 1–2 borderline calls. For step 6's eval, run each condition 2–3 times and average — don't read into single-run deltas of <5%.

### What's next

Step 4: failure-triggered reflection. When `grade(...)` returns `correct=False`, hand the (question, model_answer, expected_answer) triple to the model and ask: *"In one sentence, what general lesson would prevent this mistake in the future?"* Store via `record_lesson` with `source_attempt_id` linking back. Do NOT enable lesson retrieval during this pass — we want to accumulate lessons from a clean baseline pass first. Step 5 turns retrieval on and re-runs.

## Step 4 — failure-triggered reflection (2026-05-17)

`reflect.py` reads failures (`correct == 0`) from `attempts.csv`, asks the model to write a transferable lesson for each, stores via `memory.record_lesson`. Retrieval stays off (that's step 5). `memory.reset_lessons()` lets us wipe + regenerate while tuning. Also moved embeddings local this step — see Addendum 4.

### v1 result: naive prompting produces vacuous lessons

Prompt asked for "one general lesson... a transferable principle, not this specific fact." All 6 lessons came back as interchangeable platitudes:

- "Always cross-reference your facts with multiple reliable sources before providing an answer."  (×4, near-verbatim)
- "Always double-check your answers... use a calculator to convert units."
- "Always double-check your answers against reliable sources."

**Why this happens (the finding, not a bug):**
1. The model never diagnoses *why* it failed — it pattern-matches "I was wrong" → generic "be careful," instead of naming the failure mode (temporal order, unit conversion, counting).
2. The advice isn't self-applicable: "cross-reference reliable sources" / "use a calculator" — it has neither at inference. It's parroting human advice that doesn't apply to a frozen model.
3. Lessons are near-identical → embed to ~the same vector → retrieval can't discriminate. Running step 5 on these would show no effect (or mild noise).

**Deeper split these failures expose** (matters for the thesis):
- *Process errors* (units: knew 451°F, didn't convert to the requested Celsius) — fixable by a good lesson.
- *Knowledge errors* (believes Apple predates Microsoft; bulb before telephone) — NOT fixable by a general lesson. If the recalled fact is wrong, "check the dates" changes nothing; the only fix is memorizing the specific answer (lookup-table trap, doesn't generalize). Sequencing may be unfixable-in-principle by reflection.

### v2 result: diagnosis-first prompt — rescued the process errors, not the knowledge errors

v2 prompt forces (a) name the specific error type, (b) give a rule usable with only the model's own reasoning (no tools/sources), (c) explicit ban on "double-check / be careful / consult sources". Regenerated lessons split cleanly into two tiers:

**Tier 1 — sharp, transferable, self-applicable (process errors):**
- *units* (#4): "Confusing Fahrenheit with Celsius → always convert to the unit the question asks for." Ideal — diagnostic, generalizes across units questions, no tools needed.
- *sequencing* (#2, #3): "Confused which came earlier → compare the actual years directly, without assuming order based on fame/general knowledge." Correct meta-strategy, a principle not a fact.

**Tier 2 — still weak (isolated-knowledge errors):**
- Microsoft/Apple (#1): right diagnosis but rule still leaked "cross-reference with a reliable source" despite the ban.
- counting (#5): "make a mnemonic / visual aid" — not executable at inference.
- geography (#6): rambling, near-incoherent.

**Conclusion:** prompt design matters a lot (v1 → v2 was a big jump), but it can't manufacture a transferable principle where none exists. Process errors yield good lessons; isolated-knowledge errors don't. This is the experiment's core result taking shape.

Lessons are now content-differentiated (years / units / counting / countries), so their embeddings differ and retrieval can discriminate — unlike v1's interchangeable platitudes. Good enough to proceed to step 5.

### Caveat for step 5/6 design — memorization vs transfer

Re-running the *same* 40 questions conflates two very different things:
- **(a)** Does the lesson fix the exact question it came from? Weak test — for a same-question re-run, the most-similar retrieved lesson IS the one generated from that question, so a "fix" is closer to memorization.
- **(b)** Does a lesson transfer to *sibling* questions it did NOT come from? Strong test — real generalization.

Step 5 (re-run same 40, memory on) measures mostly (a) plus some within-category (b). Step 6 must add **held-out questions** (new items per category, never seen, no lesson generated from them) to isolate (b). Without held-out, a big step-5 jump could just be the lookup-table trap. Plan held-out questions before declaring success.

### Addendum 4 (2026-05-17): embeddings moved local

`models/embedding-001` got 404'd — Google deprecated it (second Gemini model to die mid-project, after `gemini-2.0-flash`). Rather than chase another Gemini embedding ID that may also vanish, embeddings are now a **local sentence-transformers model** (`all-MiniLM-L6-v2`, 384-dim, CPU). No API, no rate limits, no deprecation, fully reproducible. `EMBED_DIM` changed 768 → 384 (free — lessons were empty). This removes Gemini from the project entirely; `GEMINI_API_KEY` in `.env` is now unused.
