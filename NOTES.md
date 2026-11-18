# reflective_ai — design notes

A running log of design changes, intent, and what's next. Append as we go.

## Current state (2026-05-16)

Pivoting from "chatbot with RAG over its own chat log" to "agent that reflects on its own task failures." The original reflection loop in `scheduler.py` had no grounding signal — it reflected on its own prior reflections. New design follows Reflexion (Shinn et al., 2023) and Voyager (Wang et al., 2023): task → attempt → grade → lesson on failure → retrieve relevant lessons before the next attempt.

The broader research question: **can an LLM, given its own past failures as context, do better than it would cold?** That's the closest functional analog to "learning from mistakes" with current tech. Phenomenal feeling / consciousness is out of scope.

## The plan (6 steps)

1. **Pick a task with verifiable outcomes.** ← done (see Step 1 section below)
2. **Stop polluting memory + restructure schema.** ← done
3. **Add a grader.** ← done (see Step 3 section below)
4. **Failure-triggered reflection.** On a wrong answer, ask the model for a lesson; store it. ← done. v1 prompt produced platitudes; v2 (diagnosis-first, ban-on-platitudes) produced usable lessons on Llama 70B.
5. **Lesson-aware retrieval at attempt time.** ← done. Llama seen-40: 92.5% (memory off) → 97.5% (memory on). Decomposes into 1 real fix, 1 memorization, 1 stubborn wrong (the "lesson can't fix wrong knowledge" case — MS/Apple).
6. **Evaluate on held-out.** ← done. Llama 24-Q held-out: 100% (memory off) → 95.8% (memory on). Memory broke one previously-correct answer (Hoover/Empire State). See RESULTS.md for the full writeup.

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

## Step 5 — memory-aware retrieval on Llama (2026-05-24)

Wiped phi-4 lessons, switched default model to `meta/llama-3.3-70b-instruct` (phi-4-multimodal got "DEGRADED" twice on NVIDIA in one session). Re-ran baseline and reflection on Llama, then memory-on eval on the same training set.

**Llama 3.3 70B baseline (memory off):** 92.5% (37/40). Stronger than Phi-4 (85%) — fixed the M-states counting error, the DRC/Algeria geography error, and the Eiffel/Statue sequencing error. The 3 remaining failures:
- sequencing: Microsoft/Apple → "Apple" (wrong)
- sequencing: telephone/bulb → "electric light bulb" (wrong)
- units: paper ignites → "233.9" (close to 233 but our integer grader rounds to 234)

**Llama reflection lessons (v2 prompt):**
1. MS/Apple → "When comparing the founding dates of two companies, always recall or deduce the founding year of each company and compare the years directly, rather than relying on general knowledge or assumptions about which company is older."
2. telephone/bulb → "When comparing the timeline of inventions, create a mental or written timeline of major events and their corresponding years to ensure the correct order of occurrence."
3. paper ignite → "When providing a numerical value in response to a question that references a specific, well-known value (such as the title of a book), round the answer to the nearest whole number..."

Notably sharper than phi-4's lessons (which leaked "consult reliable sources" platitudes even with v2). Llama follows the diagnosis-first format.

**Llama memory-on (training set):** 97.5% (39/40). Two fixes:
- telephone/bulb FLIPPED to correct — *real lesson-driven fix*. Llama's year-recall is correct (1876/1879); lesson gave it the strategy.
- paper ignite FLIPPED to "233" — *memorization fix*. Lesson came from this exact question; retrieved for this exact question. Headline +1 but lookup-table-shaped.
- MS/Apple STILL wrong. Llama followed the lesson (recall years, compare), but its recall is wrong (believes Apple is older than Microsoft). **A sharp meta-strategy lesson cannot fix a knowledge error.** The most important data point of the experiment.

## Step 6 — held-out (2026-05-24)

Drafted `questions_holdout.csv` (24 Q, 3 per category) — each held-out question factually distinct from its training siblings but sharing the category's failure mode. Added `--questions` and `--no-memory` flags to `evaluate.py` plus `use_memory` plumbing in `agent.py` so we can run any question set in either condition without wiping lessons.

**Held-out memory OFF: 100% (24/24).** Llama got every held-out question right. None of the questions designed to target the failure modes (sequencing, units, etc.) actually tripped Llama. **Implication: failures don't cluster by category-level reasoning weakness in a strong model — they're idiosyncratic factual blind spots.** The MS/Apple miss isn't a sequencing-failure-mode; it's a specific belief.

**Held-out memory ON: 95.8% (23/24).** One previously-correct answer FLIPPED to wrong: Hoover Dam vs Empire State Building. Memory-off said Empire State (correct, 1931 < Hoover 1936); memory-on said Hoover Dam.

The mechanism is the mirror of MS/Apple: the retrieved "compare actual years" lesson prompted Llama to recall years for both, but its recall was wrong on this pair, and structured reasoning over wrong recall produced a wrong answer where gut intuition produced a right one. **Lesson + wrong recall can hurt as well as fail to help.**

See `RESULTS.md` for the consolidated writeup.

## Phase 2 setup — local Gemma 3 4B for harm measurement + interpretability (2026-11-15)

Plan: (1) re-run the pipeline on a weaker local model and count **harms** (correct → wrong under memory) alongside fixes; (2) inspect activations on the fix/harm pairs (logit lens, patching, then a probe that predicts harm before the lesson is applied). Weight-space edits (MEMIT etc.) parked until after.

- **Why local:** the NVIDIA API returns text only — no activations. Interpretability needs the model in-process.
- **Model: `google/gemma-3-4b-it`**, already in the HF cache. Chosen over Ollama's `gemma4:e2b` because Ollama can't expose activations, and the harm cases must come from the same model we inspect. Gemma Scope SAEs should exist for Gemma 3 (verify before relying on them).
- **8-bit (bitsandbytes):** bf16 weights are ~8.6GB; the RTX 4060 has 8GB. 8-bit uses 5.1GB. Activations shift slightly vs full precision — fine for logit lens/probes, a small mismatch for off-the-shelf SAEs.
- **`llm_client.py`:** `LLM_BACKEND=local` switches `generate()` to the in-process model (greedy decoding, equivalent to temp 0). `load_local()` is public so interpretability code reuses the same instance. API path unchanged and still the default.
- **venv:** torch swapped to the CUDA build (`2.12.0+cu130`), plus `bitsandbytes`, `accelerate`.
- **Smoke test:** loads in ~26s, ~3–4s per answer. Already got telephone vs bulb wrong ("The electric light bulb") — good sign this model will fail enough to measure.

## Phase 2, run 1 — Gemma 3 4B fixed/broke measurement (2026-11-16)

`python run_cycle.py --name gemma3-4b --backend local` → everything in `runs/gemma3-4b/` (log in `runs/gemma3-4b.log`). New tools: `evaluate.py --out` (snapshot each run), `compare.py` (paired off→on outcomes + harm rate), `run_cycle.py` (full cycle into its own dir, lessons kept out of the repo root).

**Grader bug found and fixed:** `_parse_number` only read digits, so Gemma's "Nine" (justices) and "Six" (Apollo landings) were graded wrong. Added number words zero–twenty. Re-graded all saved files: Phi/Llama results unchanged (those models answered in digits); Gemma held-out off 19 → 21, on 19 → 20. **The bug was hiding the one harm case** — before the fix the held-out looked like a flat 79.2% → 79.2%.

| Set | Memory OFF | Memory ON | Fixed | Broke | Harm rate |
|---|---|---|---|---|---|
| Training (40) | 85.0% (34) | 87.5% (35) | 1 | 0 | 0% |
| Held-out (24) | 87.5% (21) | 83.3% (20) | 0 | 1 | 4.8% |

- **The only training fix is memorization.** RGB "NOT a primary color": lesson 5 was written from this exact question and retrieved for it.
- **The held-out harm is a different mechanism from Llama's.** Apollo landings: off "Six" (correct) → on "11". Retrieved lessons 4;2;1 — lesson 4 is the *Moon-distance* lesson ("convert to a more familiar scale before calculating"), pulled in by surface similarity ("Moon"), not relevance. Llama's harm (Hoover Dam) was a relevant lesson exercising wrong recall; this one is an **irrelevant lesson retrieved on a keyword** that derailed a correct answer. "11" is plausibly "Apollo 11" leaking in as the salient number — a concrete target for the interpretability step.
- **Lesson quality is weak.** 3/6 lessons are near-platitudes ("mentally verify the order of magnitude", "group into smaller chunks"); lesson 4 is self-contradictory ("convert to kilometers … even if already in kilometers").
- **N is too small for probes.** 1 fix + 1 harm across 64 questions. The interpretability step needs dozens of each → the question set has to grow substantially (hundreds) before training a harm probe.

## Phase 2, run 2 — Gemma 3 4B on 500 TriviaQA questions (2026-11-16)

64 questions gave 1 fix + 1 harm — too few. `build_triviaqa.py` samples 250 train + 250 held-out (seed 0) from TriviaQA rc.nocontext validation; `expected_answer` = aliases joined by "|". Grader now accepts any alias and matches whole words (re-graded all earlier files: no changes). Run: `runs/gemma3-4b-trivia/` (logs `.log` + `.resume.log` — first attempt was killed for low system RAM after step 2; resumed with `run_cycle.py --from-step 3`).

| Set | Memory OFF | Memory ON | Fixed | Broke | Harm rate |
|---|---|---|---|---|---|
| Training (250) | 48.4% (121) | 44.8% (112) | 9 | 18 | 14.9% |
| Held-out (250) | 52.8% (132) | 48.8% (122) | 11 | 21 | 15.9% |

129 lessons written (one per training failure).

- **Memory is net harmful at scale.** It breaks ~2× as many answers as it fixes, on both splits. Per split the sign test is p≈0.11–0.12; pooled (39 broke vs 20 fixed) p=0.018.
- **Even training doesn't benefit,** though every failed training question has its own lesson in the store. Only 9/129 fixed. Lessons are rules, not facts, so a lesson from a question can't supply that question's missing answer — the Llama finding again, now at N=250.
- **Harm isn't copying.** 0/39 broken answers appear verbatim in the retrieved lessons. The broken answers swap to a *different plausible entity* (Maastricht → Lisbon, Wellington → Buenos Aires, Chess → Go, Tin → Indium, George III → George IV).
- **Mechanism (hypothesis): topic-matched heuristics override correct recall.** Maastricht→Lisbon and Wellington→Buenos Aires both retrieved lessons 64;43;7 — geography heuristics like "prioritize the most immediately obvious and widely known neighboring country". Retrieval matches on topic, not on whether the question has the failure the lesson was written for, and the heuristic then steers the model off an answer it had right. Same family as the 64-question Apollo case.
- **Grader noise exists but doesn't flip the sign.** Spot check of the 59 changed pairs: a few are alias artifacts (e.g. "DLS (Duckworth-Lewis-Stern)" → "The DLS method" counted as broke; "Sir Walter Scott" → "Scott"; Gregorovitch "fixed" via a loose Harry Potter alias). Roughly 4–6 per side; the broke > fixed gap survives.
- **Interpretability now has data:** 39 broke / 20 fixed / 214 stayed-right / 227 stayed-wrong pairs in `*_pairs.csv`. Enough for logit lens + patching on the broke cases and a cross-validated broke-vs-stayed_right probe; thin for anything fancier.

## Phase 2, interpretability — harm probe + logit lens (2026-11-16)

`interp_capture.py` → `runs/gemma3-4b-trivia/activations.npz` (500 pairs × 34 layers, residual stream at the answer position, memory-off and memory-on prompts) + per-layer logit lens. `interp_analyze.py` → `analysis.txt`, `probe_auc_by_layer.csv`.

- **Capture gotchas.** (a) First capture stored float16 → 90 infs: Gemma has a massive-activation channel (dim 443, layers 28–32, values > 65504). Re-captured in float32. (b) The pooled out-of-fold AUC understated cross-validated scores (different fold models scale differently); switched to per-fold AUC. (c) Logit lens checked: top-layer lens reproduces the real logits exactly (error 0.000).

**Harm probe** (253 questions right with memory off, 39 broken by memory; features = memory-OFF residual stream, i.e. before any lesson is seen):

| Predictor | AUC |
|---|---|
| Model's own confidence (prob of its answer's first token), no fitting | **0.811** |
| Lesson similarity (max cosine question↔retrieved lessons) | 0.556 |
| Linear probe, best layer (22), 5-fold × 3 CV | 0.785 |
| Permutation null for best-layer AUC (20 shuffles) | mean 0.624, max 0.688 |
| Probe L22 + confidence + similarity | 0.787 |

- **The activations do encode "memory will break this"** — the probe clears the permutation null comfortably, and the signal builds with depth (≈0.55–0.65 through layer 16, ≈0.72–0.79 from layer 18 on).
- **But it's the confidence signal.** The probe doesn't beat the model's own output probability, and adding it on top of confidence gives nothing. At this N, harm prediction doesn't need interpretability.
- **Topic-match strength doesn't predict harm** (similarity AUC 0.556 ≈ chance). The "topic-matched heuristic" story survives qualitatively (broken answers swap to related entities) but *how* similar the lesson is doesn't matter.
- **Broken answers are the model's shaky knowledge:** median confidence broke 0.92 vs stayed-right 1.00.

**Logit lens** (mean logit(OFF answer) − logit(ON answer) at the answer position):
- **Broke (n=34):** with and without the lesson the curves are identical through layer ~18; the gap opens at 21 (+0.26), grows through 27 (+1.46), 30 (+2.79), and is +9.5 at the output. Without the lesson the correct answer only takes the lead between layers 27 and 30 — **the right answer forms late, and the lesson acts in those same late layers** where the answer is being assembled, not in early reading of the question.
- **Fixed (n=13):** same late-layer timing, opposite sign. Lessons act at the same depth whether they help or hurt.
- Caveats: first answer token only; 8-bit model; small n for fixed.

**Confidence gate, simulated** from the existing off/on answers (use lessons only if memory-off confidence < τ; τ picked on train, applied to held-out): **the best τ on train is "never use lessons"** — 48.4% train / 52.8% held-out, identical to memory off. No threshold recovers the fixes without the harms, because fixed cases are about as confident as broken ones (median 0.876 vs 0.92). Confidence separates *broke vs stayed-right* well, but not *fixed vs broke*.

**Bottom line for this phase:** on factual QA, Reflexion-style lessons are net harmful (~2:1 broke:fixed); harm lands on answers the model holds with less than full confidence; it's applied in the late layers where the answer forms; and no confidence gate makes memory net-positive. The lessons themselves have no recoverable value here — the fix has to change *what* is stored (facts / two-track), not *when* it's retrieved.

## Phase 3 setup — two-track memory (2026-11-17)

Design + predictions pre-registered in RESULTS.md ("Phase 3 design") before any run.

- **Code.** `memory.py` gains a correction track (`record_correction` / `retrieve_corrections`, own CSV + FAISS, indexed by the *question*). `agent.py` takes `memory_mode` = lessons | corrections | both; with no corrections the prompt is byte-identical to phase 2 (checked against `git show HEAD:agent.py`). `reflect.py` records a correction per failure (no model call); `--corrections-only` keeps existing lessons. `evaluate.py --memory-mode`. Runner: `run_twotrack.py` (skips any eval whose output exists → resumable).
- **Phase 2 lessons saw the whole alias list.** `reflect.py` passed `expected_answer` straight into the reflection prompt, which for TriviaQA is "Gorbacev|Gorbachav|…". Now uses the canonical answer. Phase 3 reuses the phase 2 lessons unchanged so the lessons arm stays identical; noted as a phase 2 caveat.
- **Canonical answer first.** `build_triviaqa.py` now puts TriviaQA's `value` first, then sorted aliases. Rebuilt files: same questions, same alias sets (verified), only order changed.
- **Determinism check.** Re-ran 40 memory-off training questions in a fresh process: 40/40 answers identical to phase 2. So 8-bit greedy is reproducible here and phase 2's off/lessons runs are reused, not re-run.
- **Paraphrase set** (`build_paraphrases.py` → `questions_trivia_paraphrase.csv`, 236 of 250). Model availability on NVIDIA, 2026-11-17:
  - `meta/llama-3.3-70b-instruct`: **410 Gone — end of life 2026-08-26.** Phase 1 can't be re-run on this endpoint any more.
  - `mistralai/mistral-large-4-1t-a52b`: empty content, then timeouts.
  - `nvidia/nemotron-3-ultra-550b-a55b`: works but added answer hints (quoted "Arma virumque cano" into the Virgil question).
  - **`nvidia/nemotron-3.5-super-vl-120b-a12b`: used.** Fast, clean, different family from Gemma.
  - 14 rejected after 3 tries because the paraphrase contained an accepted alias — mostly questions that name the answer themselves ("Olea europaea" → olive). Median string similarity to source 0.60; 3 near-copies (> 0.9).
- **Git:** `activations.npz` (165MB) blocked the push (GitHub's 100MB limit). Removed from the two unpushed commits with plumbing (backup branch `backup/before-npz-removal`), `*.npz` added to `.gitignore`. Regenerate with `interp_capture.py`.

## Phase 3 run — two-track results (2026-11-17)

`run_twotrack.py --source runs/gemma3-4b-trivia --name gemma3-4b-twotrack`. The first launch was killed for low memory during train/corrections, after the 129 corrections were built. A re-run of the same command resumed and finished the remaining 8 evals (about 1h40 total). Log: `runs/gemma3-4b-twotrack.resume.log`.

- **Headline:** corrections fix 126/129 training and 106/119 paraphrase failures. Held-out: corrections 14 fixed / 15 broke, lessons 11 / 21, both 17 / 21. All 4 pre-registered predictions hold in direction (table in RESULTS.md).
- **compare.py bug, fixed:** phase 2 off runs have no `retrieved_correction_ids` column, so the merge didn't suffix it and the pairs CSVs silently dropped correction ids. Now `retrieved_*` columns are always taken from the on run. Pairs CSVs regenerated; counts unchanged.
- **Everything that fixed something was a copy.** Train: 126/126 fixes equal the stored correct answer. Paraphrase: 102/106. Own correction retrieved at rank 1 for 100% of train and paraphrase (min cosine 0.763, median 0.945). So the paraphrase set tested MiniLM's robustness to rewording, not the model. My pre-registered framing overclaimed what this set could show.
- **Held-out changes are noise on fragile answers.** 0 of 29 changed answers (fixed or broke) copy a retrieved correction. 17 of 33 broken held-out answers break in at least 2 arms. Broken answers have lower memory-off confidence (from `activations.npz` `conf_off`; median 0.92–0.97 vs 1.00, MWU p < 1e-4) in every arm. Same pattern on train (10 broke under corrections, all questions with no correction of their own).
- **Similarity gate, simulated (post hoc):** top-1 cosine to a stored question is ≥ 0.763 when the own correction exists and ≤ 0.641 on held-out. Gate at 0.7 gives train 98.8%, paraphrase 93.2%, held-out 52.8% (= off), with 0 broke anywhere. Exact rather than estimated, because a gated-out prompt is byte-identical to memory-off and greedy decoding is deterministic. One train question gated in via a neighbour (0.76) and didn't break. Needs a fresh split to confirm.
- **Unfixed:** train 3 (2 grader artifacts: "Thetis" vs "(HMS) Thetis", the 9-country list). Paraphrase 13, including grader artifacts ("GM", "tonycurtis") and paraphrases that changed the question ("currently serves as Taoiseach" lost its date, so "Simon Harris" is arguably right).
- Analysis script was ad hoc (scratchpad, not committed); its numbers are reproducible from the pairs CSVs + `corrections.csv` + `activations.npz`.

## Phase 4 setup — gate replication + placebo (2026-11-17)

Design + predictions pre-registered in RESULTS.md ("Phase 4 design") before any run.

- **Gate** (`agent.py`, `--gate` / `CORRECTION_GATE`): if the top-1 correction's cosine similarity is < 0.7, drop all corrections, so the prompt is byte-identical to memory-off. Otherwise use the usual top-3. That's the exact rule simulated in phase 3, not a per-correction filter. `retrieve_corrections` now returns `sim` = 1 − d/2 (MiniLM vectors are unit-length; checked, and the held-out max of 0.641 matches the phase 3 analysis).
- **Placebos** (`memory_mode`): `placebo_lorem` puts "Notes:" + lorem ipsum in the corrections slot, cut at a word to the length of that question's real corrections block (within 12 chars). `placebo_random` puts in 3 corrections drawn at random, seeded by crc32 of the question.
- **Prompt checks** on 40 questions against `git show HEAD:agent.py`: off, ungated corrections and both are byte-identical to phase 3. Gated equals off below 0.7 and ungated at or above it. Lorem length matches. Random is reproducible.
- **Fresh split:** `build_triviaqa.py --seed 1 --exclude <phase 2/3 files>` writes `questions_trivia_v2.csv` and `questions_trivia_v2_holdout.csv` (250 + 250, 0 overlap with the old 500 or with each other; 8551 eligible).
- **Runner:** `run_phase4.py`. Paraphrases go over the API in the background while the GPU runs the placebos. Resumable; logs in `runs/gemma3-4b-phase4.log` (first launch) and `runs/gemma3-4b-phase4.resume.log`.
- **First launch skipped building corrections.** Importing `memory.py` creates an empty `corrections.csv`, so the train/off eval left one behind and the "build if missing" check saw it as built. Caught during train/corrections, which would have run with an empty store and so been a copy of memory-off. Stopped before that output was saved. The runner now checks for rows and refuses to run the arms unless #corrections = #training failures. On the relaunch: 119 corrections for 119 failures (fresh train off = 52.4%, reused from the first launch). Outputs: `runs/gemma3-4b-placebo/`, `runs/gemma3-4b-phase4/`.

## Phase 4 run — results (2026-11-18)

Took 4 launches: the first skipped the corrections build (see setup), then the job was stopped by Claude Code's low-memory reaper twice. Each relaunch resumed from the finished files. Logs: `runs/gemma3-4b-phase4{,.resume,.resume2,.resume3}.log`.

- **All 6 predictions hold** (tables in RESULTS.md). Fresh gated: train 100% (119 fixed / 0 broke), paraphrase 89.4% (98 / 0), held-out = off (0 / 0). Ungated broke 17 / 16 / 10.
- **Gated == simulation, answer for answer:** 0 mismatches in 735 (gated-in train 119, paraphrase 115, held-out 0). So the phase 3 simulation method was exact.
- **Margins:** fresh held-out top-1 max 0.572. Paraphrase own-correction min 0.730 (thinner than phase 3's 0.763). Train right-off (no own correction) max 0.565.
- **Copies:** train 118 / 119 fixes, paraphrase 96 / 102. 0 / 43 broken answers copy a correction. The 4 paraphrase fixes lost under the gate are paraphrases with no own correction (chance flips).
- **Placebo:** lorem 11 fixed / 14 broke, random 13 / 18, against corrections 14 / 15. Pairwise broke overlap across lessons / corrections / random / lorem is 7–11 vs ~2 expected (hypergeom p < 2e-4). 98 / 132 right answers never break under any arm. Broken answers have lower memory-off confidence in every arm (lorem median 0.82, p = 7.6e-8).
- **RAM investigation (why the reaper kept firing):** an eval process holds ~2 GB resident on top of 4.7 GB VRAM. Its ~7.5 GB "private" commit isn't resident and can't be freed: a pre-quantized 8-bit copy, `torch._C._host_emptyCache()` and `_heapmin` all left it unchanged. It's most likely the mmap'd safetensors commit charge. The reaper fires on system-wide pressure (VS Code ~3.6 GB, R ~2 GB, browser). Long runs belong in a user terminal or under `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.

## Open follow-ups (not done tonight)

For the **strategic forward plan** (two-track memory experiment + the broader human-cognition arc), see RESULTS.md → "Where this goes next." The items below are the *tactical* leftover work from the current experiment.


- Re-run seen-40 baseline + memory-on with Llama to repopulate `results_baseline.csv` / save a `results_memory.csv`. The held-out result files are saved; the seen-40 attempt files were overwritten by the held-out runs and only exist in logged stdout.
- Step 4 prompt v3: explicitly steer the model toward identifying *factual gaps* vs *process errors* as distinct lesson types. Currently it conflates them.
- Curated held-out v2: pre-screen candidate held-out questions against the target model and keep only the ones it fails. Current held-out was designed by hypothesizing failure modes; a model-targeted design would have actual signal.
