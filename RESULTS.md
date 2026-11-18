# RESULTS — reflective_ai

A controlled experiment testing whether a frozen LLM can improve on novel questions by applying lessons extracted from its own past mistakes — without any fine-tuning. Reflexion-style memory, deterministic grader, training/held-out split.

- **Phase 1** (Llama 3.3 70B, 64 hand-written questions): small-N limit result — lessons help only when the model's underlying knowledge is right, and can hurt when it isn't.
- **Phase 2** (Gemma 3 4B, 500 TriviaQA questions, plus activations): at scale, lessons are **net harmful** — they break ~2× as many correct answers as they fix; the harm lands on less-confident knowledge, acts in the late layers where the answer forms, and no confidence gate rescues it. See [Phase 2](#phase-2--harm-at-scale-gemma-3-4b-triviaqa-interpretability).
- **Phase 3** (same model and questions, plus 236 paraphrases): storing the **corrected fact** instead of a lesson fixes 126 / 129 training failures and 106 / 119 paraphrased ones, but as a lookup: the retriever matches the rewording and the model copies the stored answer. On unrelated questions, corrections are noise (14 fixed, 15 broke), and a similarity gate removes all of that harm. Lessons on top only hurt. See [Phase 3](#phase-3--two-track-memory-corrections-vs-lessons).

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

## Phase 2 — harm at scale (Gemma 3 4B, TriviaQA, interpretability)

Phase 1 had too few failures to measure anything (Llama hit 100% on the held-out with memory off) and no access to activations. Phase 2 changes three things: a weaker **local** model whose internals we can read, **500 questions** instead of 64, and a metric that counts **harm** instead of only net accuracy.

### Setup changes

- **Model:** `google/gemma-3-4b-it`, run locally in 8-bit (bitsandbytes) on an 8GB RTX 4060, greedy decoding. Same prompt, reflection prompt, retrieval (top-3), and embeddings as phase 1.
- **Questions:** 250 training + 250 held-out sampled (seed 0) from TriviaQA rc.nocontext validation (`build_triviaqa.py`). Unlike phase 1 these aren't grouped into failure-mode categories — they're random trivia.
- **Grader:** still deterministic. Now accepts any TriviaQA alias, matches whole words, and reads number words ("Nine"). Re-grading every phase 1 file with the new grader changes nothing.
- **Harm metric (`compare.py`):** each question is classified by its memory-off → memory-on outcome: *fixed* (wrong → right), *broke* (right → wrong), stayed right, stayed wrong. **Harm rate** = broke / right-with-memory-off.

### Results

| Set | Memory OFF | Memory ON | Fixed | Broke | Harm rate |
|---|---|---|---|---|---|
| Training (250) | 48.4% | 44.8% | 9 | 18 | 14.9% |
| Held-out (250) | 52.8% | 48.8% | 11 | 21 | 15.9% |

Pooled: 39 broke vs 20 fixed, two-sided sign test p = 0.018. (Per split p ≈ 0.11–0.12.) A pilot on the phase 1 questions showed the same shape at tiny N (held-out: 0 fixed, 1 broke).

### Findings

**6. At scale, lessons are net harmful.** Memory breaks about two correct answers for every one it fixes, on both splits. This is the phase 1 Hoover Dam backfire, no longer an anecdote.

**7. Even exact-match lessons don't fix knowledge errors.** Every failed training question has its own lesson in the store and retrieves it — yet only 9/129 are fixed. A lesson is a rule, not the missing fact. Phase 1's finding 3, now at N = 250.

**8. Harm swaps a correct answer for a related wrong one — it isn't copying.** Maastricht → Lisbon, Wellington → Buenos Aires, Chess → Go, Tin → Indium, George III → George IV. None of the 39 broken answers appear in the retrieved lesson text. The retrieved lessons are topic-matched heuristics ("prioritize the most immediately obvious and widely known neighboring country") that steer the model off an answer it had right.

**9. The model's internals predict harm — but no better than its own confidence.** Among the 253 questions answered correctly without memory, a linear probe on the memory-off residual stream (before any lesson is seen) predicts which ones memory will break:

| Predictor | AUC |
|---|---|
| Model's own confidence (probability of its answer's first token) | **0.811** |
| Linear probe, best layer (22 of 34), cross-validated | 0.785 |
| Probe + confidence + lesson similarity | 0.787 |
| Shuffled-label null for best-layer probe (20 shuffles) | max 0.688 |
| Lesson similarity to question | 0.556 |

The probe signal is real (well clear of the null) and builds with depth from layer ~18. But it carries nothing beyond confidence: **memory breaks the answers the model holds less firmly** (median confidence 0.92 for broken vs 1.00 for stayed-right). How closely the lesson matches the question doesn't matter (≈ chance).

**10. The lesson acts in the late layers where the answer is assembled.** Logit lens at the answer position, logit(memory-off answer) − logit(memory-on answer), averaged over broken cases:

| Layer | 0 | 9 | 18 | 21 | 24 | 27 | 30 | 33 (output) |
|---|---|---|---|---|---|---|---|---|
| No lesson | −0.52 | −0.48 | −0.03 | −0.45 | −0.66 | −0.23 | +2.30 | +5.58 |
| With lesson | −0.41 | −0.47 | −0.15 | −0.71 | −1.10 | −1.69 | −0.49 | −3.92 |

The two curves are indistinguishable through layer ~18, start to separate at 21, and split sharply from 27 on. Without the lesson the correct answer only takes the lead between layers 27 and 30 — it forms late, and that's exactly where the lesson pushes against it. Lessons that *fix* an answer act at the same depth in the opposite direction.

**11. No confidence gate rescues the lessons.** Simulating "use lessons only when memory-off confidence is below τ" with τ chosen on training: the best τ is *never use lessons*. Fixed and broken answers are about equally confident (median 0.88 vs 0.92), so no threshold keeps the fixes without the harms. Confidence separates broke from stayed-right, but not fixed from broke.

### What phase 2 means

Phase 1 said reflection helps only under narrow conditions. Phase 2 says that **on factual recall, those conditions almost never hold, and the default effect is harm.** A rule-shaped lesson has almost nothing to contribute to a question whose failure is a missing or wrong fact, and when it's retrieved for a question the model *does* know — but not firmly — it overrides that knowledge in the layers where the answer is being formed. Changing *when* lessons are retrieved (gating on confidence) can't fix this. The fix has to change *what* is stored — which is exactly the two-track experiment below.

### Phase 2 caveats

- **One model, 8-bit.** Activations are from the 8-bit model; fine for probes and logit lens, slightly off-distribution for off-the-shelf SAEs (not used).
- **Single seed, greedy decoding,** one reflection prompt. Gemma's lessons are weak (several near-platitudes); a better reflection prompt might change the fixed count, though finding 7 suggests the ceiling is low for rule-shaped lessons on trivia.
- **The reflection prompt saw the full alias list as "the correct answer"** (e.g. "Gorbacev|Gorbachav|…") rather than one canonical answer — a bug found in phase 3 setup. It may have degraded lesson quality. Fixed going forward; phase 3 reuses these lessons as-is so the lessons arm stays comparable.
- **Grader noise.** TriviaQA aliases are sometimes loose; spot-checking the 59 changed pairs found ~4–6 alias artifacts per side (e.g. "The DLS method" counted as broke). Not enough to close the broke/fixed gap.
- **Logit lens uses the first answer token only,** and averages over 34 broke / 13 fixed cases.

## Phase 3 — two-track memory (corrections vs lessons)

Phase 2 showed that rule-shaped lessons can't fix a wrong fact. Phase 3 stores the fact itself. A **correction** is a failed question, the model's wrong answer, and the correct (canonical) answer. It is indexed by the question's embedding, and the top 3 are put in the prompt above any lessons. Same model (Gemma 3 4B, 8-bit, greedy), same 129 phase 2 lessons, same 129 failures (each gets one correction). Design and predictions were [pre-registered](#phase-3-design-updated-after-phase-2-pre-registered-2026-11-16-before-any-run) before the run. The memory-off and lessons runs are reused from phase 2, since a fresh-process re-run of 40 questions gave 40/40 identical answers.

### Results

| Set | Arm | Accuracy | Fixed | Broke | Harm rate |
|---|---|---|---|---|---|
| Training (250) | off | 48.4% | | | |
| | lessons | 44.8% | 9 / 129 | 18 / 121 | 14.9% |
| | **corrections** | **94.8%** | **126 / 129** | 10 / 121 | 8.3% |
| | both | 90.0% | 118 / 129 | 14 / 121 | 11.6% |
| Paraphrase (236) | off | 49.6% | | | |
| | lessons | 45.8% | 8 / 119 | 17 / 117 | 14.5% |
| | **corrections** | **89.8%** | **106 / 119** | 11 / 117 | 9.4% |
| | both | 89.0% | 104 / 119 | 11 / 117 | 9.4% |
| Held-out (250) | off | 52.8% | | | |
| | lessons | 48.8% | 11 / 118 | 21 / 132 | 15.9% |
| | corrections | 52.4% | 14 / 118 | 15 / 132 | 11.4% |
| | both | 51.2% | 17 / 118 | 21 / 132 | 15.9% |

Held-out sign tests (fixed vs broke): lessons p = 0.11, corrections p = 1.0, both p = 0.63.

### Predictions, checked

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | Training, corrections fix > 60% | 126 / 129 = 97.7% | **Holds** (expected: it's a lookup) |
| 2 | Paraphrase, corrections fix > 40% of those wrong without memory | 106 / 119 = 89.1% | **Holds, but doesn't test what it was meant to.** See finding 13 |
| 3 | Held-out, corrections harm < 15.9% | 11.4% (15 vs 21 broken) | **Holds in direction**, but 15 vs 21 isn't a significant difference |
| 4 | Both = same fixes, harm ≥ corrections | Fewer fixes (118 vs 126 train), harm ≥ on every set | **Holds.** Lessons cost fixes as well as adding harm |

### Findings

**12. Storing the fact fixes the fact.** With corrections, 126 of 129 training failures are fixed, and every one of the 126 answers is the stored correct answer. Phase 2's lessons fixed 9. Of the 3 left, 2 are grader artifacts ("Thetis" vs alias "(HMS) Thetis"; a 9-item list). The third is a question the model answered with dimensions instead of a ship's name.

**13. The paraphrase test measured the retriever, not the model.** Every paraphrase whose source question had a correction retrieved that correction at rank 1 (122 / 122). The lowest cosine similarity was 0.76, and the median was 0.95. 102 of the 106 fixes are the stored answer copied over. So the corrections "transfer" to new wording only because the embedding treats the rewording as the same question. That shows the lookup is robust to wording. It doesn't show the model learned the fact, and the pre-registered framing ("learning vs lookup keyed on wording") was too generous to this test. A real transfer test needs a question that *uses* the fact without asking for it, such as a second hop or an inverse question. Of the 13 paraphrases still wrong, several are grader artifacts ("GM", "tonycurtis") and a few paraphrases changed the question ("who currently serves as Taoiseach" dropped the date).

**14. Off-topic memory acts as noise on fragile answers, whatever it says.** On held-out, where no correction is relevant, corrections fixed 14 and broke 15. That's symmetric, and **none of the 29 changed answers is copied from a retrieved correction.** The "fixes" are as accidental as the breaks. The same questions break across arms: 17 of the 33 broken held-out answers break under at least two arms. In every arm, broken answers had lower memory-off confidence than answers memory never broke (held-out median 0.92–0.97 vs 1.00, Mann-Whitney p < 0.0001). Prediction 3's risk, that the model copies an irrelevant correction, did not happen. The picture: **any extra context flips answers the model holds loosely, and lessons tilt those flips toward harm** (11 fixed vs 21 broke). Irrelevant corrections don't tilt them (14 vs 15). Training shows the same thing: the 10 training answers that corrections broke are questions the model already had right, which therefore have no correction of their own and only retrieved neighbours'.

**15. A similarity gate separates relevant from irrelevant corrections almost perfectly.** The top-1 cosine similarity between a question and the nearest stored question is ≥ 0.76 for every paraphrase or training question whose own correction exists. It is ≤ 0.64 for every held-out question. Simulating "use corrections only when top-1 similarity ≥ 0.7":

| Set | Off | Corrections, ungated | Corrections, gated at 0.7 | Fixed / broke (gated) |
|---|---|---|---|---|
| Training | 48.4% | 94.8% | **98.8%** | 126 / 0 |
| Paraphrase | 49.6% | 89.8% | **93.2%** | 103 / 0 |
| Held-out | 52.8% | 52.4% | **52.8%** | 0 / 0 |

This is exact rather than an estimate, because a gated-out prompt is byte-identical to the memory-off prompt and decoding is deterministic. But **the threshold was chosen after seeing these results**, so it needs confirming on a fresh split. Compare finding 11: confidence couldn't gate lessons, because a lesson's value doesn't depend on how well it matches the question. A correction's value depends on exactly that.

### What phase 3 means

The two-track hypothesis splits into two answers. The correction track works, but only as what it is: a cache of fixed facts, keyed robustly enough to survive rewording, with its harm removable by a similarity gate. The principle track adds nothing on factual recall. Layered on top of corrections, it loses fixes (126 → 118) and adds harm. Across three phases, nothing here is evidence of the model *learning* from a mistake. When it improves, a lookup supplied the answer. When it gets worse, extra context knocked loose an answer it held weakly. That points the next step at either (a) changing what the model knows (weight editing, e.g. ROME/MEMIT), or (b) testing whether a stored fact can be *used* rather than just retrieved.

### Phase 3 caveats

- **Corrections are only as good as the answer key.** TriviaQA's canonical answers are sometimes odd ("Pent house"), and the stored correct answer is taken from the key without checking.
- **The 0.7 gate is post hoc** (finding 15), and the gap it exploits (0.64 vs 0.76) is specific to these 500 random trivia questions with MiniLM embeddings. Near-duplicate questions with different answers ("consolation prize a chequebook and pen" vs "… a dustbin", both in training) would defeat a pure similarity gate.
- **Paraphrases come from one model** (Nemotron 3.5 Super). 14 were rejected for leaking the answer, and 3 are near-copies of the original.
- **Same single seed and 8-bit model as phase 2.** The harm counts on held-out are small (15–21), and differences between arms there are within noise.

## Phase 4 design — does the gate hold, and is the harm just noise? (pre-registered 2026-11-17, before any run)

Phase 3 left two claims resting on post-hoc analysis: the 0.7 similarity gate (finding 15), and "off-topic memory acts as noise on fragile answers, whatever it says" (finding 14). Phase 4 tests both. Same model, prompt, top-3 retrieval and grader.

**A. Gate replication on fresh questions.** 250 new training and 250 new held-out questions from TriviaQA (seed 1, disjoint from the phase 2/3 500), plus Nemotron paraphrases of the new training questions. Corrections are built from the new training failures. The gate is fixed in advance: **include the top-3 corrections only if the top-1 cosine similarity is ≥ 0.70**, otherwise use the memory-off prompt. This is the rule simulated in phase 3. Each set runs memory-off, ungated corrections and gated corrections.

1. **Held-out, gated:** at most 1 broken answer, and accuracy within 1 question of memory-off.
2. **Training, gated:** keeps ≥ 95% of ungated fixes, with at most 2 broken.
3. **Paraphrase, gated:** keeps ≥ 90% of ungated fixes, with at most 2 broken. The threshold is looser than training because a paraphrase is further from its stored question.
4. **Held-out, ungated:** harm reappears (broke ≥ 5 of the questions right without memory), so the gate has something to remove.

**B. Placebo on the phase 3 held-out set.** The same 250 held-out questions, with the corrections block replaced by:

- **lorem:** lorem-ipsum filler of the same character length as that question's real corrections block, under a "Notes:" header. Extra text with no facts.
- **random:** 3 corrections drawn at random from the store (seeded per question) instead of the 3 nearest. Facts, but less related than the nearest.

5. **Lorem:** at least half as many changed answers (fixed + broke) as real corrections (≥ 15 vs 29), at least 8 broken, and broken answers with lower memory-off confidence than unbroken ones. If lorem changes fewer than 10 answers, finding 14 is wrong: the content of off-topic facts matters, not just the extra text.
6. **Random:** fixed and broke each within ±6 of the corrections arm (14 / 15), since nearest held-out corrections are already unrelated (median similarity 0.34).

## Where this goes next

This experiment is one slice of a longer arc. The two-track experiment below is now done (see [Phase 3](#phase-3--two-track-memory-corrections-vs-lessons)). The candidates after it are: confirm the similarity gate on a fresh split; add a placebo arm (irrelevant text of matched length) to test finding 14 directly; build a fact-*use* test (two-hop or inverse questions) for real transfer; and try weight editing (ROME/MEMIT), which changes what the model knows instead of what it's shown.

### The immediate next experiment — two-track memory (correction + principle)

The clearest finding above is that **strategy-shaped lessons alone cannot fix a wrong belief.** Humans, when corrected, store *both* the corrected fact AND a meta-strategy. We tested only the second half and found it inadequate on its own — and in one case (Hoover/Empire State), it actively hurt by exercising wrong recall under structured reasoning.

The minimum experiment to test this:

1. **Principle lessons** (what we have): *"for which-came-first questions, compare actual years."*
2. **Correction memos** (new): *"Microsoft was founded 1975; Apple 1976. Earlier wrong answer: Apple."*

Retrieved together at attempt time, the principle supplies *how* to reason and the correction supplies the *facts* to reason over. That's a closer model of how humans actually update after being corrected. It should also eliminate the Hoover/Empire backfire — alongside "compare years," the agent would retrieve "Empire State 1931 < Hoover 1936."

**Scope:** roughly 30 lines of code. Add `record_correction(question, expected_answer)` to `memory.py` alongside `record_lesson`. Have `reflect.py` record both on each failure. Add a second retrieval pass in `agent.answer`. Re-run the full pipeline (seen baseline → reflect → seen memory-on → held-out off → held-out on) and compare to the single-track numbers above.

**Pre-registered prediction:** correction-augmented memory substantially outperforms principle-only — the seen improvement comes from corrected facts being directly retrievable; the held-out improvement comes from the Hoover/Empire-style backfires being eliminated by supplying correct facts alongside the strategy. If the prediction holds, the finding becomes concrete: **the missing ingredient in LLM "learning from mistakes" isn't reflection — it's targeted fact-correction paired with strategy.** If the prediction fails, that's also informative — it would mean something deeper than memory architecture is missing.

#### Phase 3 design, updated after phase 2 (pre-registered 2026-11-16, before any run)

Phase 2 exposes a design problem: the TriviaQA held-out shares no facts with training, so a stored correction *can't* help there — only harm can be measured. To separate **learning** from **memorization**, phase 3 adds a third question set:

| Set | What it tests |
|---|---|
| Training (250, exact wording) | Memorization ceiling — the correction for this exact question is in the store |
| **Paraphrase (250)** — each training question reworded, same answer | Does a stored correction transfer to a new phrasing of the same fact? |
| Held-out (250, unrelated) | Harm — do unrelated stored items break correct answers? |

Three memory arms on the same model (Gemma 3 4B), same 129 lessons from phase 2, same top-3 retrieval: **lessons only** (phase 2), **corrections only**, **both**. A correction is the failed question, the model's wrong answer, and the correct answer, retrieved by question similarity.

Predictions:

1. **Training, corrections:** fix most of the 129 failures (> 60%). Near-lookup — memorization, not learning.
2. **Paraphrase, corrections:** fix a substantial share of paraphrased questions that are wrong without memory (> 40%). This is the real test: if it fails, "corrections" are only a lookup table keyed on wording.
3. **Held-out, corrections:** harm rate below lessons-only (< 15.9%) — a fact about an unrelated question shouldn't push the model off its own recall the way a heuristic does. Risk: the model may copy an answer from an irrelevant correction.
4. **Both vs corrections only:** same fixes, harm at least as high — the principle track adds nothing on factual recall and carries the harm.

### The broader arc

*Can LLMs reason like humans?* isn't a project; it's a series of small experiments, each isolating one specific dimension of human-like cognition and asking whether the LLM, with appropriate scaffolding, behaves that way. Plausible next slices after the two-track experiment:

- **Uncertainty calibration.** Humans hedge when they don't know; LLMs are often falsely confident. Can a reflection-style system learn to flag low-confidence answers based on past failures?
- **Self-correction without external feedback.** Can the agent catch its own mistakes through internal checking before committing to an answer? Tests whether reasoning + retrieval can substitute for a grader.
- **Cross-question pattern learning.** If the model fails N questions sharing a structural feature it wasn't told about, can reflection discover the shared feature? Tests inductive abstraction.
- **Theory of mind via reflection.** When a lesson concerns another agent's likely behavior, does the retrieval mechanism surface it appropriately?

Each of these is a small standalone experiment, comparable in scope to what's already been done. None requires committing to an open-ended research program. The work compounds — the methodology (deterministic grader, controlled task set, seen/held-out split) carries between slices.

### Methodology that transfers

Independent of which research question comes next, this experiment built up a portable toolkit worth holding onto:

- **Deterministic grading first; LLM judge only when proven necessary.** We tried the LLM judge and it rubber-stamped wrong answers as correct. Don't go back unless deterministic grading genuinely can't handle a category of answer.
- **Controlled task design with failure-mode grouping.** Even though our held-out result showed that frontier-model failures don't cluster by category, the *grouping discipline* is what surfaced that finding. Without it, the result would have looked like noise.
- **Seen vs held-out separation, always.** Without held-out, memorization is indistinguishable from learning. Don't trust a memory-on result that doesn't have a held-out comparison; the paper-ignite "fix" on this experiment is exactly the contamination it prevents.
- **Document the dead ends.** NOTES.md captures every model swap, grader iteration, prompt v1→v2 transition, and provider outage. These notes save hours when picking the project back up cold.

### Conditions for picking back up

- **For a full cycle:** `python run_cycle.py --name <run> --backend local --train <csv> --holdout <csv>` writes every step to `runs/<run>/` (resume with `--from-step N`), then `compare.py` reports fixed/broke per split.
- **For revising the held-out:** the current set was designed by hypothesizing failure modes; the null result showed this was the wrong assumption for a strong model. Pre-screen ~100 candidate held-out questions against Llama 3.3 70B and keep only the ~20 it actually fails. Converts the null into a real transfer measurement.
- **Phase 1 is no longer reproducible on NVIDIA:** `meta/llama-3.3-70b-instruct` reached end of life there on 2026-08-26 (410 Gone). The saved result files and NOTES are the record. Phase 2/3 run locally and don't depend on any API except paraphrase generation.
- **If the model degrades on NVIDIA again:** the client is OpenAI-compatible. Swap `LLM_BASE_URL` and `LLM_MODEL` in `.env` to point at any OpenAI-protocol endpoint (Ollama localhost, a different NVIDIA NIM, etc.). Embeddings are local and won't break.
- **If picking up cold months later:** read this file, then NOTES.md's "Open follow-ups" section. They contain the load-bearing context to resume without re-deriving framing.

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

Phase 2:

- `run_cycle.py` (full cycle into `runs/<name>/`), `compare.py` (fixed/broke/harm rate), `build_triviaqa.py`
- `interp_capture.py` (residual stream + logit lens), `interp_analyze.py` (harm probe, baselines, permutation null)
- `questions_trivia.csv`, `questions_trivia_holdout.csv` — 250 + 250 TriviaQA questions
- `runs/gemma3-4b/` — pilot on the phase 1 questions; `runs/gemma3-4b-trivia/` — the 500-question run, lessons, `*_pairs.csv`, `analysis.txt`, `probe_auc_by_layer.csv` (`activations.npz`, 165MB, is gitignored; regenerate with `interp_capture.py`)

Phase 3:

- `build_paraphrases.py` → `questions_trivia_paraphrase.csv` (236 reworded training questions, with `source_question`)
- `run_twotrack.py` (builds corrections, runs 3 sets × 4 arms, resumable, then compares)
- `runs/gemma3-4b-twotrack/`: `corrections.csv` (129), `{train,paraphrase,holdout}_{off,lessons,corrections,both}.csv`, and `*_pairs.csv` with retrieved lesson and correction ids
