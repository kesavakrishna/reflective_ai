# RESULTS — reflective_ai

A controlled experiment testing whether a frozen LLM can improve on novel questions by applying lessons extracted from its own past mistakes — without any fine-tuning. Reflexion-style memory, deterministic grader, training/held-out split.

- **Phase 1** (Llama 3.3 70B, 64 hand-written questions): small-N limit result — lessons help only when the model's underlying knowledge is right, and can hurt when it isn't.
- **Phase 2** (Gemma 3 4B, 500 TriviaQA questions, plus activations): at scale, lessons are **net harmful** — they break ~2× as many correct answers as they fix; the harm lands on less-confident knowledge, acts in the late layers where the answer forms, and no confidence gate rescues it. See [Phase 2](#phase-2--harm-at-scale-gemma-3-4b-triviaqa-interpretability).
- **Phase 3** (same model and questions, plus 236 paraphrases): storing the **corrected fact** instead of a lesson fixes 126 / 129 training failures and 106 / 119 paraphrased ones, but as a lookup: the retriever matches the rewording and the model copies the stored answer. On unrelated questions, corrections are noise (14 fixed, 15 broke), and a similarity gate removes all of that harm. Lessons on top only hurt. See [Phase 3](#phase-3--two-track-memory-corrections-vs-lessons).
- **Phase 4** (500 fresh questions plus paraphrases, and a placebo test): the similarity gate, fixed in advance at 0.7, replicates. It gives 100% training, 89.4% paraphrase, and held-out identical to memory-off, with **0 broken answers out of 735**. Lorem-ipsum filler of the same length breaks as many correct answers as real off-topic memory, and the same ones, so most of the "harm" in phases 2–3 comes from a longer prompt, not from what memory says. See [Phase 4](#phase-4--results).
- **Phase 5** (writing the 119 corrections into the weights with a single-layer, MEMIT-style edit): the edit gets 93 / 119 corrected questions right and breaks only 4 / 134 right training answers. But it transfers to 13 / 101 paraphrases (retrieval: 98 / 101) and breaks 12 held-out answers that can't be gated (retrieval: 0). A placebo edit toward shuffled answers installs almost none of them (2 / 119), so the edit mostly unlocks answers the model half-knew rather than writing new facts. See [Phase 5](#phase-5--results).
- **Phase 6** (114 two-hop questions that need a corrected fact as an intermediate step): with the correction in the prompt, Gemma completes the second hop on 20 / 47 target items, and gated retrieval nearly matches that (19). With the fact in the weights it manages 1 / 43, even though in 34 of those items the edited model answers the original question right and Gemma knows the second hop. So retrieval gives a fact the model can reason with about 40% of the time, while the edit gives a lookup tied to one question. See [Phase 6](#phase-6--results).
- **Phase 7** (an "Entity: … / Answer: …" chain prompt on the same two-hop set): with the fact in context it fixes 23 / 47 targets (plain prompt: 20), but it breaks 18 of the 36 two-hop answers Gemma gets right without memory, because the 4B model confuses the two slots. The plain prompt is the better trade. See [Phase 7](#phase-7--results).
- **Phase 8** (118 near-duplicate questions that reuse a stored question's wording but have a different answer): the 0.7 gate fires on 63 of them, and on 15 of those Gemma answers with the stored, wrong answer. Gated retrieval breaks 13 / 60 right answers here. No similarity threshold separates near-duplicates from true paraphrases (best: about 75% / 75%). See [Phase 8](#phase-8--results).
- **Phase 9** (multi-layer MEMIT-style edit over 5 layers, with and without reworded keys): no setting kept collateral within the pre-registered drift limit (lowest 16.4% against a 14.2% limit), so neither arm was tested. On dev, spreading over layers raised drift, and reworded keys doubled paraphrase transfer only because they nearly duplicate the paraphrases. See [Phase 9](#phase-9--results-dev-sweep-only-neither-arm-reached-the-test).

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

## Phase 4 — results

All six pre-registered predictions hold.

### A. The gate on fresh questions

119 corrections, one per fresh training failure. The gate threshold was 0.7 throughout, never tuned.

| Fresh set | Off | Corrections, ungated | Corrections, gated at 0.7 |
|---|---|---|---|
| Training (250) | 52.4% | 93.2% — fixed 119, **broke 17** | **100.0%** — fixed 119, broke 0 |
| Paraphrase (235) | 47.7% | 84.3% — fixed 102, **broke 16** | **89.4%** — fixed 98, broke 0 |
| Held-out (250) | 44.0% | 45.6% — fixed 14, **broke 10** | **44.0%** — fixed 0, broke 0 |

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | Held-out gated: ≤ 1 broke, accuracy within 1 of off | 0 broke, identical to off | **Holds** |
| 2 | Training gated: ≥ 95% of ungated fixes, ≤ 2 broke | 119 / 119, 0 broke | **Holds** |
| 3 | Paraphrase gated: ≥ 90% of ungated fixes, ≤ 2 broke | 98 / 102 = 96%, 0 broke | **Holds** |
| 4 | Held-out ungated: ≥ 5 broke | 10 broke | **Holds** |

**16. The similarity gate replicates: on 735 fresh answers it removes every broken answer and keeps 96–100% of the fixes.** No fresh held-out question gets within reach of a stored one (top-1 similarity ≤ 0.57). Every paraphrase whose source has a correction clears the gate (lowest 0.73, median 0.93). That margin is thinner than phase 3's 0.76, so 0.7 is close to the edge for paraphrases. The 4 paraphrase fixes lost to the gate are paraphrases whose own source question has no correction: answers that changed by chance, not lookups. The rest replicates phase 3 exactly. 118 / 119 training fixes and 96 / 102 paraphrase fixes are the stored answer copied over, and none of the 43 broken answers on any set copies a correction. The gated runs match the simulation answer for answer on all 735 questions, which confirms that phase 3's simulation method was exact.

### B. Placebo on the phase 3 held-out set

| Phase 3 held-out (132 right without memory) | Fixed | Broke | Harm rate |
|---|---|---|---|
| Lessons (phase 2) | 11 | 21 | 15.9% |
| Nearest corrections (phase 3) | 14 | 15 | 11.4% |
| **Random corrections** | 13 | 18 | 13.6% |
| **Lorem ipsum, same length** | 11 | 14 | 10.6% |

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 5 | Lorem: ≥ 15 changed, ≥ 8 broke, broken = less confident | 25 changed, 14 broke, median confidence 0.82 vs 1.00 (p < 1e-7) | **Holds** |
| 6 | Random within ±6 of 14 fixed / 15 broke | 13 / 18 | **Holds** |

**17. Meaningless filler does about as much damage as off-topic memory, and to the same answers.** Lorem ipsum with no facts in it breaks 14 correct answers, compared with 15 for real corrections and 21 for lessons. The broken sets overlap far beyond chance. For every pair of the four arms, 7–11 questions break under both, where chance predicts about 2 (hypergeometric p < 2e-4 for every pair). The specific swaps repeat: Liddell → Carroll, Ritchie Valens → J. P. Richardson, Wrist → another body part, Polystyrene → Polypropylene. 98 of the 132 correct answers never break under any arm, and the damage falls on about 34 fragile ones. In every arm these have lower memory-off confidence. **Most of the "harm from memory" in phases 2 and 3 is harm from a longer prompt.** Lessons may add a little on top (21 broke vs 14–18 for the others), but that gap is within noise at this N.

### What phase 4 means

Four phases come to a simple account of retrieval memory on factual QA with a small model:

- **When the stored item is the answer to this question** (the same question or a rewording), the model copies it. That's worth +40–50 points, and it's lookup, not learning.
- **When it isn't**, the stored text acts as noise, whatever it says. It flips about 10–15% of the answers the model holds loosely, slightly more often toward wrong than right.
- **A similarity gate separates the two cases cleanly** on random trivia. Gated correction memory is strictly better than no memory on every set measured: +0 on unrelated questions, +40–48 points on seen and reworded ones, 0 broken out of 735.

None of this makes the model know anything new. Gated corrections are a good cache. To change what the model knows, rather than what it is shown, the next step is weight editing.

### Phase 4 caveats

- **Same model, decoding and answer key as phases 2–3.** The fragile-answer effect (finding 17) might be smaller in a larger model that holds its answers more firmly.
- **The gate is tested on random TriviaQA,** where unrelated questions are far apart. Its weak point is near-duplicates with different answers ("consolation prize a chequebook and pen" vs "… a dustbin"), which random sampling rarely produces. The paraphrase margin (0.73 vs 0.70) is narrow.
- **Confidence for the placebo analysis comes from phase 2's activation capture** (first answer token, memory off), and covers the phase 3 held-out set only.
- **Lorem is one kind of filler.** Other content-free text (blank notes, repeated tokens) might behave differently. What this shows is that the content of off-topic corrections is not what's needed for the harm.

## Phase 5 design — write the corrections into the weights (pre-registered 2026-11-18, before any result)

Retrieval makes the model copy a stored answer. Weight editing changes what the model computes, so it can't be gated per question, but it might transfer differently. **Method** (`weight_edit.py`): a single-layer, batched least-squares edit in the style of ROME/MEMIT.

- **Key and value.** For each training failure, the key is the input to one layer's MLP `down_proj` at the question's last token, in the exact eval prompt. A change to that layer's output at that position is optimized so the model emits the correct answer.
- **The update.** One update ΔW = R Kᵀ(λC + KKᵀ)⁻¹ applies all of those changes at once. C is the key covariance over 3000 TriviaQA questions from no eval set.
- **Precision and baseline.** The candidate layers' `down_proj` weights are dequantized to fp32. The unedited model with that change is the baseline, and every run in this phase shares the same batched generation.
- **Why a new baseline.** A smoke test found that dequantizing alone changes 7 of 32 answers, and batching 1 of 8. That's finding 17 again, so phase 5 compares only against its own baseline.

**Tuning on the phase 3 split** (its 129 corrections). Layers {4, 8, 12, 16} × λ {10, 100, 1000, 10000}, scored with cheap proxies:

- *edit*: greedy would emit the answer on the edited question
- *para*: the same on its paraphrase
- *drift*: share of held-out questions whose first answer token changes

**Selection rule:** highest *para* among configurations with *edit* ≥ 80% and *drift* ≤ 5% (ties go to lower drift). If none qualifies, take the highest *edit* with drift ≤ 5% and report that the rule failed.

**One-shot test on the phase 4 split** (its 119 corrections), with the selected configuration. Arms:

- **edited**
- **placebo edit**: the same 119 questions, each edited toward *another* question's answer (fixed derangement). It has the same keys and the same procedure, but wrong facts.

Predictions:

1. **Training:** the edit fixes ≥ 60% of the 119 failures. That's below retrieval's 100%, because a shared least-squares update can't satisfy every fact exactly.
2. **Paraphrase:** among paraphrases wrong without memory whose source was edited, the edit fixes 25–60%, less than gated retrieval (which fixed 98). The key is a whole-question representation at "?", and a rewording moves it.
3. **Held-out:** the edit breaks ≥ 3 held-out answers (it can't be gated), and **the placebo edit breaks about as many** (within ± 5). Collateral comes from moving the weights, not from which facts were written.
4. **Placebo on its own targets:** on the 119 edited training questions, it makes the model emit the *shuffled* answer for ≥ 50%. The edits are specific, not generic damage.
5. **Overall:** weight editing doesn't beat gated retrieval in accuracy on any of the three sets.

**Deviations from this design, made on the dev split before the test split was touched** (details in NOTES.md, "Phase 5 dev sweep 1"):

- **The covariance was wrong.** The first sweep was stopped after 2 layers (edit 5–28%, drift 16–30% at every λ).
  - The original C left out the shared instruction prefix, and the update rewrote it (2× ‖m‖ on average).
  - More importantly, all edited keys sit at the "?" token and share a large common direction, so each edit spilled ≈ 1× ‖m‖ onto every unseen question's "?".
  - C is now (E_all[kkᵀ] + E_pos[kkᵀ]) / 2: all prompt tokens, plus the last-question-token keys of the 3000 covariance questions.
- **Weight decay** on the delta raised from 0.01 to 0.5 (the ROME default). Deltas had been 1.5–3.4× ‖m‖.
- **λ grid widened** to {1, 10, 100, 1000, 10⁴, 10⁵}.
- **Drift is judged against a null floor.** A random weight change of relative size 10⁻⁶ flips 12–18% of held-out first tokens, against 0% with no change: finding 17, likely amplified by LLM.int8's activation re-quantization. So "drift ≤ 5%" can't be met by any edit.
  - The rule now reads: drift ≤ (mean drift of 3 null perturbations at that layer) + 5 points.
  - Everything else in the rule is unchanged, including the fallback.
  - Test predictions 1–5 are unchanged. Prediction 3 already compares against the placebo edit, which carries the same floor.

## Phase 5 — results

### Tuning (phase 3 split, proxies)

Edit success is teacher-forced exact match on the 129 edited questions. The drift floor is the mean held-out first-token drift under 3 random weight changes of size 10⁻⁶.

| layer | floor | best edit (λ) | its para | its drift (excess) | edit at the drift limit (λ) |
|---|---|---|---|---|---|
| 4 | 10.9% | 27.9% (10⁴) | 6.6% | 20.8% (+9.9) | 10.1% (10⁵) |
| 8 | 11.3% | 56.6% (1) | 17.2% | 22.0% (+10.7) | 15.5% (10⁵) |
| 12 | 9.6% | 75.2% (10) | 18.0% | 16.0% (+6.4) | **59.7% (10⁴)** |
| 16 | 7.9% | 75.2% (1) | 29.5% | 18.4% (+10.5) | 25.6% (10⁵) |

- **The rule failed.** No configuration reaches edit ≥ 80%, so the pre-registered fallback picked **layer 12, λ = 10⁴**: the highest edit with drift ≤ floor + 5. The full table is in `runs/gemma3-4b-edit/selection.txt`.
- **Depth helps.** Edit (about 25 → 55 → 70–75%) and para (7 → 15 → 18 → 30%) both rise with depth.

### One-shot test (phase 4 split, 119 corrections)

Each arm is compared with the unedited run in the same pipeline.

| set | arm | accuracy (unedited → arm) | fixed | broke |
|---|---|---|---|---|
| train | **edited** | 53.6% → **87.6%** | 89 / 116 | **4** / 134 |
| train | placebo | 53.6% → 48.0% | 7 / 116 | 21 / 134 |
| paraphrase | **edited** | 46.0% → 51.5% | 19 / 127 | 6 / 108 |
| paraphrase | placebo | 46.0% → 43.0% | 5 / 127 | 12 / 108 |
| held-out | **edited** | 46.4% → 44.0% | 6 / 134 | **12** / 116 |
| held-out | placebo | 46.4% → 45.6% | 7 / 134 | 9 / 116 |

Against gated retrieval (phase 4), each with its own memory-off baseline:

| set | weight edit | gated retrieval |
|---|---|---|
| the 119 corrected questions | 93 / 119 right (78.2%) | 119 / 119 (100%) |
| paraphrases of edited questions, wrong without memory | 13 / 101 fixed (12.9%) | 98 / 101 (97.0%) |
| held-out | 12 broken, 6 fixed (44.0%, from 46.4%) | 0 broken, 0 fixed (44.0% = off) |

### Predictions, checked

1. **Training fixes ≥ 60%: holds.** 93 / 119 (78.2%) are right after the edit. Of the 111 wrong in this pipeline's unedited run, 86 are fixed and 1 breaks. Every fix contains the canonical corrected answer.
2. **Paraphrase 25–60%, fewer than gated retrieval: fails on the low side.** Only 13 / 101 (12.9%), against 98 / 101 for gated retrieval.
3. **Edit breaks ≥ 3 held-out answers, and the placebo about as many: holds.** 12 against 9 (+3).
4. **Placebo emits its shuffled answer on ≥ 50%: fails badly.** 2 / 119 (1.7%). Only 9 / 119 even start with the target's first token, against 0 unedited.
5. **The edit doesn't beat gated retrieval on any set: holds.** It's worse on train and paraphrase, and ties on held-out accuracy only because its 12 breaks and 6 fixes net to −6 against an unedited baseline 2.4 points higher.

### Findings

18. **A single-layer edit nearly matches retrieval on the exact questions, and barely transfers.**
    - It fixes 86 of 111 failures and breaks only 4 of 134 right training answers.
    - Paraphrases of the same questions get 13 / 101. The key is the "?" token's representation of the exact wording, so the edit is a lookup keyed even more narrowly than retrieval. Retrieval matches rewordings through the embedding model (97%).
19. **The edit "writes" true facts and not false ones: it amplifies what the model nearly knows.**
    - The placebo uses the same keys and procedure and converges just as well per fact (83% of deltas reach NLL < 0.1, against 97% for the true facts).
    - The batched solve realizes the same 73% of every delta in both arms (median projection).
    - But false facts need deltas twice as large (median ‖δ‖ 6.8 against 3.4; ‖ΔW‖_F 44.8 against 21.9). 73% of a push toward a true answer is enough, and 73% of a push toward a false one isn't: 2 / 119 installed.
    - So at a regularization strong enough to keep collateral near the floor, the edit mostly pulls out answers the model already half-knows. It isn't writing arbitrary new facts.
20. **Collateral is real, can't be gated, and sits near what any change of this size does.**
    - The edit breaks 12 of 116 right held-out answers. The placebo, with twice the ΔW, breaks 9; on train and paraphrase it breaks more than the edit (21 and 12, against 4 and 6).
    - Gated retrieval breaks 0, because it changes nothing when no stored question is close. A weight edit is always on.
21. **Method lessons** (dev split; NOTES.md, "Phase 5 dev sweep 1" and "2"):
    - The covariance must cover the shared prompt and the key position. Without them the edit rewrote every prompt (20–30% drift).
    - This 8-bit model flips 10–15% of first tokens under *any* weight change, so collateral needs a null or placebo.
    - Teacher-forced exact match understates generation success: 59.7% on dev against 78.2% at test.

### What phase 5 means

On this model and task, gated retrieval beats a closed-form weight edit on every measure:

| measure | retrieval | weight edit |
|---|---|---|
| failures fixed | 100% | 78% |
| paraphrases fixed | 97% | 13% |
| held-out answers broken | 0 | 12 |

The edit's advantages are an unchanged prompt and no retrieval step at run time. Its result is also narrower than "the model learned the fact". It mostly unlocks answers Gemma half-knew, keyed to the exact question wording. Whether either method gives knowledge the model can *use* is phase 6's question.

### Phase 5 caveats

- **One method, one layer, one λ.** The λ was chosen by the fallback rule, because the selection rule failed. Multi-layer MEMIT spreads each edit over several layers, which shrinks the per-layer change. It might transfer and collateralize differently.
- **The keys come from the exact eval prompt.** MEMIT averages keys over random prefixes for robustness. Doing that here might help paraphrases.
- **The placebo isn't size-matched.** Its ΔW is 2× the edit's, so "about as many broken" is conservative for the edit.
- **Different pipelines.** The dequantized-fp32 pipeline has its own baseline (53.6 / 46.0 / 46.4%), so retrieval's numbers come from the int8 pipeline. Each is compared only with its own memory-off run.
- **The test set is smaller than it looks.** 111 failures, 101 eligible paraphrases and 116 right held-out answers. Differences of a few answers (12 against 9) are within noise.

## Phase 6 design — can the model *use* a corrected fact? (pre-registered 2026-11-19, before any result)

Every fix so far has been a copy of the stored answer: the model was asked for exactly the fact it was given. Phase 6 asks for a *different* fact that needs the corrected one as an intermediate step.

**Questions** (`build_twohop.py`, written by Nemotron over the API, not by Gemma). For each of the 119 phase 4 corrections (question Q, corrected answer A):

| kind | example (A = Emily Dickinson) | answer |
|---|---|---|
| two-hop | "In which Massachusetts town was the 19th century American poet who wrote over 1,000 poems published between 1890 and 1945 after her death born?" | Amherst |
| direct | "In which Massachusetts town was Emily Dickinson born?" | Amherst |

- The two-hop question refers to A only through Q's description.
- Items are rejected automatically if the two-hop question names A or its own answer, the direct question doesn't name A, or the new answer is A.

**Arms:**

- **Retrieval** (int8 pipeline, phase 4 correction store), each against its own memory-off run:
  - *ungated* corrections;
  - *gated* (0.7);
  - *oracle*: retrieval by the source question Q, so the right correction is always in the prompt. This separates "can it use the fact" from "does retrieval find it".
- **Weights** (dequantized pipeline): the phase 5 selected configuration and its test deltas, as *edited* and *placebo*, each against its own unedited run.

**Primary set:** items Gemma can finish but doesn't start, in each pipeline's own memory-off run. That means the direct question is answered right and the two-hop question wrong. A fix is a target item answered right. The usual fixed/broke counts on all items are reported too.

**Predictions:**

1. **Solvability:** memory off, Gemma answers ≥ 50% of the direct questions right.
2. **The gate finds the fact:** on two-hop questions, the top-1 retrieved correction is the item's own and has cos ≥ 0.7 for ≥ 50% of items. The two-hop questions reuse Q's description.
3. **Oracle retrieval composes:** with its correction in the prompt, the model fixes ≥ 40% of target items. A 4B model can do one hop in context once the bridge entity is stated.
4. **Gated ≤ oracle:** gated retrieval fixes no more target items than oracle, and at least half as many.
5. **Weight edits don't compose:** the edit fixes ≤ 10% of target items, and no more than the placebo edit + 5 points. This matches multi-hop failures of ROME/MEMIT edits reported in the literature (MQuAKE).
6. **Copying shows up as a failure mode:** in the retrieval arms, ≥ 20% of the target items that stay wrong are answered with the bridge entity A itself. The model copies the stored answer instead of using it.

## Phase 6 — results

There are 114 items; 5 more were rejected automatically. Target items are those where Gemma answers the direct question right and the two-hop question wrong with memory off. There are 47 in the retrieval pipeline and 43 in the weights pipeline.

| pipeline | arm | two-hop accuracy (all 114) | target items fixed |
|---|---|---|---|
| retrieval | off | 31.6% | — |
| retrieval | ungated corrections | 46.5% (fixed 23, broke 6) | 19 / 47 (40.4%) |
| retrieval | gated 0.7 | 46.5% (fixed 23, broke 6) | 19 / 47 (40.4%) |
| retrieval | oracle | 49.1% (fixed 24, broke 4) | **20 / 47 (42.6%)** |
| weights | off | 32.5% | — |
| weights | edited (L12, λ 10⁴) | 31.6% (fixed 2, broke 3) | **1 / 43 (2.3%)** |
| weights | placebo | 27.2% (fixed 4, broke 10) | 3 / 43 (7.0%) |

### Predictions, checked

1. **Solvability ≥ 50%: holds.** The direct questions score 67.5% in the int8 pipeline and 65.8% dequantized. They score 64.9% and 63.2% under the edited and placebo weights.
2. **The gate finds the fact on ≥ 50%: holds.** The item's own correction is top-1 with cos ≥ 0.7 on 105 / 114 (92.1%), and top-1 at any similarity on 113.
3. **Oracle fixes ≥ 40% of targets: holds, narrowly.** 20 / 47 (42.6%).
4. **Oracle/2 ≤ gated ≤ oracle: holds.** 19 against 20.
5. **The edit fixes ≤ 10% and ≤ placebo + 5 points: holds.** 1 / 43, against the placebo's 3 / 43.
6. **≥ 20% of the targets still wrong under retrieval are answered with A: holds, narrowly.** 6 / 28 (21.4%) ungated and gated, 6 / 27 oracle, against 1 of the same items with memory off.

### Findings

22. **With the fact in the prompt, Gemma completes the second hop in about 4 of 10 cases.**
    - Oracle fixes 20 / 47 targets, and gated retrieval nearly matches it (19). The two-hop questions reuse the original description, so the gate fires on 92%. On these questions retrieval is effectively the oracle.
    - On all 114 items retrieval fixes 23–24 and breaks 4–6.
23. **The weight edit's facts can't be used: Gemma "knows" A when asked the original question, knows A → B, and still can't chain them.**
    - In 34 of the 43 weight-pipeline targets, the edited model answers the source question right *and* Gemma answers the direct question right. The two-hop question is fixed in **1** of those 34. The placebo, which installs no facts, fixes 3.
    - This fits finding 18: the edit is keyed to the "?" representation of the exact question wording, and a question that merely contains the description doesn't trigger it.
24. **Composition, not knowledge, is the bottleneck for retrieval.**
    - Of the 27 targets oracle leaves wrong, 6 answer the bridge entity itself (copying: "Sweden" for the capital of Sweden).
    - The other 21 give a wrong second hop even though the direct version is answered right in the same pipeline: Switzerland → "Zurich", Numb3rs → "Chicago", Priam → "Helen".
    - The model holds both hops separately but chains them only about 43% of the time, even with the bridge stated in the prompt.

### What phase 6 means

- **Retrieval gives working memory.** It puts the corrected fact where the model can sometimes reason with it, up to the limit of a 4B model's one-hop composition (about 43%).
- **The weight edit gives a lookup.** It's tied to a specific question, and the fact behind it isn't available as knowledge: 1 / 34 even when both hops are known.

So on this model and task, neither method changes what the model *knows* in a usable sense, and retrieval is better on every measure tested in phases 4–6.

### Phase 6 caveats

- **Small numbers:** 47 and 43 target items. P3 and P6 hold by 1–2 items.
- **Generated questions:** Nemotron wrote them, and the filters don't catch every factual or wording error. The direct-question filter removes unanswerable ones but not ambiguous two-hop phrasing (e.g. "the town … where the TV series is set").
- **Grader near-misses:** e.g. "First" for "First Amendment" counts as wrong.
- **Not a test of retrieval robustness:** the two-hop questions reuse the source description, which is why the gate fires 92% of the time.
- **One weight configuration:** the phase 5 selection. Multi-layer MEMIT and keys averaged over random prefixes are untested. MEMIT-style edits are known to compose poorly too (MQuAKE).

## Phase 7 design — does naming the entity first help composition? (pre-registered 2026-11-19, before any run)

Phase 6's oracle fixes 20 / 47 two-hop targets. Of the 27 it misses, 6 answer the bridge entity itself and 21 give a wrong second hop. Phase 7 changes only the instruction:

> Answer the question below. The question may describe a person, place or thing without naming it. First name what it describes, then answer the question that is actually asked. Reply in exactly this format, and nothing else:
> Entity: \<name\>
> Answer: \<short answer\>

Only the text after the last "Answer:" is graded; the full reply is stored. The arms are on the same 114 two-hop questions, int8 pipeline, phase 4 correction store: **off_chain, gated_chain, oracle_chain**. The target items are phase 6's 47, defined by the original prompt's memory-off runs. Each chain arm is compared with the matching phase 6 arm.

**Predictions:**

1. **Chain alone doesn't supply the fact:** off_chain fixes ≤ 10% of targets. Gemma got the source question wrong, so naming "the entity" mostly names the wrong one.
2. **Chain helps when the fact is there:** oracle_chain fixes ≥ 55% of targets, against oracle's 42.6%.
3. **Copying drops:** among targets still wrong under oracle_chain, at most 2 answer the bridge entity itself (oracle: 6).
4. **Gated tracks oracle:** gated_chain is within 2 targets of oracle_chain, because the gate fires on 92% of these questions.
5. **No new harm:** on the 36 two-hop questions right with memory off, oracle_chain breaks no more than oracle + 2 (oracle broke 4).

## Phase 7 — results

Same 114 two-hop questions and the same 47 targets as phase 6. Chain arms are graded on the text after "Answer:"; every reply had that line.

| arm | two-hop accuracy (114) | targets fixed (of 47) | right-off broken (of 36) | still-wrong targets answering the bridge |
|---|---|---|---|---|
| oracle (phase 6) | 49.1% | 20 (42.6%) | 4 | 6 / 27 |
| gated (phase 6) | 46.5% | 19 (40.4%) | 6 | 6 / 28 |
| off_chain | 23.7% | 6 (12.8%) | 16 | 2 / 41 |
| gated_chain | 37.7% | 17 (36.2%) | 15 | 5 / 30 |
| oracle_chain | 40.4% | **23 (48.9%)** | **18** | 5 / 24 |

### Predictions, checked

1. **off_chain fixes ≤ 10%: fails, narrowly.** 6 / 47 (12.8%).
2. **oracle_chain fixes ≥ 55%: fails.** 23 / 47 (48.9%), 3 more than oracle.
3. **≤ 2 still-wrong targets copy the bridge: fails.** 5, against oracle's 6.
4. **gated_chain within 2 of oracle_chain: fails.** 17 against 23.
5. **No new harm (≤ 6 breaks): fails badly.** oracle_chain breaks 18 of the 36 two-hop questions Gemma answers right without memory. off_chain alone breaks 16, so the prompt causes the harm, not the memory.

### Findings

25. **Asking the model to name the entity first costs more than it gains.** oracle_chain fixes 3 more targets than oracle, and it breaks 14 more right answers.
    - The Entity line often isn't the bridge. Under oracle_chain it names the bridge on only 20 / 47 targets, even with the correction in the prompt. Sometimes it holds the final answer instead (19 / 114 replies), or the two slots are swapped (2 / 114: "Entity: Minsk / Answer: Belarus").
    - Scoring post hoc as correct whenever the expected answer appears anywhere in the reply gives oracle_chain 27 / 47 targets. It still breaks 12 / 36.
    - So the two-step format is a burden for a 4B model, and it doesn't make the hop more reliable.
26. **The composition ceiling stands.** With the fact in context, the best prompt tested completes the second hop for about half of targets: 20–23 / 47 strictly, 27 leniently. The original one-line prompt is the better trade.

### Phase 7 caveats

- **One instruction, not tuned.** Few-shot examples of the format might fix the slot confusion. That wasn't tried, to avoid tuning on the test set.
- **Grading on the "Answer:" line** penalizes replies with the answer in the wrong slot. The lenient rescoring is reported, but it's post hoc.

## Phase 8 design — near-duplicate questions with different answers (pre-registered 2026-11-19, before any run)

Phase 6 showed the 0.7 gate fires on 92% of questions that reuse a stored question's description. That's good when the answer is related, and harmful when a question reuses the wording but asks about something else.

**Questions** (`build_neardup.py`, Nemotron): for each phase 4 correction (Q, A), a question Q′ that changes as few words of Q as possible but has a different answer B. For example, "…American **author** wrote over 1,000 **letters**…" → Mark Twain. Items are rejected if B matches A, Q′ contains B, or Q′ = Q.

**Arms** (int8 pipeline, phase 4 correction store): off, ungated corrections, gated 0.7.

**Primary measure:** *copy harm*, the share of near-duplicates answered with the stored answer A. It doesn't depend on whether Nemotron's B is right. Fixed/broke against B are reported too, but B may sometimes be wrong.

**Predictions:**

1. **The gate fires on ≥ 50% of near-duplicates.** Their own source correction is top-1 with cos ≥ 0.7, because MiniLM similarity is dominated by shared wording.
2. **Copy harm ≥ 15% under gated retrieval** (answer contains A), against ≤ 3% memory off.
3. **Gated ≈ ungated:** copy harm under gated is within 3 items of ungated, since the gate fires on most items.
4. **Breakage:** among near-duplicates answered right memory off, gated retrieval breaks ≥ 15%.
5. **No threshold separates the cases.** The threshold would have to keep a correction for phase 4 paraphrases of the same question (the case where it should apply) and block it for near-duplicates. No single threshold keeps ≥ 90% of phase 4 paraphrases at or above it while keeping ≥ 90% of near-duplicates below it.

## Phase 8 — results

118 near-duplicates (1 rejected), int8 pipeline, phase 4 correction store. Scoring: `analyze_phase78.py 8` → `runs/gemma3-4b-neardup/analysis.md`.

| arm | accuracy against B (118) | copy harm: answer contains the stored A | fixed (of 58 wrong off) | broke (of 60 right off) |
|---|---|---|---|---|
| off | 50.8% | 3 (2.5%) | – | – |
| corrections, ungated | 43.2% | 15 (12.7%) | 8 | 17 |
| gated 0.7 | 42.4% | **16 (13.6%)** | 3 | **13** |

**Not all items are near.** Nemotron changed more wording than asked on many items. Their median MiniLM cosine to the source question is 0.718, against 0.930 for phase 4 paraphrases, and 55 / 118 fall below the 0.7 gate. The gate fired on exactly the other 63, and every time its top-1 was the item's own source correction. Copy harm by similarity, gated arm:

| cos to source question | items | gate fired | copy harm |
|---|---|---|---|
| < 0.6 | 38 | 0 | 1 (also copied with memory off) |
| 0.6–0.7 | 17 | 0 | 0 |
| 0.7–0.8 | 22 | 22 | 2 |
| 0.8–0.9 | 18 | 18 | 5 |
| ≥ 0.9 | 23 | 23 | 8 |

### Predictions, checked

1. **The gate fires on ≥ 50%: holds, narrowly.** 63 / 118 (53.4%), always with the item's own source correction top-1.
2. **Copy harm ≥ 15% gated, ≤ 3% off: fails, narrowly.** 16 / 118 (13.6%) gated, 3 / 118 (2.5%) off. Among the 63 items where the gate fired, it is 15 / 63 (23.8%).
3. **Gated within 3 of ungated: holds.** 16 against 15.
4. **Gated breaks ≥ 15% of right-off: holds.** 13 / 60 (21.7%). 7 of the 13 answer with the stored A.
5. **No threshold separates paraphrases from near-duplicates: holds.** The best threshold is 0.88, which keeps 75.7% of 115 phase 4 paraphrases and blocks 75.4% of near-duplicates. Nothing reaches 90% / 90%.

### Findings

27. **The gate can't tell "the same question reworded" from "the same wording, a different question."** When a near-duplicate clears 0.7, Gemma answers with the stored answer on 15 / 63 (24%). The rate rises with similarity, to 8 / 23 above 0.9. Examples:
    - "Bronze is an alloy of what 2 metals?" → "Copper and Zinc", the stored answer for brass.
    - "Which company uses the advertising slogan 'Save Money, Live Better'?" → ALDI (expected Walmart).
    - "What is the name of the vast realm where the Norse gods live?" → Valhalla (expected Asgard).

    Phase 4's "0 broken out of 735" holds for questions unrelated to the store. Near-duplicates are a separate failure, and the gate doesn't cover it: 13 of the 60 near-duplicates Gemma answers right without memory break.
28. **No cosine threshold fixes it.** Paraphrases and near-duplicates overlap in MiniLM similarity. Raising the gate from 0.7 to its best value, 0.88, would block 3 in 4 near-duplicates and also lose about 1 in 4 paraphrase fixes. Separating the two cases needs more than surface similarity, for example a check that the stored answer fits the new question.

### Phase 8 caveats

- **Nemotron's answers B aren't verified.** Copy harm doesn't depend on them, but accuracy, fixed and broke do. 3 items get the stored answer even with memory off, because Gemma's own wrong answer coincides with it (for example Joan Crawford for "Which Bette's career revived…?").
- **55 / 118 items aren't near duplicates** by the gate's own measure. The sample that actually tests the gate is the 63 that cleared it.

## Phase 9 design — can a better edit be less of a lookup? (pre-registered 2026-11-19, before any run)

Phase 5's single-layer edit transferred to 13% of paraphrases and 1 / 43 two-hop targets. Phase 9 tests two standard improvements (`memit.py`).

**Method:**

- **Multi-layer MEMIT** over layers {8, 10, 12, 14, 16}:
  - One residual-stream change at the question's last token, after layer 16, is optimized per fact. It uses MEMIT's GPT-J settings: clamp 0.75 ‖h‖, weight decay 0.5.
  - The layers are edited in order. Before each one, the model is re-run with the earlier edits in place, and that layer supplies 1 / (layers left) of the residual change still missing.
  - Gemma's post-feedforward RMSNorm is inverted numerically for each target.
  - Each layer's solve is phase 5's, with the `cov_mix` covariance.
- **Arms:**
  - *memit*: one key per fact.
  - *memit_aug*: the same, plus 2 automatic rewordings per question as extra keys with the same target. The rewordings come from a different prompt than the test paraphrases (`build_rewordings.py`); any rewording identical to the test paraphrase is rejected.
- **No placebo arm** this time, to save GPU time. Collateral is judged against the null floor and phase 5's numbers.

**Tuning** (phase 3 split, the phase 5 proxies):

- λ ∈ {10², 10³, 10⁴, 10⁵} per arm.
- Floor: mean held-out drift under 3 random 10⁻⁶ changes to all 5 layers.
- Selection: phase 5's rule (edit ≥ 80% and drift ≤ floor + 5, best para; otherwise the fallback).
- An arm with no λ inside the drift limit is not tested, and that is reported.

**Test** (phase 4 split): train, paraphrase and held-out, plus phase 6's two-hop and direct sets, against this pipeline's own unedited run (its dequantized layers differ from phase 5's).

**Predictions:**

1. **Spreading lowers collateral:** memit breaks ≤ 8 held-out answers (phase 5: 12), with ≥ 70% of the 119 corrected questions right.
2. **Spreading alone doesn't fix the lookup:** memit fixes ≤ 25% of wrong-off paraphrases of edited questions (phase 5: 12.9%).
3. **Reworded keys help:** memit_aug fixes ≥ 35% of them, at least 15 points more than memit.
4. **Neither arm composes:** each fixes ≤ 10% of two-hop targets (direct right, two-hop wrong, in this pipeline's unedited run).
5. **Neither arm beats gated retrieval** on train, paraphrase or held-out accuracy.

**Deviation, made before any phase 9 dev or test result** (NOTES.md, "Phases 8–9 — API builds and reaper"):

- **What failed:** MEMIT's residual-stream target can't be delivered in Gemma 3. Every MLP output passes through a post-feedforward RMSNorm, whose output size is bounded, while the residual stream after layer 16 has norms in the tens of thousands. The 3-fact smoke test needed per-layer residual steps of 4,000–20,000, and the norm inversion had a relative error of about 0.9.
- **Replacement:** each fact gets one change per layer, on the down_proj outputs of all 5 layers. They are optimized jointly, with weight decay summed over the layers so the change is shared out. The layers are then solved in order, as before.
- **Unchanged:** the arms, tuning, selection rule and predictions.

## Phase 9 — results (dev sweep only: neither arm reached the test)

Dev sweep on the phase 3 split: 129 facts; held-out drift on 250 questions; `memit.py dev` → `runs/gemma3-4b-memit/dev_sweep.csv`, log `runs/gemma3-4b-memit-dev.log`.

- **Floor:** 9.2%, the mean of 3 random 10⁻⁶ changes to all 5 layers (10.4%, 9.2%, 8.0%).
- **Drift limit:** floor + 5 = 14.2%.

| arm | λ | edit | para | held-out drift | train-right drift |
|---|---|---|---|---|---|
| memit | 10² | 68.2% | 27.9% | 25.6% | 15.7% |
| memit | 10³ | 60.5% | 29.5% | 30.0% | 15.7% |
| memit | 10⁴ | 77.5% | 32.0% | 21.2% | 9.1% |
| memit | 10⁵ | 39.5% | 20.5% | 16.4% | 6.6% |
| memit_aug | 10² | 77.5% | 64.8% | 23.6% | 14.0% |
| memit_aug | 10³ | 73.6% | 67.2% | 23.2% | 15.7% |
| memit_aug | 10⁴ | 79.1% | 63.1% | 24.8% | 14.9% |
| memit_aug | 10⁵ | 70.5% | 54.9% | 19.2% | 10.7% |
| *phase 5, L12 (the tested edit)* | 10⁴ | 59.7% | 15.6% | 13.6% (floor 9.6%) | 8.3% |
| *phase 5, L16* | 10³ | 72.1% | 32.0% | 17.6% (floor 7.9%) | 9.1% |

**Outcome: the pre-registered rule stops both arms.** No λ keeps held-out drift within 14.2%. The lowest values are 16.4% (memit at λ 10⁵, where edit success has fallen to 39.5%) and 19.2% (memit_aug at λ 10⁵). As pre-registered, an arm with no λ inside the limit isn't tested. **Predictions 1–5 therefore have no test result.** The λ grid wasn't extended after seeing this, because that would mean tuning past the pre-registered plan.

(`memit.py select` first printed the floor as "nan%": pandas reads the arm name "null" as a missing value. That is fixed, and with the right floor the outcome is the same, since no row was within any limit below 16.4%.)

**What the dev numbers point to** (dev proxies, not scored as predictions):

- **Collateral rose rather than fell** (against prediction 1's direction). The best memit setting drifts 21.2%, against phase 5 L12's 13.6%. λ barely controls it: across 10²–10⁴, drift stays at 21–30% while edit success holds. At 10⁵ the edit collapses before drift reaches the limit.
- **memit alone transfers to 32% of paraphrases** (prediction 2 said ≤ 25%). Phase 5's single L16 edit already reached that on dev (32.0%), so spreading over layers adds about what editing a later layer adds.
- **Reworded keys roughly double transfer** (prediction 3's direction): memit_aug 55–67% against memit 21–32%. But the rewordings are near-copies of the paraphrases (MiniLM cosine):

  | | cos(rewording, test paraphrase), median | the closer of each question's 2 | share ≥ 0.95 | cos(paraphrase, original), median |
  |---|---|---|---|---|
  | dev (`rewordings_trivia.csv`, 244) | 0.942 | 0.960 | 45.1% | 0.945 |
  | test (`rewordings_trivia_v2.csv`, 229) | 0.923 | 0.936 | 30.1% | 0.930 |

  The extra keys sit about as close to the paraphrases as the paraphrases sit to the original. The gain is coverage of wording, the same mechanism as retrieval.

### Findings

29. **Spreading the edit over 5 layers raised collateral in Gemma 3 4B, and λ can't bring it down without killing the edit.** Every setting drifts 16–30% on held-out questions, against a 9.2% floor. A likely cause, not tested: each fact's change is up to 4 ‖m‖ at each of 5 layers, and it spills onto the prompt's other tokens. Phase 5's diagnosis found that spillover was the main source of drift, and the covariance only constrains directions common in ordinary text.
30. **Extra keys buy paraphrase coverage the way retrieval does, not a more usable fact.** Two automatic rewordings per question double dev paraphrase transfer, but they nearly duplicate the paraphrases. Whether any of this helps two-hop use wasn't measured, because no setting passed the drift rule.

### What phases 5 and 9 mean together

Both weight-edit attempts end in the same place: a lookup keyed on wording, with collateral on unrelated questions that can't be gated. Retrieval with the 0.7 gate remains the better store for corrected facts in this model. Its known weak spot is near-duplicate questions (phase 8).

### Phase 9 caveats

- **Not standard MEMIT.** Gemma's post-norm made residual targets infeasible (see the deviation above), so this is per-layer down_proj targets solved in sequence. A residual-target MEMIT on a model without post-norm could behave differently.
- **4-point λ grid, fixed in advance.** Larger λ, a smaller clamp, or editing at the subject token weren't tried.
- **Dev size:** 129 facts. Held-out drift is on 250 questions, so its standard error is about ±2.5 points.

## Where this goes next

This experiment is one slice of a longer arc. Done so far:

- the two-track experiment below ([Phase 3](#phase-3--two-track-memory-corrections-vs-lessons));
- the gate replication and placebo test ([Phase 4](#phase-4--results));
- single-layer weight editing ([Phase 5](#phase-5--results));
- the two-hop fact-use test ([Phase 6](#phase-6--results));
- the chain prompt ([Phase 7](#phase-7--results));
- the near-duplicate gate stress test ([Phase 8](#phase-8--results));
- multi-layer editing, stopped at dev ([Phase 9](#phase-9--results-dev-sweep-only-neither-arm-reached-the-test)).

Candidates now (none started; each needs a go-ahead):

- **A gate that checks fit, not just similarity.** Phase 8's near-duplicates are retrieval's one demonstrated harm. Options: a cross-encoder or the model itself judging "does this stored correction answer this question?", or an answer-type check. Score on phase 8's 63 gated items (should block) and phase 4's paraphrases (should keep).
- **A cleaner near-duplicate set.** Keep only items with cos ≥ 0.8 to their source, and check B by a second model, so the gate question rests on more than 63 items.
- **Weight editing, if it's worth one more try:** edit at the subject's last token instead of "?", with a smaller clamp and λ above 10⁵. That would be a new pre-registered sweep. Phases 5 and 9 both point to a wording-keyed lookup with ungateable collateral, so this has the lowest expected value of the three.

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

Phase 4:

- `run_phase4.py` (placebo + fresh-split gate, resumable); gate and placebo modes live in `agent.py`
- `questions_trivia_v2.csv`, `questions_trivia_v2_holdout.csv`, `questions_trivia_v2_paraphrase.csv`: fresh 250 / 250 / 235 (seed 1)
- `runs/gemma3-4b-phase4/`: fresh `corrections.csv` (119), `{train,paraphrase,holdout}_{off,corrections,gated}.csv`, `*_pairs.csv`
- `runs/gemma3-4b-placebo/`: `holdout_placebo_{lorem,random}.csv` and pairs, against the phase 3 correction store

Phase 5:

- `weight_edit.py`: dequantize, covariance, deltas, the closed-form edit, the proxies (`dev`) and the one-shot test (`test`). Caches in `runs/gemma3-4b-edit/cache/` are gitignored, at 420 MB per covariance.
- `select_edit.py`: the selection rule, applied to `runs/gemma3-4b-edit/dev_sweep.csv`. Its output is in `selection.txt`.
- `analyze_phase5.py` → `runs/gemma3-4b-edit/analysis.md`
- `runs/gemma3-4b-edit/`:
  - `{train,paraphrase,holdout}_{off,edited,placebo}.csv` and `*_pairs.csv`;
  - `placebo_targets.csv`;
  - the logs `dev_sweep1.log`, `dev.log`, `dev_L12_16.log`, `dev_L16_1e5.log`, `test*.log`.

Phase 6:

- `build_twohop.py` → `questions_twohop.csv`, `questions_twohop_direct.csv` (114 items), `questions_twohop_rejected.csv` (5)
- `run_phase6.py` (retrieval and weights arms, resumable); `analyze_phase6.py` → `runs/gemma3-4b-twohop/analysis.md`
- `runs/gemma3-4b-twohop/` (retrieval arms, int8) and `runs/gemma3-4b-twohop-edit/` (weights arms, dequantized)

Phases 7 and 8:

- `run_phase78.py` (resumable); `analyze_phase78.py [7] [8]` → `runs/gemma3-4b-chain/analysis.md`, `runs/gemma3-4b-neardup/analysis.md`
- `agent.py`: `CHAIN_INSTRUCTION`, `build_prompt(style="chain")`, `final_answer()`; `evaluate.py --style chain`, `--retrieve-by`
- `runs/gemma3-4b-chain/`: `twohop_{off,gated,oracle}_chain.csv`; log `runs/gemma3-4b-phase78.log`
- `build_neardup.py` → `questions_neardup.csv` (118), `questions_neardup_rejected.csv` (1)
- `runs/gemma3-4b-neardup/`: `neardup_{off,corrections,gated}.csv` and pairs, `neardup_sims.csv` (per-item cosine, gate, copy); log `runs/gemma3-4b-phase8.log`

Phase 9:

- `memit.py` (`dev`, `select`, `test`; design in its docstring); `weight_edit.Editor(layers=...)`
- `build_rewordings.py` → `rewordings_trivia.csv` (dev, 244) and `rewordings_trivia_v2.csv` (test, 229)
- `runs/gemma3-4b-memit/dev_sweep.csv`; logs `runs/gemma3-4b-memit-dev.log`, `runs/memit_smoke{,2,3}.log`
