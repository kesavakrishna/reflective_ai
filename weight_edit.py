"""Phase 5: write corrections into the weights instead of the prompt (MEMIT-style batched edit).

Retrieval (phases 3–4) makes the model *copy* a stored answer. This changes what the model computes:
for every training failure it finds a change to one MLP layer's output that makes Gemma produce the
correct answer, then solves for a single weight update to that layer's down_proj that produces all
of those changes at once while leaving other inputs as untouched as possible.

Method (one layer l, n facts):
  key   k_i = input to layers[l].mlp.down_proj at the last token of the question, in the eval prompt
  value m_i = W k_i, its output; delta_i is optimized so that m_i + delta_i makes the model generate
              the correct answer (teacher-forced NLL, weight decay, norm clamp)
  update  dW = R K^T (lam * C + K K^T)^-1,  R = [delta_i], K = [k_i], C = E[k k^T] over the keys of
              other TriviaQA questions (none from any eval set) — the MEMIT/ROME least-squares edit
Gemma 3 applies an RMSNorm after down_proj, so the edit targets down_proj's output directly (delta is
optimized through the norm) rather than the residual stream; that makes it exact for a single layer.

The model is the usual 8-bit Gemma with the down_proj of every CANDIDATE_LAYERS layer dequantized to
fp32 (an int8 or bf16 weight would round small updates away). That model, unedited, is the baseline:
edited and unedited runs share everything but dW, including batched generation.

  python weight_edit.py cov                      # key covariances for the candidate layers (cached)
  python weight_edit.py dev                      # tune layer / lam on the phase 3 split (proxies only)
  python weight_edit.py test --layer L --lam X   # one shot on the phase 4 split; writes runs/<name>/
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from agent import build_prompt  # noqa: E402
from grader import grade  # noqa: E402
from llm_client import load_local  # noqa: E402

CANDIDATE_LAYERS = (4, 8, 12, 16)
CACHE = os.path.join("runs", "gemma3-4b-edit", "cache")
TRIVIA_PARQUET = os.getenv(
    "TRIVIA_PARQUET",
    "C:/Users/kesav/AppData/Local/Temp/claude/a393efa983fe7735/eb1ace4e-d9a6-45d3-8ef3-7fd43e0a9bce/"
    "scratchpad/triviaqa_download/validation.parquet")
EVAL_FILES = ["questions_trivia.csv", "questions_trivia_holdout.csv", "questions_trivia_paraphrase.csv",
              "questions_trivia_v2.csv", "questions_trivia_v2_holdout.csv", "questions_trivia_v2_paraphrase.csv"]
N_COV_QUESTIONS = 3000
GEN_BATCH = 16
MAX_NEW_TOKENS = 64

# delta optimization (ROME/MEMIT defaults, adapted): steps, Adam lr on delta / rms(m), clamp ||delta||
# to CLAMP * ||m||, weight decay on (||delta|| / ||m||)^2, stop once every target token has p > 0.9.
STEPS, LR, CLAMP, WD = 40, 0.5, 4.0, 0.5  # WD was 0.01 in dev sweep 1 (deltas 1.5-3.4x ||m||)


class FP32Linear(torch.nn.Module):
    """A dequantized down_proj kept in fp32 so small edits survive; returns the input dtype."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.float(), requires_grad=False)

    def forward(self, x):
        return (x.float() @ self.weight.T).to(x.dtype)


class Editor:
    def __init__(self, layers=CANDIDATE_LAYERS):
        self.edit_layers = tuple(layers)  # down_projs dequantized to fp32 (phase 9 uses its own range)
        self.model, self.tok = load_local()
        self.model.requires_grad_(False)
        self.layers = self.model.model.language_model.layers
        self.dev = self.model.device
        self.base = {}
        for l in self.edit_layers:
            q = self.layers[l].mlp.down_proj
            w = q.weight
            W = w.CB.float() * w.SCB.float()[:, None] / 127.0  # LLM.int8 row-wise dequantization
            self.layers[l].mlp.down_proj = FP32Linear(W).to(self.dev)
            self.base[l] = self.layers[l].mlp.down_proj.weight.data.clone()
        torch.cuda.empty_cache()
        self.end_of_turn = self.tok.convert_tokens_to_ids("<end_of_turn>")

    # --- prompts -------------------------------------------------------------------------------
    def text(self, question):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(question, [], [])}],
            tokenize=False, add_generation_prompt=True)

    def encode(self, question):
        """Token ids of the eval prompt and the index of the question's last token."""
        text = self.text(question)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        end = text.rindex("\nAnswer:")
        pos = max(i for i, (s, _) in enumerate(enc["offset_mapping"]) if s < end)
        return enc["input_ids"], pos

    def target_ids(self, answer):
        return self.tok(str(answer), add_special_tokens=False)["input_ids"] + [self.end_of_turn]

    # --- hooks ---------------------------------------------------------------------------------
    def _forward(self, ids, layer=None, pos=None, delta=None, capture=None, stop_after=None):
        """One unpadded forward. Optionally add delta to down_proj's output at (layer, pos), capture
        down_proj inputs/outputs at {layer: list} and stop after layer `stop_after`."""
        handles = []

        class Stop(Exception):
            pass

        if capture:
            for l, store in capture.items():
                def hook(mod, inp, out, store=store):
                    store.append((inp[0][0].detach().float(), out[0].detach().float()))
                handles.append(self.layers[l].mlp.down_proj.register_forward_hook(hook))
        if delta is not None:
            def add(mod, inp, out):
                mask = torch.zeros(out.shape[1], 1, device=out.device, dtype=torch.float32)
                mask[pos] = 1.0
                return (out.float() + mask * delta).to(out.dtype)
            handles.append(self.layers[layer].mlp.down_proj.register_forward_hook(add))
        if stop_after is not None:
            def stop(mod, inp, out):
                raise Stop
            handles.append(self.layers[stop_after].register_forward_hook(stop))
        try:
            x = torch.tensor([ids], device=self.dev)
            return self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
        except Stop:
            return None
        finally:
            for h in handles:
                h.remove()

    # --- key covariance -----------------------------------------------------------------------
    def covariance(self, layer):
        """The keys the update must leave alone, as a second-moment matrix over N_COV_QUESTIONS TriviaQA
        questions that appear in no eval set: C = (E_all[k k^T] + E_pos[k k^T]) / 2.
          E_all: every token of the eval prompt, the shared instruction prefix included (an update that
                 moves the prefix moves every answer; dev sweep 1 left it out).
          E_pos: only the key position (the question's last token). Every edited key sits there and
                 those keys share a large common direction, so without this term each edit spills
                 ~1x ||m|| onto every unseen question's last token (dev sweep 1 diagnosis, NOTES.md).
        Cached per layer (420MB each)."""
        path = os.path.join(CACHE, f"cov_mix_L{layer}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location=self.dev)
        os.makedirs(CACHE, exist_ok=True)
        S = torch.zeros(10240, 10240, device=self.dev)
        Sp = torch.zeros(10240, 10240, device=self.dev)
        n, t0 = 0, time.time()
        qs = self._cov_questions()
        for i, q in enumerate(qs):
            ids, pos = self.encode(q)
            cap = {layer: []}
            with torch.no_grad():
                self._forward(ids, capture=cap, stop_after=layer)
            k = cap[layer][0][0]
            S += k.T @ k
            Sp += torch.outer(k[pos], k[pos])
            n += k.shape[0]
            if i % 1000 == 0:
                print(f"  cov L{layer} {i}/{N_COV_QUESTIONS}  {n} keys  {time.time() - t0:.0f}s", flush=True)
        C = (S / n + Sp / len(qs)) / 2
        del S, Sp
        torch.save(C, path)
        json.dump({"n_keys": n, "n_questions": N_COV_QUESTIONS}, open(os.path.join(CACHE, f"cov_mix_L{layer}.json"), "w"))
        print(f"  cov L{layer}: {n} keys -> {path}", flush=True)
        return C

    def _cov_questions(self):
        used = set().union(*(set(pd.read_csv(f)["question"].str.strip()) for f in EVAL_FILES if os.path.exists(f)))
        qs = pd.read_parquet(TRIVIA_PARQUET, columns=["question"])["question"].str.strip()
        qs = qs[~qs.isin(used) & (qs.str.len() <= 200)].drop_duplicates()
        return qs.sample(N_COV_QUESTIONS, random_state=2).tolist()

    # --- edits ---------------------------------------------------------------------------------
    def compute_delta(self, question, answer, layer):
        ids, pos = self.encode(question)
        tgt = self.target_ids(answer)
        full = ids + tgt
        cap = {layer: []}
        with torch.no_grad():
            self._forward(ids, capture=cap, stop_after=layer)
        k, m = cap[layer][0][0][pos], cap[layer][0][1][pos]
        scale = m.norm() / math.sqrt(m.numel())
        u = torch.zeros_like(m, requires_grad=True)
        opt = torch.optim.Adam([u], lr=LR)
        labels = torch.tensor(tgt, device=self.dev)
        sl = slice(len(ids) - 1, len(full) - 1)
        for step in range(STEPS):
            delta = u * scale
            logits = self._forward(full, layer=layer, pos=pos, delta=delta)[sl].float()
            logp = torch.log_softmax(logits, -1)
            nll = -logp.gather(1, labels[:, None]).mean()
            loss = nll + WD * (delta.norm() / m.norm()) ** 2
            if logp.gather(1, labels[:, None]).exp().min() > 0.9:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                d = u * scale
                if d.norm() > CLAMP * m.norm():
                    u.mul_(CLAMP * m.norm() / d.norm())
        return k, (u * scale).detach(), float(nll.detach()), step

    def deltas(self, corrections, layer, tag):
        """Optimized deltas for every correction, checkpointed every 10 so a killed run resumes."""
        path = os.path.join(CACHE, f"deltas_{tag}_wd{WD:g}_L{layer}.pt")
        part = path + ".partial"
        if os.path.exists(path):
            return torch.load(path, map_location=self.dev)
        os.makedirs(CACHE, exist_ok=True)
        D = torch.load(part, map_location=self.dev) if os.path.exists(part) else {"K": [], "R": [], "info": []}
        t0 = time.time()
        rows = list(corrections.itertuples())
        for i in range(len(D["K"]), len(rows)):
            r = rows[i]
            k, d, nll, steps = self.compute_delta(r.question, r.correct_answer, layer)
            D["K"].append(k)
            D["R"].append(d)
            D["info"].append((nll, steps))
            if i % 10 == 9 or i == len(rows) - 1:
                torch.save(D, part)
                print(f"  {tag} L{layer} delta {i + 1}/{len(rows)}  last nll {nll:.3f}  {time.time() - t0:.0f}s", flush=True)
        out = {"K": torch.stack(D["K"]), "R": torch.stack(D["R"]), "info": D["info"]}
        torch.save(out, path)
        os.remove(part)
        return out

    def apply(self, layer, D, lam):
        """Reset every candidate layer to its base weight, then add the least-squares edit at `layer`."""
        self.reset()
        if lam is None:
            return
        C = self.covariance(layer)
        K, R = D["K"], D["R"]  # n x dk, n x d
        A = lam * C + K.T @ K
        X = torch.linalg.solve(A, K.T)  # dk x n
        self.layers[layer].mlp.down_proj.weight.data += R.T @ X.T
        del A, X, C
        torch.cuda.empty_cache()

    def reset(self):
        for l in self.edit_layers:
            self.layers[l].mlp.down_proj.weight.data.copy_(self.base[l])

    # --- measurement ---------------------------------------------------------------------------
    @torch.no_grad()
    def teacher_forced(self, question, answer):
        """True if greedy decoding would emit the answer: every target token is the argmax."""
        ids, _ = self.encode(question)
        tgt = self.target_ids(answer)[:-1]
        logits = self._forward(ids + tgt)[len(ids) - 1:len(ids) - 1 + len(tgt)]
        return bool((logits.argmax(-1) == torch.tensor(tgt, device=self.dev)).all())

    @torch.no_grad()
    def first_token(self, question):
        ids, _ = self.encode(question)
        return int(self._forward(ids)[-1].argmax())

    @torch.no_grad()
    def generate(self, questions):
        self.tok.padding_side = "left"
        answers = []
        for i in range(0, len(questions), GEN_BATCH):
            texts = [self.text(q) for q in questions[i:i + GEN_BATCH]]
            enc = self.tok(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(self.dev)
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            answers += [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
        return answers

    def run_set(self, qfile, out):
        df = pd.read_csv(qfile)
        answers = self.generate(df["question"].tolist())
        rows = []
        for i, (r, a) in enumerate(zip(df.itertuples(), answers)):
            ok, note = grade(r.question, a, r.expected_answer)
            rows.append({"id": i + 1, "question": r.question, "answer": a, "correct": int(ok),
                         "expected_answer": r.expected_answer, "notes": note, "category": r.category})
        pd.DataFrame(rows).to_csv(out, index=False)
        acc = np.mean([x["correct"] for x in rows])
        print(f"  {out}: {acc:.1%}", flush=True)
        return out


def corrections_with_canonical(path):
    c = pd.read_csv(path)
    c["correct_answer"] = c["correct_answer"].astype(str)
    return c


def dev(args):
    """Tune layer and lam on the phase 3 split with cheap proxies (no generation):
    edit  = teacher-forced exact match on the 129 edited training questions
    para  = the same on paraphrases whose source question was edited
    drift = share of held-out / right-without-memory training questions whose first answer token changes
    null  = drift under a random weight change of relative size 1e-6 at the same layer (3 seeds). Any
            perturbation flips ~15% of first tokens in this 8-bit model (dev sweep 1, NOTES.md), so drift
            is judged against this floor, not against 0.
    """
    E = Editor()
    corr = corrections_with_canonical("runs/gemma3-4b-twotrack/corrections.csv")
    para = pd.read_csv("questions_trivia_paraphrase.csv")
    ans = dict(zip(corr["question"], corr["correct_answer"]))
    para = para[para["source_question"].isin(ans)]
    hold = pd.read_csv("questions_trivia_holdout.csv")["question"].tolist()
    train_right = pd.read_csv("runs/gemma3-4b-twotrack/train_off.csv").query("correct == 1")["question"].tolist()
    E.reset()
    base_first = {q: E.first_token(q) for q in hold + train_right}
    out = os.path.join("runs", "gemma3-4b-edit", "dev_sweep.csv")
    # Keep finished layers from an earlier (killed) sweep; rerun only the requested ones.
    kept = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    if not kept.empty:  # drop only what is rerun: the nulls and the requested lams of the requested layers
        lam = kept["lam"].astype(str)
        rerun = lam.str.startswith("null") | pd.to_numeric(lam, errors="coerce").isin(args.lams)
        kept = kept[~(kept["layer"].isin(args.layers) & rerun)]
    rows = []

    def drifts():
        return (np.mean([E.first_token(q) != base_first[q] for q in hold]),
                np.mean([E.first_token(q) != base_first[q] for q in train_right]))

    for layer in args.layers:
        W = E.base[layer]
        for seed in range(3):
            E.reset()
            g = torch.Generator(device=W.device).manual_seed(seed)
            N = torch.randn(W.shape, generator=g, device=W.device)
            E.layers[layer].mlp.down_proj.weight.data += 1e-6 * W.norm() / N.norm() * N
            del N
            dh, dt = drifts()
            rows.append({"layer": layer, "lam": f"null{seed}", "edit": np.nan, "para": np.nan, "drift_holdout": dh, "drift_train_right": dt})
            print(f"L{layer:2d} null seed {seed}: drift held-out {dh:.1%}  train-right {dt:.1%}", flush=True)
        E.reset()
        D = E.deltas(corr, layer, "dev")
        for lam in args.lams:
            E.apply(layer, D, lam)
            edit = np.mean([E.teacher_forced(q, a) for q, a in ans.items()])
            par = np.mean([E.teacher_forced(r.question, ans[r.source_question]) for r in para.itertuples()])
            dh, dt = drifts()
            rows.append({"layer": layer, "lam": lam, "edit": edit, "para": par, "drift_holdout": dh, "drift_train_right": dt})
            print(f"L{layer:2d} lam {lam:>8g}: edit {edit:.1%}  para {par:.1%}  drift held-out {dh:.1%}  train-right {dt:.1%}", flush=True)
            pd.concat([kept, pd.DataFrame(rows)]).to_csv(out, index=False)
    E.reset()


def test(args):
    """One shot on the phase 4 split: baseline (unedited) and edited generations for train, paraphrase
    and held-out, then fixed/broke via compare.py."""
    from compare import compare
    d = os.path.join("runs", args.name)
    os.makedirs(d, exist_ok=True)
    E = Editor()
    corr = corrections_with_canonical("runs/gemma3-4b-phase4/corrections.csv")
    sets = {"train": "questions_trivia_v2.csv", "paraphrase": "questions_trivia_v2_paraphrase.csv",
            "holdout": "questions_trivia_v2_holdout.csv"}
    if args.sets:  # phase 6 reuses the cached test deltas on other question sets
        sets = dict(x.split("=", 1) for x in args.sets)
    for split, qfile in sets.items():
        out = os.path.join(d, f"{split}_off.csv")
        if not os.path.exists(out):
            E.reset()
            E.run_set(qfile, out)
    # Placebo edit: the same questions and procedure, but each question is edited toward another
    # question's answer (a fixed derangement). Same keys, similar magnitudes, wrong facts — so its
    # effect on unedited questions is the collateral of "changing these weights this much".
    shuffled = corr.copy()
    perm = np.random.default_rng(0).permutation(len(corr))
    while (perm == np.arange(len(corr))).any():
        perm = np.random.default_rng(int(perm[0]) + 1).permutation(len(corr))
    shuffled["correct_answer"] = corr["correct_answer"].values[perm]
    # Saved so prediction 4 (does the placebo emit its shuffled answer?) can be graded afterwards.
    shuffled[["question", "correct_answer"]].rename(columns={"correct_answer": "placebo_answer"}).to_csv(
        os.path.join(d, "placebo_targets.csv"), index=False)
    for arm, c in (("edited", corr), ("placebo", shuffled)):
        D = E.deltas(c, args.layer, f"test_{arm}")
        E.apply(args.layer, D, args.lam)
        for split, qfile in sets.items():
            out = os.path.join(d, f"{split}_{arm}.csv")
            if not os.path.exists(out):
                E.run_set(qfile, out)
    E.reset()
    for arm in ("edited", "placebo"):
        for split in sets:
            compare(os.path.join(d, f"{split}_off.csv"), os.path.join(d, f"{split}_{arm}.csv"),
                    out=os.path.join(d, f"{split}_{arm}_pairs.csv"), label=f"{arm} / {split}")
            print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cov")
    pd_ = sub.add_parser("dev")
    pd_.add_argument("--layers", type=int, nargs="+", default=list(CANDIDATE_LAYERS))
    pd_.add_argument("--lams", type=float, nargs="+", default=[1, 10, 100, 1000, 10000, 100000])
    pt = sub.add_parser("test")
    pt.add_argument("--layer", type=int, required=True)
    pt.add_argument("--lam", type=float, required=True)
    pt.add_argument("--name", default="gemma3-4b-edit")
    pt.add_argument("--sets", nargs="+", default=None, metavar="SPLIT=CSV",
                    help="evaluate these question sets instead of the phase 4 train/paraphrase/holdout")
    a = p.parse_args()
    if a.cmd == "cov":
        E = Editor()
        for l in CANDIDATE_LAYERS:
            E.covariance(l)
    elif a.cmd == "dev":
        dev(a)
    else:
        test(a)
