"""Phase 9: multi-layer MEMIT, with and without reworded keys. See RESULTS.md ("Phase 9 design").

Phase 5's single-layer edit fixed the exact questions but transferred to 13% of paraphrases and 1 / 43
two-hop questions. Two changes from MEMIT (Meng et al. 2023) that could make an edit less of a lookup:

  1. Spread over layers. For each fact, one change delta_l to the down_proj output at the question's
     last token is optimized for every layer in LAYERS at once, with the weight decay summed over the
     layers, so the change is shared out instead of piled on one layer. The layers are then solved in
     order: before each one the model (earlier layers already edited) is re-run, and that layer's
     down_proj is solved so every prompt's output reaches its target m_l + delta_l.
     (MEMIT proper optimizes one residual-stream target and splits it across layers. In Gemma 3 every
     MLP output passes through a post-feedforward RMSNorm, so an MLP can't deliver residual steps of
     the size needed. The smoke test found residual targets of 4e3-2e4 against a norm-inverse error
     of ~0.9. Hence targets in down_proj-output space; see NOTES.md.)
  2. Reworded keys ("aug" arm). Each fact also gets N_AUG automatic rewordings of its question (from
     build_rewordings.py, not the test paraphrases). delta is optimized jointly over all of a fact's
     prompts, and every prompt adds its own key row with its own target, so the edit has to fire on
     more than one wording.

The per-layer solve is the phase 5 one: dW = R K^T (lam C + K K^T)^-1, with C the cov_mix covariance.

  python memit.py dev    # tune lam per arm on the phase 3 split (proxies, like weight_edit.py dev)
  python memit.py test --lam-memit X --lam-aug Y   # one shot on the phase 4 split + two-hop sets
"""

import argparse
import math
import os
import time

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from weight_edit import CACHE, Editor, corrections_with_canonical  # noqa: E402

LAYERS = (8, 10, 12, 14, 16)
TOP = LAYERS[-1]
# per-layer deltas on down_proj outputs: phase 5's settings (clamp 4 ||m||, weight decay 0.5, summed over layers)
STEPS, LR, CLAMP, WD = 40, 0.5, 4.0, 0.5
N_AUG = 2
OUT = os.path.join("runs", "gemma3-4b-memit")
ARMS = ("memit", "memit_aug")


class Memit(Editor):
    def __init__(self):
        super().__init__(LAYERS)
        for l in LAYERS:  # every key covariance from the unedited model, before any edit is applied
            self.reset()
            self.covariance(l)
        torch.cuda.empty_cache()

    # --- forward with per-layer deltas on down_proj outputs ---------------------------------
    def _run(self, ids, pos, deltas=None, capture=(), stop_after=None):
        """Forward one prompt. deltas: {layer: vector} added to that layer's down_proj output at `pos`.
        capture: layers whose down_proj (input, output) at `pos` to return. Returns (logits, {l: (k, m)})."""
        store, handles = {}, []

        class Stop(Exception):
            pass

        for l in set(capture) | set(deltas or {}):
            def hook(mod, inp, out, l=l):
                if l in capture:
                    store[l] = (inp[0][0, pos].detach().float(), out[0, pos].detach().float())
                if deltas and l in deltas:
                    mask = torch.zeros(out.shape[1], 1, device=out.device, dtype=torch.float32)
                    mask[pos] = 1.0
                    return (out.float() + mask * deltas[l]).to(out.dtype)
            handles.append(self.layers[l].mlp.down_proj.register_forward_hook(hook))
        if stop_after is not None:
            def stop(mod, inp, out):
                raise Stop
            handles.append(self.layers[stop_after].register_forward_hook(stop))
        try:
            x = torch.tensor([ids], device=self.dev)
            logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
        except Stop:
            logits = None
        finally:
            for h in handles:
                h.remove()
        return logits, store

    # --- per-fact targets: one delta per layer, optimized jointly -----------------------------
    def target(self, questions, answer):
        """Jointly optimize a change delta_l to the down_proj output of every layer in LAYERS (shared by
        all of a fact's prompts) so the model emits `answer`. Weight decay is summed over layers, so the
        change spreads across them. Returns, per prompt, the target outputs v_l = m_l + delta_l."""
        tgt = self.target_ids(answer)
        enc = [self.encode(q) for q in questions]
        with torch.no_grad():
            m0 = [self._run(ids, pos, capture=LAYERS, stop_after=TOP)[1] for ids, pos in enc]
        mn = {l: torch.stack([m[l][1] for m in m0]).norm(dim=1).mean() for l in LAYERS}
        d = m0[0][LAYERS[0]][1].numel()
        u = {l: torch.zeros(d, device=self.dev, requires_grad=True) for l in LAYERS}
        opt = torch.optim.Adam(list(u.values()), lr=LR)
        labels = torch.tensor(tgt, device=self.dev)
        for step in range(STEPS):
            deltas = {l: u[l] * mn[l] / math.sqrt(d) for l in LAYERS}
            nlls, pmin = [], 1.0
            for ids, pos in enc:
                logits = self._run(ids + tgt, pos, deltas=deltas)[0][len(ids) - 1:len(ids) - 1 + len(tgt)].float()
                lp = torch.log_softmax(logits, -1).gather(1, labels[:, None])
                nlls.append(-lp.mean())
                pmin = min(pmin, float(lp.exp().min()))
            nll = torch.stack(nlls).mean()
            if pmin > 0.9:
                break
            loss = nll + WD * sum((deltas[l].norm() / mn[l]) ** 2 for l in LAYERS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for l in LAYERS:
                    dl = u[l] * mn[l] / math.sqrt(d)
                    if dl.norm() > CLAMP * mn[l]:
                        u[l].mul_(CLAMP * mn[l] / dl.norm())
        deltas = {l: (u[l] * mn[l] / math.sqrt(d)).detach() for l in LAYERS}
        # Target = each layer's output in the joint forward (which already carries the upstream deltas'
        # effect) plus its own delta, not the unedited output plus delta.
        with torch.no_grad():
            mj = [self._run(ids, pos, deltas=deltas, capture=LAYERS, stop_after=TOP)[1] for ids, pos in enc]
        v = [{l: m[l][1] + deltas[l] for l in LAYERS} for m in mj]
        rel = {l: float(deltas[l].norm() / mn[l]) for l in LAYERS}
        return v, float(nll.detach()), step, rel

    def targets(self, corrections, rewordings, tag):
        """Residual targets for every fact (cached; checkpointed every 10 facts)."""
        path = os.path.join(CACHE, f"memit_{tag}.pt")
        part = path + ".partial"
        if os.path.exists(path):
            return torch.load(path, map_location=self.dev)
        T = torch.load(part, map_location=self.dev) if os.path.exists(part) else []
        t0 = time.time()
        rows = list(corrections.itertuples())
        for i in range(len(T), len(rows)):
            r = rows[i]
            qs = [r.question] + (rewordings.get(r.question, [])[:N_AUG] if rewordings is not None else [])
            v, nll, steps, rel = self.target(qs, r.correct_answer)
            T.append({"questions": qs, "v": v, "nll": nll, "steps": steps, "rel": rel})
            if i % 10 == 9 or i == len(rows) - 1:
                torch.save(T, part)
                print(f"  {tag} target {i + 1}/{len(rows)}  last nll {nll:.3f}  {time.time() - t0:.0f}s", flush=True)
        torch.save(T, path)
        os.remove(part)
        return T


    # --- solving the layers in order --------------------------------------------------------
    def apply(self, T, lam, verbose=False):
        """Reset, then edit LAYERS in order. Before each layer the model (earlier layers already edited)
        is re-run, and that layer's down_proj is solved so every prompt's output moves to its target v_l."""
        self.reset()
        if lam is None:
            return
        prompts = [(self.encode(q), v) for t in T for q, v in zip(t["questions"], t["v"])]
        for l in LAYERS:
            K, R = [], []
            with torch.no_grad():
                for (ids, pos), v in prompts:
                    k, m = self._run(ids, pos, capture=(l,), stop_after=l)[1][l]
                    K.append(k)
                    R.append(v[l] - m)
            K, R = torch.stack(K), torch.stack(R)
            C = self.covariance(l)
            A = lam * C + K.T @ K
            X = torch.linalg.solve(A, K.T)
            self.layers[l].mlp.down_proj.weight.data += R.T @ X.T
            if verbose:
                got = (K @ (R.T @ X.T).T - R).norm(dim=1) / R.norm(dim=1).clamp_min(1e-8)
                print(f"    L{l}: {len(K)} keys, |R| median {R.norm(dim=1).median():.2f}, "
                      f"solve residual median {got.median():.2f}", flush=True)
            del A, X, C
            torch.cuda.empty_cache()

    def null(self, seed):
        """A random relative-1e-6 change to every edited layer (the drift floor)."""
        self.reset()
        for l in LAYERS:
            W = self.base[l]
            g = torch.Generator(device=W.device).manual_seed(seed * 100 + l)
            N = torch.randn(W.shape, generator=g, device=W.device)
            self.layers[l].mlp.down_proj.weight.data += 1e-6 * W.norm() / N.norm() * N
            del N


def rewordings_map(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run build_rewordings.py first")
    r = pd.read_csv(path)
    return r.groupby("source_question")["rewording"].apply(list).to_dict()


def dev(args):
    """Phase 3 split, proxies as in weight_edit.py dev: edit / para / drift (against a null floor)."""
    E = Memit()
    corr = corrections_with_canonical("runs/gemma3-4b-twotrack/corrections.csv")
    rew = rewordings_map("rewordings_trivia.csv")
    para = pd.read_csv("questions_trivia_paraphrase.csv")
    ans = dict(zip(corr["question"], corr["correct_answer"]))
    para = para[para["source_question"].isin(ans)]
    hold = pd.read_csv("questions_trivia_holdout.csv")["question"].tolist()
    train_right = pd.read_csv("runs/gemma3-4b-twotrack/train_off.csv").query("correct == 1")["question"].tolist()
    E.reset()
    base = {q: E.first_token(q) for q in hold + train_right}

    def drifts():
        return (np.mean([E.first_token(q) != base[q] for q in hold]),
                np.mean([E.first_token(q) != base[q] for q in train_right]))

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "dev_sweep.csv")
    rows = pd.read_csv(out, keep_default_na=False, na_values=[""]).to_dict("records") if os.path.exists(out) else []
    if not any(str(r["lam"]).startswith("null") for r in rows):
        for seed in range(3):
            E.null(seed)
            dh, dt = drifts()
            rows.append({"arm": "null", "lam": f"null{seed}", "edit": np.nan, "para": np.nan,
                         "drift_holdout": dh, "drift_train_right": dt})
            print(f"null seed {seed}: drift held-out {dh:.1%}  train-right {dt:.1%}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    done = {(r["arm"], str(float(r["lam"]))) for r in rows if not str(r["lam"]).startswith("null")}
    for arm in ARMS:
        T = E.targets(corr, rew if arm == "memit_aug" else None, f"dev_{arm}")
        for lam in args.lams:
            if (arm, str(float(lam))) in done:
                continue
            E.apply(T, lam, verbose=True)
            edit = np.mean([E.teacher_forced(q, a) for q, a in ans.items()])
            par = np.mean([E.teacher_forced(r.question, ans[r.source_question]) for r in para.itertuples()])
            dh, dt = drifts()
            rows.append({"arm": arm, "lam": lam, "edit": edit, "para": par, "drift_holdout": dh, "drift_train_right": dt})
            print(f"{arm} lam {lam:>8g}: edit {edit:.1%}  para {par:.1%}  drift held-out {dh:.1%}  "
                  f"train-right {dt:.1%}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    E.reset()


def select(path=os.path.join(OUT, "dev_sweep.csv"), edit_min=0.80, margin=0.05):
    """The phase 5 rule, per arm: highest para with edit >= 80% and drift <= floor + 5 points (ties to
    lower drift); else the highest edit within the drift limit (rule failed)."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep the arm "null" as text
    floor = df[df["arm"] == "null"]["drift_holdout"].mean()
    picks = {}
    for arm in ARMS:
        c = df[df["arm"] == arm].copy()
        c["lam"] = c["lam"].astype(float)
        c["ok"] = c["drift_holdout"] <= floor + margin + 1e-9
        q = c[c["ok"] & (c["edit"] >= edit_min)]
        if len(q):
            p, rule = q.sort_values(["para", "drift_holdout"], ascending=[False, True]).iloc[0], "satisfied"
        elif c["ok"].any():
            p, rule = c[c["ok"]].sort_values(["edit", "drift_holdout"], ascending=[False, True]).iloc[0], "FAILED -> fallback"
        else:
            p, rule = None, "nothing within the drift limit"
        print(c.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"{arm}: floor {floor:.1%}, rule {rule}" + (f", pick lam {p.lam:g} (edit {p.edit:.1%}, para "
              f"{p.para:.1%}, drift {p.drift_holdout:.1%})" if p is not None else "") + "\n")
        picks[arm] = None if p is None else float(p.lam)
    return picks


def test(args):
    from compare import compare
    E = Memit()
    corr = corrections_with_canonical("runs/gemma3-4b-phase4/corrections.csv")
    rew = rewordings_map("rewordings_trivia_v2.csv")
    sets = {"train": "questions_trivia_v2.csv", "paraphrase": "questions_trivia_v2_paraphrase.csv",
            "holdout": "questions_trivia_v2_holdout.csv", "twohop": "questions_twohop.csv",
            "direct": "questions_twohop_direct.csv"}
    os.makedirs(OUT, exist_ok=True)
    for split, qfile in sets.items():
        out = os.path.join(OUT, f"{split}_off.csv")
        if not os.path.exists(out):
            E.reset()
            E.run_set(qfile, out)
    lams = {"memit": args.lam_memit, "memit_aug": args.lam_aug}
    for arm in ARMS:
        if lams[arm] is None:
            print(f"{arm}: no lam selected, skipped", flush=True)
            continue
        T = E.targets(corr, rew if arm == "memit_aug" else None, f"test_{arm}")
        applied = False
        for split, qfile in sets.items():
            out = os.path.join(OUT, f"{split}_{arm}.csv")
            if not os.path.exists(out):
                if not applied:
                    E.apply(T, lams[arm], verbose=True)
                    applied = True
                E.run_set(qfile, out)
    E.reset()
    for arm in ARMS:
        for split in sets:
            on = os.path.join(OUT, f"{split}_{arm}.csv")
            if os.path.exists(on):
                compare(os.path.join(OUT, f"{split}_off.csv"), on, out=os.path.join(OUT, f"{split}_{arm}_pairs.csv"),
                        label=f"{arm} / {split}")
                print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pd_ = sub.add_parser("dev")
    pd_.add_argument("--lams", type=float, nargs="+", default=[100, 1000, 10000, 100000])
    sub.add_parser("select")
    pt = sub.add_parser("test")
    pt.add_argument("--lam-memit", type=float, default=None)
    pt.add_argument("--lam-aug", type=float, default=None)
    a = p.parse_args()
    {"dev": dev, "select": lambda a: select(), "test": test}[a.cmd](a)
