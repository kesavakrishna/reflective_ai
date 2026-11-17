"""Analyze activations from interp_capture.py. See NOTES.md.

1. Harm probe. Among questions the model gets RIGHT with memory off, predict which ones memory will
   break, from the memory-OFF residual stream (i.e. before any lesson is seen). Per-layer logistic
   regression, cross-validated AUC. It has to beat two cheap explanations that need no activations:
     - confidence: the model's own probability on its (correct) answer's first token
     - lesson similarity: cosine similarity between the question and the lessons it retrieved
   Picking the best of 34 layers inflates AUC, so a label-permutation null for the max is reported too.
2. Logit lens on broke / fixed cases: per layer, logit(OFF answer) − logit(ON answer) at the answer
   position, with and without the lessons — where in depth the lesson flips the answer.

  python interp_analyze.py runs/gemma3-4b-trivia
"""

import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_PERM = 20


def cv_auc(X, y, seeds=(0, 1, 2)):
    """Mean per-fold AUC over a few 5-fold splits. Per fold, not pooled: each fold's model scales its
    scores differently, so pooling out-of-fold scores understates AUC."""
    aucs = []
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000))
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs))


def lesson_similarity(meta, run_dir):
    model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    lessons = pd.read_csv(os.path.join(run_dir, "lessons.csv")).set_index("id")["text"]
    q = model.encode(meta["question"].tolist(), normalize_embeddings=True)
    sims = []
    for qi, ids in zip(q, meta["retrieved_lesson_ids"].astype(str)):
        texts = [lessons[int(x)] for x in ids.split(";") if x.strip().isdigit()]
        sims.append(float((model.encode(texts, normalize_embeddings=True) @ qi).max()) if texts else 0.0)
    return np.array(sims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    args = p.parse_args()

    z = np.load(os.path.join(args.run_dir, "activations.npz"))
    meta = pd.read_csv(os.path.join(args.run_dir, "activations_meta.csv"))
    acts, conf = z["acts_off"].astype(np.float32), z["conf_off"]
    n_layers = acts.shape[1]

    # ---- 1. harm probe ----
    right = (meta["correct_off"] == 1).to_numpy()
    y = (meta.loc[right, "outcome"] == "broke").to_numpy().astype(int)
    X_all = acts[right]
    print(f"Harm probe: {right.sum()} questions right with memory off, {y.sum()} broken by memory\n")

    sim = lesson_similarity(meta, args.run_dir)
    print(f"Baseline  low confidence      AUC {roc_auc_score(y, -conf[right]):.3f}")
    print(f"Baseline  lesson similarity   AUC {roc_auc_score(y, sim[right]):.3f}")
    cv_conf = cv_auc(np.c_[np.log(conf[right] + 1e-9), sim[right]], y)
    print(f"Baseline  conf + similarity   AUC {cv_conf:.3f}  (cross-validated LR)\n")

    layer_auc = np.array([cv_auc(X_all[:, l], y) for l in range(n_layers)])
    for l in range(n_layers):
        print(f"layer {l:2d}  AUC {layer_auc[l]:.3f}  {'#' * int(max(0, layer_auc[l] - 0.5) * 100)}")
    best = int(layer_auc.argmax())
    print(f"\nBest layer {best}: AUC {layer_auc[best]:.3f}")

    # Null for "max AUC over layers": shuffle labels, take the best layer each time (1 split, every 2nd layer for speed).
    rng = np.random.default_rng(SEED)
    null = [max(cv_auc(X_all[:, l], rng.permutation(y), seeds=(0,)) for l in range(0, n_layers, 2)) for _ in range(N_PERM)]
    print(f"Permutation null for best-layer AUC: mean {np.mean(null):.3f}, max {np.max(null):.3f} ({N_PERM} shuffles)")

    combo = cv_auc(np.c_[X_all[:, best], np.log(conf[right] + 1e-9), sim[right]], y)
    print(f"Probe (layer {best}) + conf + similarity: AUC {combo:.3f}  vs baselines alone {cv_conf:.3f}")

    # ---- 2. logit lens ----
    print("\nLogit lens: mean logit(OFF answer) - logit(ON answer), per layer. >0 = OFF answer winning.")
    for outcome in ("broke", "fixed"):
        m = (meta["outcome"] == outcome).to_numpy() & (np.abs(z["lens_off"]).sum(1) > 0)  # drop same-first-token pairs
        off, on = z["lens_off"][m].mean(0), z["lens_on"][m].mean(0)
        print(f"\n{outcome} (n={m.sum()})   layer: no lesson / with lesson / gap")
        for l in range(0, n_layers, 3):
            print(f"  {l:2d}: {off[l]:+7.2f}  {on[l]:+7.2f}  {off[l] - on[l]:+6.2f}")
        print(f"  {n_layers - 1:2d}: {off[-1]:+7.2f}  {on[-1]:+7.2f}  {off[-1] - on[-1]:+6.2f}")

    pd.DataFrame({"layer": range(n_layers), "probe_auc": layer_auc}).to_csv(
        os.path.join(args.run_dir, "probe_auc_by_layer.csv"), index=False)


if __name__ == "__main__":
    main()
