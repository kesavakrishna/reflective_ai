"""Build a larger train/held-out question set from TriviaQA (rc.nocontext, validation split).

64 hand-written questions gave Gemma 3 4B one fix and one harm — far too few to train a probe on.
TriviaQA lists every accepted alias per answer, which keeps grading deterministic: expected_answer
holds the aliases joined with "|" and grader.py accepts any of them. See NOTES.md.

  python build_triviaqa.py <path/to/validation.parquet>
"""

import argparse

import pandas as pd

SEED = 0
MIN_ALIAS_LEN = 3  # 1–2 char aliases ("US", "4") match inside unrelated answers too easily


def build(parquet_path, n_train, n_holdout, train_out, holdout_out):
    df = pd.read_parquet(parquet_path, columns=["question", "answer"])
    # Canonical answer first (correction memos show it), then the remaining aliases sorted.
    df["aliases"] = df["answer"].map(
        lambda a: [str(a["value"]).strip()] + sorted({str(x).strip() for x in a["aliases"]} - {str(a["value"]).strip()})
    )
    df = df[df["aliases"].map(
        lambda al: bool(al) and all("|" not in x for x in al) and min(map(len, al)) >= MIN_ALIAS_LEN
    )]
    df = df[df["question"].str.len() <= 200].drop_duplicates("question")

    sample = df.sample(n=n_train + n_holdout, random_state=SEED)
    out = pd.DataFrame({
        "question": sample["question"].str.strip(),
        "expected_answer": sample["aliases"].map("|".join),
        "category": "trivia",
    })
    out.iloc[:n_train].to_csv(train_out, index=False)
    out.iloc[n_train:].to_csv(holdout_out, index=False)
    print(f"{len(df)} eligible -> {n_train} train ({train_out}), {n_holdout} held-out ({holdout_out})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="TriviaQA rc.nocontext validation parquet")
    p.add_argument("--n-train", type=int, default=250)
    p.add_argument("--n-holdout", type=int, default=250)
    p.add_argument("--train-out", default="questions_trivia.csv")
    p.add_argument("--holdout-out", default="questions_trivia_holdout.csv")
    args = p.parse_args()
    build(args.parquet, args.n_train, args.n_holdout, args.train_out, args.holdout_out)
