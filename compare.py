"""Paired memory-off vs memory-on comparison: what memory FIXED and what it BROKE.

Accuracy alone hides harm — Llama's held-out went 24 → 23 and that single number didn't say a
correct answer had flipped to wrong. Here every question is classified by its off → on outcome:
  fixed        wrong → right
  broke        right → wrong   (the harm case)
  stayed_right right → right
  stayed_wrong wrong → wrong
The per-question pairs CSV is the input for the interpretability step. See NOTES.md.
"""

import argparse

import pandas as pd

OUTCOMES = {(0, 1): "fixed", (1, 0): "broke", (1, 1): "stayed_right", (0, 0): "stayed_wrong"}


def compare(off_path, on_path, out=None, label=""):
    off = pd.read_csv(off_path)
    on = pd.read_csv(on_path)
    # Retrieval ids describe the memory-on run; take them from there even if the off run lacks
    # the column (phase 2 off runs predate retrieved_correction_ids, so the merge wouldn't suffix it).
    off = off.drop(columns=[c for c in off.columns if c.startswith("retrieved_")])
    on = on.rename(columns={c: f"{c}_on" for c in on.columns if c.startswith("retrieved_")})
    pairs = off.merge(on, on="question", suffixes=("_off", "_on"))
    if len(pairs) != len(off) or len(pairs) != len(on):
        raise ValueError(f"runs don't pair up: off={len(off)} on={len(on)} matched={len(pairs)}")

    pairs["outcome"] = [
        OUTCOMES[(int(a), int(b))] for a, b in zip(pairs["correct_off"], pairs["correct_on"])
    ]
    counts = pairs["outcome"].value_counts().reindex(list(OUTCOMES.values()), fill_value=0)
    n_right_off = int(pairs["correct_off"].sum())
    n_wrong_off = len(pairs) - n_right_off

    print(f"=== {label or on_path} ===")
    print(f"Accuracy  off {pairs['correct_off'].mean():.1%}  ->  on {pairs['correct_on'].mean():.1%}")
    print(f"Fixed  {counts['fixed']:3d}   (of {n_wrong_off} wrong with memory off)")
    print(f"Broke  {counts['broke']:3d}   (of {n_right_off} right with memory off)")
    print(f"Net    {counts['fixed'] - counts['broke']:+3d}")
    if n_right_off:
        print(f"Harm rate: {counts['broke'] / n_right_off:.1%} of correct answers broken by memory")

    for outcome in ("broke", "fixed"):
        rows = pairs[pairs["outcome"] == outcome]
        if rows.empty:
            continue
        print(f"\n{outcome.upper()}:")
        for _, r in rows.iterrows():
            print(f"  {r['question'][:70]}")
            print(f"    off: {str(r['answer_off'])[:50]!r}  on: {str(r['answer_on'])[:50]!r}"
                  f"  expected: {str(r['expected_answer_on']).split('|')[0]}"
                  + (f"  lessons: {r['retrieved_lesson_ids_on']}" if "retrieved_lesson_ids_on" in r else "")
                  + (f"  corrections: {r['retrieved_correction_ids_on']}" if "retrieved_correction_ids_on" in r else ""))

    if out:
        cols = ["question", "category_on", "expected_answer_on", "answer_off", "answer_on",
                "correct_off", "correct_on", "outcome", "retrieved_lesson_ids_on", "retrieved_correction_ids_on"]
        cols = [c for c in cols if c in pairs.columns]
        renamed = pairs[cols].rename(columns={
            "category_on": "category", "expected_answer_on": "expected_answer",
            "retrieved_lesson_ids_on": "retrieved_lesson_ids",
            "retrieved_correction_ids_on": "retrieved_correction_ids",
        })
        renamed.to_csv(out, index=False)
        print(f"\nSaved pairs to {out}")
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("off", help="memory-off run CSV (from evaluate.py --out)")
    p.add_argument("on", help="memory-on run CSV")
    p.add_argument("--out", default=None, help="save per-question pairs to this CSV")
    args = p.parse_args()
    compare(args.off, args.on, out=args.out)
