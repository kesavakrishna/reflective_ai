"""Apply the phase 5 selection rule to runs/gemma3-4b-edit/dev_sweep.csv (RESULTS.md, "Phase 5 design"
and its deviations):

  floor(layer) = mean held-out drift of the 3 null perturbations at that layer
  qualifies    = edit >= 80% and drift_holdout <= floor + 5 points
  pick         = highest para among qualifying configs, ties to lower drift
  fallback     = if none qualifies: highest edit with drift_holdout <= floor + 5 points (rule failed)

  python select_edit.py          # prints the table and the pick; last line is "LAYER LAM"
"""

import sys

import pandas as pd

SWEEP = "runs/gemma3-4b-edit/dev_sweep.csv"
EDIT_MIN, DRIFT_MARGIN = 0.80, 0.05


def select(path=SWEEP):
    df = pd.read_csv(path)
    null = df[df["lam"].astype(str).str.startswith("null")]
    floor = null.groupby("layer")["drift_holdout"].mean()
    cfg = df[~df.index.isin(null.index)].copy()
    cfg["lam"] = cfg["lam"].astype(float)
    cfg["floor"] = cfg["layer"].map(floor)
    cfg["excess"] = cfg["drift_holdout"] - cfg["floor"]
    cfg["drift_ok"] = cfg["excess"] <= DRIFT_MARGIN + 1e-9
    cfg["qualifies"] = cfg["drift_ok"] & (cfg["edit"] >= EDIT_MIN)
    print(cfg[["layer", "lam", "edit", "para", "drift_holdout", "floor", "excess", "drift_ok", "qualifies"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    missing = sorted(set(df["layer"]) ^ {4, 8, 12, 16})
    if missing:
        print(f"WARNING: layers missing from the sweep: {missing}")
    q = cfg[cfg["qualifies"]]
    if len(q):
        pick = q.sort_values(["para", "drift_holdout"], ascending=[False, True]).iloc[0]
        print("\nrule: satisfied")
    else:
        ok = cfg[cfg["drift_ok"]]
        if ok.empty:
            sys.exit("no configuration meets the drift limit; nothing to select")
        pick = ok.sort_values(["edit", "drift_holdout"], ascending=[False, True]).iloc[0]
        print("\nrule: FAILED (no config has edit >= 80% within the drift limit) -> fallback")
    print(f"pick: layer {int(pick.layer)}, lam {pick.lam:g}  (edit {pick.edit:.1%}, para {pick.para:.1%}, "
          f"drift {pick.drift_holdout:.1%} vs floor {pick.floor:.1%})")
    print(f"{int(pick.layer)} {pick.lam:g}")
    return int(pick.layer), float(pick.lam)


if __name__ == "__main__":
    select()
