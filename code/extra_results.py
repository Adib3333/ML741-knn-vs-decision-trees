"""The three extra result files the expanded report needs.

  rev_trainsize.csv  the four configurations scored on the tuning partition and
                     on the evaluation partition, which differ by 3.4x in
                     training records per fold
  rev_speccorr.csv   the three pairs the brief calls perfectly correlated, with
                     the correlation actually measured
  rev_perfold.csv    per-fold macro F1 and balanced accuracy
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

from common import OUT_DIR

RAW = os.environ.get("A2_DATA", "networkTraffic.csv")
ORDER = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]


def train_size_comparison() -> None:
    """Tuning-partition scores against evaluation-partition scores.

    The two differ by 3.4x in training records per fold, so comparing across
    them shows whether the ranking depends on how much training data there is.
    The tuning scores are recomputed here rather than read out of a log, so the
    figure is reproducible.
    """
    import json
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from common import load_data, numeric_columns
    from evaluate import TUNE_FRACTION
    from multipliers_tree import macro_f1_from_counts
    from final_eval import knn_probs, tree_probs

    tb = json.load(open(f"{OUT_DIR}/retune_f1_best_tree.json"))["tree"]
    arr = np.load(f"{OUT_DIR}/allconfig_multipliers.npy")
    mult = {n: arr[i] for i, n in enumerate(ORDER)}

    X, y = load_data()
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=TUNE_FRACTION, stratify=y, random_state=42)
    X_tune = X_tune.reset_index(drop=True)
    y_tune = y_tune.reset_index(drop=True)
    num = numeric_columns(X_tune)

    makers = {
        "k-NN":         lambda a, b, c: knn_probs(a, b, c, num, smote=False),
        "k-NN + SMOTE": lambda a, b, c: knn_probs(a, b, c, num, smote=True),
        "tree":         lambda a, b, c: tree_probs(a, b, c, num, tb, smote=False),
        "tree + SMOTE": lambda a, b, c: tree_probs(a, b, c, num, tb, smote=True),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_tune, y_tune))

    tuned = {}
    for name, mk in makers.items():
        sc = []
        for tr, te in splits:
            P = mk(X_tune.iloc[tr], y_tune.iloc[tr], X_tune.iloc[te])
            sc.append(macro_f1_from_counts(y_tune.iloc[te].to_numpy(),
                                           np.argmax(P * mult[name], axis=1)))
        tuned[name] = float(np.mean(sc))
        print(f"  tuning partition {name:14s} macro F1 = {tuned[name]:.4f}", flush=True)

    folds = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")
    ev = folds.groupby("configuration")["f1_macro"].mean()

    sizes = pd.read_csv(f"{OUT_DIR}/split_sizes.csv").set_index("partition")
    tr_tune = int(sizes.loc["tuning", "instances"]) * 4 / 5
    tr_eval = int(sizes.loc["evaluation", "instances"]) * 9 / 10

    df = pd.DataFrame([dict(configuration=c, tuning_f1=tuned[c],
                            evaluation_f1=float(ev[c]),
                            gain=float(ev[c]) - tuned[c]) for c in ORDER])
    df["tuning_rank"] = df.tuning_f1.rank(ascending=False).astype(int)
    df["evaluation_rank"] = df.evaluation_f1.rank(ascending=False).astype(int)
    df.to_csv(f"{OUT_DIR}/rev_trainsize.csv", index=False)

    pd.DataFrame([dict(train_records_tuning=int(tr_tune),
                       train_records_evaluation=int(tr_eval),
                       ratio=tr_eval / tr_tune,
                       ranking_identical=bool((df.tuning_rank == df.evaluation_rank).all()),
                       mean_gain=float(df.gain.mean()))]).to_csv(
        f"{OUT_DIR}/rev_trainsize_summary.csv", index=False)
    print("rev_trainsize.csv written, ranking identical:",
          bool((df.tuning_rank == df.evaluation_rank).all()))


def spec_correlations() -> None:
    """The pairs the specification calls perfectly correlated, measured."""
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, low_memory=False)
    num = [c for c in raw.columns
           if c not in ("proto", "state", "service", "attack_cat")]
    d = raw[num].apply(pd.to_numeric)

    pairs = [("sbytes", "sloss"), ("dbytes", "dloss"),
             ("is_ftp_login", "ct_ftp_cmd")]
    rows = [dict(feature_a=a, feature_b=b,
                 correlation=float(d[a].corr(d[b])),
                 exactly_one=bool(d[a].corr(d[b]) == 1.0))
            for a, b in pairs]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/rev_speccorr.csv", index=False)

    C = d.drop(columns=["id"]).corr().abs()
    iu = np.triu(np.ones(C.shape, dtype=bool), k=1)
    v = C.to_numpy()[iu]
    pd.DataFrame([dict(pairs_exactly_one=int((v == 1.0).sum()),
                       pairs_above_099=int((v >= 0.99).sum()),
                       pairs_above_095=int((v >= 0.95).sum()),
                       max_correlation=float(v.max()))]).to_csv(
        f"{OUT_DIR}/rev_speccorr_summary.csv", index=False)
    print("rev_speccorr.csv written, exact unit correlations:",
          int((v == 1.0).sum()))


def per_fold() -> None:
    folds = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")
    wide = folds.pivot(index="fold", columns="configuration",
                       values=["f1_macro", "balanced_accuracy"])
    wide.to_csv(f"{OUT_DIR}/rev_perfold.csv")
    print(f"rev_perfold.csv written, {len(wide)} folds")


if __name__ == "__main__":
    train_size_comparison()
    spec_correlations()
    per_fold()
