"""Resume tuning: tree stage 2, best-configuration record, figures.

The k-NN grids and the tree stage 1 grid finished in the previous run, so their
saved CSVs are reused instead of recomputed.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import (RANDOM_SEED, OUT_DIR, ensure_dirs, load_data,
                       numeric_columns, build_ct_preprocessor, log)
import tune as T

LOGP = f"{OUT_DIR}/tune_log.txt"


def main() -> None:
    ensure_dirs()
    X, y = load_data()
    num = numeric_columns(X)
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=T.TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    X_tune, y_tune = X_tune.reset_index(drop=True), y_tune.reset_index(drop=True)
    log(f"resume: tuning partition {X_tune.shape}", LOGP)

    knn_s1 = pd.read_csv(f"{OUT_DIR}/tune_knn_stage1.csv")
    knn_s2 = pd.read_csv(f"{OUT_DIR}/tune_knn_stage2.csv")
    ct_s1 = pd.read_csv(f"{OUT_DIR}/tune_ct_stage1.csv")

    b2 = knn_s2.iloc[0]
    knn_best = dict(k=int(b2.k), weights=str(b2.weights), metric=str(b2.metric),
                    bal_mean=float(b2.bal_mean), bal_std=float(b2.bal_std))
    log(f"k-NN final: {knn_best}", LOGP)

    def ct_pipe(criterion, max_depth, min_samples_leaf, class_weight, ccp_alpha):
        return Pipeline([
            ("prep", build_ct_preprocessor(num)),
            ("clf", DecisionTreeClassifier(
                criterion=str(criterion), max_depth=T.as_depth(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                class_weight=T.as_weight(class_weight),
                ccp_alpha=float(ccp_alpha), random_state=RANDOM_SEED)),
        ])

    c = ct_s1.iloc[0]
    depth0 = T.as_depth(c.max_depth)
    depths = [max(1, depth0 - 3), depth0, depth0 + 3, None] if depth0 else [None]
    ct_s2 = T.run_grid(
        "tree stage 2", ct_pipe,
        dict(criterion=[str(c.criterion)], max_depth=depths,
             min_samples_leaf=[int(c.min_samples_leaf)],
             class_weight=[T.as_weight(c.class_weight)],
             ccp_alpha=[0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]),
        X_tune, y_tune, T.TUNE_FOLDS)
    ct_s2.to_csv(f"{OUT_DIR}/tune_ct_stage2.csv", index=False)

    c2 = ct_s2.iloc[0]
    ct_best = dict(criterion=str(c2.criterion), max_depth=T.as_depth(c2.max_depth),
                   min_samples_leaf=int(c2.min_samples_leaf),
                   class_weight=T.as_weight(c2.class_weight),
                   ccp_alpha=float(c2.ccp_alpha),
                   bal_mean=float(c2.bal_mean), bal_std=float(c2.bal_std))
    log(f"tree final: {ct_best}", LOGP)

    with open(f"{OUT_DIR}/tune_best.json", "w") as f:
        json.dump(dict(knn=knn_best, tree=ct_best,
                       tune_fraction=T.TUNE_FRACTION, tune_folds=T.TUNE_FOLDS,
                       seed=RANDOM_SEED), f, indent=2)

    T.figure_knn_k(knn_s1)
    T.figure_ct_depth(ct_s1)
    log("resume complete", LOGP)


if __name__ == "__main__":
    main()
