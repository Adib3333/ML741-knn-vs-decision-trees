"""Step 6: control parameter selection under macro F1.

The first search maximised balanced accuracy, which is the mean per-class recall
and therefore charges nothing for false positives. Macro F1 charges for precision
and recall together, so the two objectives disagree.

The disagreement is visible in the selected tree. class_weight 'balanced' pushed
Shellcode recall to 0.939 while its precision fell to 0.273. That trade is free
under balanced accuracy and close to a disaster under macro F1.

So the search runs again against macro F1, with the class weighting turned into a
dial rather than a fixed choice:

    w_c = ( N / (K * n_c) ) ** alpha

alpha = 0 is no weighting, alpha = 1 is what sklearn calls 'balanced'. Weights
come from the training fold inside fit, never from the held-out fold.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from a2_common import (RANDOM_SEED, OUT_DIR, ensure_dirs, load_data,
                       numeric_columns, build_knn_preprocessor,
                       build_ct_preprocessor, log)
import a2_step2_tune as T

LOGP = f"{OUT_DIR}/retune_f1_log.txt"


class PowerWeightedTree(BaseEstimator, ClassifierMixin):
    """Tree whose class weights are raised to a tunable power.

    alpha = 0 is no weighting, alpha = 1 is sklearn's 'balanced'. In between it
    trades recall on the rare classes against precision on them, which is the
    trade macro F1 actually scores.
    """

    def __init__(self, alpha: float = 1.0, criterion: str = "entropy",
                 max_depth=None, min_samples_leaf: int = 1,
                 ccp_alpha: float = 0.0, random_state: int = RANDOM_SEED):
        self.alpha = alpha
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        # inverse frequency, raised to alpha; computed on the training fold only
        w = (len(y) / (len(classes) * counts)) ** float(self.alpha)
        cw = {int(c): float(wi) for c, wi in zip(classes, w)}
        self.model_ = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=T.as_depth(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            ccp_alpha=float(self.ccp_alpha),
            class_weight=cw,
            random_state=self.random_state).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def by_f1(grid: pd.DataFrame) -> pd.DataFrame:
    return grid.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def main(which: str = "both") -> None:
    ensure_dirs()
    X, y = load_data()
    num = numeric_columns(X)
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=T.TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    X_tune = X_tune.reset_index(drop=True)
    y_tune = y_tune.reset_index(drop=True)
    log(f"macro F1 retune: tuning partition {X_tune.shape}", LOGP)

    best = {}

    # --- tree
    if which in ("both", "tree"):
        def ct_pipe(alpha, max_depth, min_samples_leaf, criterion, ccp_alpha):
            return Pipeline([
                ("prep", build_ct_preprocessor(num)),
                ("clf", PowerWeightedTree(alpha=alpha, criterion=criterion,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          ccp_alpha=ccp_alpha))])

        g1 = T.run_grid("tree stage A (macro F1)", ct_pipe,
                        dict(alpha=[0.0, 0.25, 0.5, 0.75, 1.0],
                             max_depth=[13, 20, 30, None],
                             min_samples_leaf=[1, 2, 5],
                             criterion=["entropy"],
                             ccp_alpha=[0.0]),
                        X_tune, y_tune, T.TUNE_FOLDS)
        g1 = by_f1(g1)
        g1.to_csv(f"{OUT_DIR}/retune_ct_f1_stageA.csv", index=False)
        b = g1.iloc[0]
        log(f"tree stage A winner: alpha={b.alpha} depth={b.max_depth} "
            f"leaf={b.min_samples_leaf} -> f1={b.f1_mean:.4f}", LOGP)

        # refine around the winner
        a0 = float(b.alpha)
        alphas = sorted({max(0.0, a0 - 0.125), a0, min(1.0, a0 + 0.125)})
        d0 = T.as_depth(b.max_depth)
        depths = [d0] if d0 is None else sorted({d0 - 3, d0, d0 + 5, None},
                                                key=lambda v: (v is None, v))
        g2 = T.run_grid("tree stage B (macro F1)", ct_pipe,
                        dict(alpha=alphas,
                             max_depth=depths,
                             min_samples_leaf=sorted({int(b.min_samples_leaf), 1, 2, 5}),
                             criterion=["entropy", "gini"],
                             ccp_alpha=[0.0, 1e-6, 1e-5]),
                        X_tune, y_tune, T.TUNE_FOLDS)
        g2 = by_f1(g2)
        g2.to_csv(f"{OUT_DIR}/retune_ct_f1_stageB.csv", index=False)
        b = g2.iloc[0]
        best["tree"] = dict(alpha=float(b.alpha), criterion=str(b.criterion),
                            max_depth=T.as_depth(b.max_depth),
                            min_samples_leaf=int(b.min_samples_leaf),
                            ccp_alpha=float(b.ccp_alpha),
                            f1_mean=float(b.f1_mean), f1_std=float(b.f1_std),
                            bal_mean=float(b.bal_mean))
        log(f"TREE BEST (macro F1): {best['tree']}", LOGP)

    # --- k-NN
    if which in ("both", "knn"):
        def knn_pipe(k, weights, metric):
            return Pipeline([("prep", build_knn_preprocessor(num)),
                             ("clf", KNeighborsClassifier(n_neighbors=k,
                                                          weights=weights,
                                                          metric=metric,
                                                          n_jobs=-1))])

        gk = T.run_grid("k-NN (macro F1)", knn_pipe,
                        dict(k=[1, 2, 3, 4, 5, 7, 9],
                             weights=["uniform", "distance"],
                             metric=["manhattan"]),
                        X_tune, y_tune, T.TUNE_FOLDS)
        gk = by_f1(gk)
        gk.to_csv(f"{OUT_DIR}/retune_knn_f1.csv", index=False)
        b = gk.iloc[0]
        best["knn"] = dict(k=int(b.k), weights=str(b.weights), metric=str(b.metric),
                           f1_mean=float(b.f1_mean), f1_std=float(b.f1_std),
                           bal_mean=float(b.bal_mean))
        log(f"K-NN BEST (macro F1): {best['knn']}", LOGP)

    path = f"{OUT_DIR}/retune_f1_best_{which}.json"
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    log(f"written {path}", LOGP)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
