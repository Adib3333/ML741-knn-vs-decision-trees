"""Re-tune k-NN after the representation changed.

The normalisation study swapped min-max for the rank transform and dropped the
correlation filter, which changes the shape of the space. So k, the vote
weighting and the distance measure get searched again rather than assumed.
"""
from __future__ import annotations

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from a2_common import (RANDOM_SEED, OUT_DIR, ensure_dirs, load_data,
                       numeric_columns, build_knn_preprocessor, log)
import a2_step2_tune as T

LOGP = f"{OUT_DIR}/tune_log.txt"


def main() -> None:
    ensure_dirs()
    X, y = load_data()
    num = numeric_columns(X)
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=T.TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    X_tune, y_tune = X_tune.reset_index(drop=True), y_tune.reset_index(drop=True)
    log(f"retune: tuning partition {X_tune.shape}", LOGP)

    def knn_pipe(k, weights, metric):
        return Pipeline([("prep", build_knn_preprocessor(num)),
                         ("clf", KNeighborsClassifier(n_neighbors=k,
                                                      weights=weights,
                                                      metric=metric, n_jobs=-1))])

    grid = T.run_grid("k-NN retune", knn_pipe,
                      dict(k=[1, 3, 5, 7, 9, 11, 15, 21, 31, 51],
                           weights=["uniform", "distance"],
                           metric=["euclidean", "manhattan"]),
                      X_tune, y_tune, T.TUNE_FOLDS)
    grid.to_csv(f"{OUT_DIR}/tune_knn_final.csv", index=False)
    b = grid.iloc[0]
    knn_best = dict(k=int(b.k), weights=str(b.weights), metric=str(b.metric),
                    bal_mean=float(b.bal_mean), bal_std=float(b.bal_std))
    log(f"k-NN retuned best: {knn_best}", LOGP)

    with open(f"{OUT_DIR}/tune_best.json") as f:
        best = json.load(f)
    best["knn"] = knn_best
    with open(f"{OUT_DIR}/tune_best.json", "w") as f:
        json.dump(best, f, indent=2)

    T.figure_knn_k(grid)
    log("retune complete", LOGP)


if __name__ == "__main__":
    main()
