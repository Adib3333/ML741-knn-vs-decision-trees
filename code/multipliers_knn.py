"""The same decision multipliers for k-NN.

k-NN also emits a class distribution, namely the weighted share of the k
neighbours belonging to each class,

    P(c | x) = sum of w_i over neighbours of class c  /  sum of all w_i

with w_i = 1 for uniform voting and 1/d_i for distance weighting. Predict
argmax_c m_c * P(c | x), multipliers fitted by coordinate ascent on macro F1.

One thing that does not come up with the tree. At k = 5 the vote share takes only
a handful of values, so the distribution is coarse and there is little for a
multiplier to flip. A larger neighbourhood gives a finer distribution and more to
work with, even where the plain arg max is no better. The best k under this rule
is therefore not necessarily the best k under arg max, so it is searched again.

To keep that cheap, the neighbour query runs once per fold at the largest k.
sklearn returns neighbours sorted by distance, so any smaller k is a truncation
of the same result and three values of k cost one search.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from common import (RANDOM_SEED, OUT_DIR, CLASS_NAMES, CLASS_ORDER,
                       ensure_dirs, load_data, numeric_columns,
                       build_knn_preprocessor, log)
from evaluate import TUNE_FRACTION
from multipliers_tree import macro_f1_from_counts, fit_multipliers

LOGP = f"{OUT_DIR}/knn_thresholds_log.txt"
K_LIST = [5, 15, 31]
K_MAX = max(K_LIST)
METRIC = "manhattan"
INNER_FOLDS = 5
EVAL_FOLDS = 10


def fold_probabilities(Xtr, ytr, Xte, num, k_list=K_LIST, k_max=K_MAX):
    """Return {k: probability matrix} from one neighbour query.

    The preprocessor fits on the training fold only, same as in the pipeline
    used everywhere else.
    """
    prep = build_knn_preprocessor(num)
    Ztr = prep.fit_transform(Xtr, ytr)
    Zte = prep.transform(Xte)

    nn = NearestNeighbors(n_neighbors=k_max, metric=METRIC, n_jobs=-1).fit(Ztr)
    dist, idx = nn.kneighbors(Zte)          # both sorted by increasing distance
    labels = np.asarray(ytr)[idx]           # (n_test, k_max)

    out = {}
    for k in k_list:
        d = dist[:, :k]
        lab = labels[:, :k]
        # distance weighting, with exact matches taking the vote outright
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        exact = ~np.isfinite(w)
        rows_with_exact = exact.any(axis=1)
        w[exact] = 1.0
        # where an exact match exists, only the exact matches vote
        w[rows_with_exact] *= exact[rows_with_exact]
        P = np.zeros((len(lab), 10))
        for c in range(10):
            P[:, c] = (w * (lab == c)).sum(axis=1)
        P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        out[k] = P
    return out


def main():
    ensure_dirs()
    open(LOGP, "w").close()

    X, y = load_data()
    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X, y, train_size=TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    for d in (X_tune, X_eval):
        d.reset_index(drop=True, inplace=True)
    y_tune = y_tune.reset_index(drop=True)
    y_eval = y_eval.reset_index(drop=True)
    num = numeric_columns(X_tune)

    # --- out of fold probabilities on tuning
    log("tuning partition: one neighbour query per fold, k up to "
        f"{K_MAX}", LOGP)
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)
    probs = {k: [] for k in K_LIST}
    ys = []
    for f, (tr, te) in enumerate(skf.split(X_tune, y_tune), 1):
        got = fold_probabilities(X_tune.iloc[tr], y_tune.iloc[tr],
                                 X_tune.iloc[te], num)
        for k in K_LIST:
            probs[k].append(got[k])
        ys.append(y_tune.iloc[te].to_numpy())
        log(f"  fold {f} done", LOGP)

    # --- fit multipliers for each k
    best_k, best_w, best_score = None, None, -1.0
    for k in K_LIST:
        plain = float(np.mean([macro_f1_from_counts(yv, np.argmax(P, axis=1))
                               for P, yv in zip(probs[k], ys)]))
        log(f"\nk={k}: plain arg max macro F1 = {plain:.4f}", LOGP)
        w, tuned = fit_multipliers(probs[k], ys)
        log(f"k={k}: tuned macro F1 = {tuned:.4f}  (gain {tuned-plain:+.4f})", LOGP)
        if tuned > best_score:
            best_k, best_w, best_score = k, w, tuned
    log(f"\nselected k={best_k}, tuning macro F1 = {best_score:.4f}", LOGP)
    log("multipliers: " + ", ".join(f"{CLASS_NAMES[c]}={best_w[c]:.3f}"
                                    for c in CLASS_ORDER), LOGP)

    # --- apply unchanged to the evaluation partition
    log("\nevaluation partition, multipliers held fixed", LOGP)
    skf10 = StratifiedKFold(n_splits=EVAL_FOLDS, shuffle=True,
                            random_state=RANDOM_SEED)
    rows = []
    for f, (tr, te) in enumerate(skf10.split(X_eval, y_eval), 1):
        got = fold_probabilities(X_eval.iloc[tr], y_eval.iloc[tr],
                                 X_eval.iloc[te], num, k_list=[5, best_k],
                                 k_max=max(5, best_k))
        yv = y_eval.iloc[te].to_numpy()
        rows.append(dict(
            fold=f,
            baseline=macro_f1_from_counts(yv, np.argmax(got[5], axis=1)),
            tuned=macro_f1_from_counts(yv, np.argmax(got[best_k] * best_w, axis=1))))
        log(f"  fold {f:2d}: baseline(k=5)={rows[-1]['baseline']:.4f} "
            f"tuned(k={best_k})={rows[-1]['tuned']:.4f}", LOGP)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/knn_threshold_folds.csv", index=False)
    np.save(f"{OUT_DIR}/knn_threshold_multipliers.npy", best_w)
    json.dump({"k": int(best_k), "metric": METRIC,
               "multipliers": [float(v) for v in best_w]},
              open(f"{OUT_DIR}/knn_threshold_best.json", "w"), indent=2)

    from scipy import stats
    log("\nEVALUATION PARTITION", LOGP)
    log(f"  baseline k-NN (k=5)   : {df.baseline.mean():.4f} "
        f"+- {df.baseline.std(ddof=1):.4f}", LOGP)
    log(f"  tuned k-NN (k={best_k})    : {df.tuned.mean():.4f} "
        f"+- {df.tuned.std(ddof=1):.4f}", LOGP)
    log(f"  gain                  : {df.tuned.mean()-df.baseline.mean():+.4f}", LOGP)
    log(f"  Wilcoxon p            : "
        f"{stats.wilcoxon(df.baseline, df.tuned).pvalue:.5f}", LOGP)


if __name__ == "__main__":
    main()
