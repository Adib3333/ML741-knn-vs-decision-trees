"""Cost sensitive decision multipliers for the tree.

The per-class breakdown says everything left is a precision loss on the rare
classes. The tree finds 77.7 per cent of Worms, but only 29.1 per cent of what it
calls Worms is Worms, so it predicts Worms about 347 times where 130 exist. alpha
is one knob for all ten classes, and one knob cannot fix ten different
precision-recall balances.

So the knob gets replaced by a decision rule. The tree already emits a class
distribution at each leaf. Instead of taking the arg max of it, predict

    argmax_c  w_c * P(c | leaf)

with the ten multipliers fitted by coordinate ascent on macro F1. All ones
reproduces the ordinary arg max, so the search cannot come back worse.

Multipliers are fitted on the tuning partition using its own internal
cross-validation, then applied unchanged to the evaluation partition.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from common import (RANDOM_SEED, OUT_DIR, CLASS_NAMES, CLASS_ORDER,
                       ensure_dirs, load_data, numeric_columns,
                       build_ct_preprocessor, log)
from evaluate import TUNE_FRACTION
from retune_f1 import PowerWeightedTree

LOGP = f"{OUT_DIR}/thresholds_log.txt"
INNER_FOLDS = 5


def macro_f1_from_counts(true, pred, K=10):
    M = np.zeros((K, K), dtype=np.int64)
    np.add.at(M, (true, pred), 1)
    tp = np.diag(M).astype(float)
    p = tp / np.maximum(M.sum(0), 1)
    r = tp / np.maximum(M.sum(1), 1)
    return float(np.mean(2 * p * r / np.maximum(p + r, 1e-12)))


def fit_multipliers(prob_folds, y_folds, K=10, rounds=6, grid=None):
    """Coordinate ascent on the log multipliers, maximising mean macro F1."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.2, 1.0, 17), np.linspace(1.1, 6.0, 50)])
    w = np.ones(K)

    def score(w):
        return float(np.mean([
            macro_f1_from_counts(yv, np.argmax(P * w, axis=1))
            for P, yv in zip(prob_folds, y_folds)]))

    best = score(w)
    for r in range(rounds):
        improved = False
        for c in range(K):
            cur, trial = w[c], None
            for g in grid:
                w[c] = g
                s = score(w)
                if s > best + 1e-9:
                    best, trial = s, g
            w[c] = trial if trial is not None else cur
            improved |= trial is not None
        log(f"  round {r+1}: macro F1 = {best:.4f}", LOGP)
        if not improved:
            break
    return w, best


def main():
    ensure_dirs()
    open(LOGP, "w").close()
    tb = json.load(open(f"{OUT_DIR}/retune_f1_best_tree.json"))["tree"]

    X, y = load_data()
    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X, y, train_size=TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    for d in (X_tune, X_eval):
        d.reset_index(drop=True, inplace=True)
    y_tune = y_tune.reset_index(drop=True)
    y_eval = y_eval.reset_index(drop=True)
    num = numeric_columns(X_tune)

    def make():
        return Pipeline([("prep", build_ct_preprocessor(num)),
                         ("clf", PowerWeightedTree(
                             alpha=tb["alpha"], criterion=tb["criterion"],
                             max_depth=tb["max_depth"],
                             min_samples_leaf=tb["min_samples_leaf"],
                             ccp_alpha=tb["ccp_alpha"],
                             random_state=RANDOM_SEED))])

    # --- fit multipliers on tuning
    log("collecting out-of-fold probabilities on the tuning partition", LOGP)
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    prob_folds, y_folds = [], []
    for tr, te in skf.split(X_tune, y_tune):
        pipe = make().fit(X_tune.iloc[tr], y_tune.iloc[tr])
        P = pipe.named_steps["clf"].model_.predict_proba(
            pipe.named_steps["prep"].transform(X_tune.iloc[te]))
        prob_folds.append(P)
        y_folds.append(y_tune.iloc[te].to_numpy())

    base = float(np.mean([macro_f1_from_counts(yv, np.argmax(P, axis=1))
                          for P, yv in zip(prob_folds, y_folds)]))
    log(f"tuning partition, plain arg max      : macro F1 = {base:.4f}", LOGP)
    w, tuned = fit_multipliers(prob_folds, y_folds)
    log(f"tuning partition, tuned multipliers  : macro F1 = {tuned:.4f}", LOGP)
    log("multipliers: " + ", ".join(f"{CLASS_NAMES[c]}={w[c]:.3f}"
                                    for c in CLASS_ORDER), LOGP)

    # --- apply unchanged to evaluation
    log("\napplying the fitted multipliers to the evaluation partition", LOGP)
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf10.split(X_eval, y_eval), 1):
        pipe = make().fit(X_eval.iloc[tr], y_eval.iloc[tr])
        P = pipe.named_steps["clf"].model_.predict_proba(
            pipe.named_steps["prep"].transform(X_eval.iloc[te]))
        yv = y_eval.iloc[te].to_numpy()
        rows.append(dict(fold=fold,
                         plain=macro_f1_from_counts(yv, np.argmax(P, axis=1)),
                         tuned=macro_f1_from_counts(yv, np.argmax(P * w, axis=1))))
        log(f"  fold {fold:2d}: plain={rows[-1]['plain']:.4f} "
            f"tuned={rows[-1]['tuned']:.4f}", LOGP)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/threshold_folds.csv", index=False)
    np.save(f"{OUT_DIR}/threshold_multipliers.npy", w)
    log(f"\nEVALUATION PARTITION", LOGP)
    log(f"  plain arg max      : {df.plain.mean():.4f} +- {df.plain.std(ddof=1):.4f}", LOGP)
    log(f"  tuned multipliers  : {df.tuned.mean():.4f} +- {df.tuned.std(ddof=1):.4f}", LOGP)
    log(f"  gain               : {df.tuned.mean()-df.plain.mean():+.4f}", LOGP)
    from scipy import stats
    print(f"Wilcoxon p = {stats.wilcoxon(df.plain, df.tuned).pvalue:.5f}")


if __name__ == "__main__":
    main()
