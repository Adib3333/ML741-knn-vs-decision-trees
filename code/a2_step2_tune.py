"""Step 2: control parameter tuning.

The data is split once, with stratification, into a 25 per cent tuning partition
and a 75 per cent evaluation partition. Nothing here touches the evaluation
partition, so the reported numbers are never selected on the folds they are
reported from.

Inside the tuning partition the search runs in two stages: a coarse grid over the
whole space, then a finer grid around the stage 1 winner. Both use stratified
five-fold cross-validation with balanced accuracy as the criterion.

  results/tune_knn_stage1.csv, tune_knn_stage2.csv
  results/tune_ct_stage1.csv, tune_ct_stage2.csv
  results/tune_best.json, split_sizes.csv
  figures/fig_tune_knn_k.pdf, fig_tune_ct_depth.pdf
"""
from __future__ import annotations

import json
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score

from a2_common import (RANDOM_SEED, OUT_DIR, FIG_DIR, ensure_dirs, load_data,
                       numeric_columns, build_knn_preprocessor,
                       build_ct_preprocessor, Timer, log)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})

TUNE_FRACTION = 0.25
TUNE_FOLDS = 5
LOGP = f"{OUT_DIR}/tune_log.txt"


def as_depth(v):
    """Normalise a grid value for max_depth back to int or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return int(v)


def as_weight(v):
    """Normalise a grid value for class_weight back to str or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def evaluate(pipe, X, y, folds: int, seed: int = RANDOM_SEED) -> dict:
    """Stratified cross-validation returning balanced accuracy and macro F1."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    bal, mf1, fit_s, pred_s = [], [], [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        with Timer() as t:
            pipe.fit(Xtr, ytr)
        fit_s.append(t.seconds)
        with Timer() as t:
            pred = pipe.predict(Xte)
        pred_s.append(t.seconds)
        bal.append(balanced_accuracy_score(yte, pred))
        mf1.append(f1_score(yte, pred, average="macro", zero_division=0))
    return dict(bal_mean=float(np.mean(bal)), bal_std=float(np.std(bal, ddof=1)),
                f1_mean=float(np.mean(mf1)), f1_std=float(np.std(mf1, ddof=1)),
                fit_seconds=float(np.mean(fit_s)),
                predict_seconds=float(np.mean(pred_s)))


def run_grid(name: str, make_pipe, grid: dict, X, y, folds: int) -> pd.DataFrame:
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log(f"{name}: {len(combos)} configurations", LOGP)
    rows = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        res = evaluate(make_pipe(**params), X, y, folds)
        rows.append({**params, **res})
        log(f"  [{i:3d}/{len(combos)}] {params} -> "
            f"bal={res['bal_mean']:.4f}+-{res['bal_std']:.4f} "
            f"f1={res['f1_mean']:.4f} ({res['fit_seconds']:.1f}s fit, "
            f"{res['predict_seconds']:.1f}s predict)", LOGP)
    return pd.DataFrame(rows).sort_values("bal_mean", ascending=False)


def main() -> None:
    ensure_dirs()
    open(LOGP, "w").close()

    X, y = load_data()
    num = numeric_columns(X)
    log(f"loaded X={X.shape}  y={y.shape}  numeric={len(num)}", LOGP)

    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X, y, train_size=TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    for d in (X_tune, X_eval, y_tune, y_eval):
        d.reset_index(drop=True, inplace=True)
    log(f"tuning partition {X_tune.shape[0]}  evaluation partition {X_eval.shape[0]}", LOGP)

    pd.DataFrame([
        dict(partition="tuning", instances=len(X_tune),
             fraction=round(len(X_tune) / len(X), 4)),
        dict(partition="evaluation", instances=len(X_eval),
             fraction=round(len(X_eval) / len(X), 4)),
    ]).to_csv(f"{OUT_DIR}/split_sizes.csv", index=False)

    # --- k-NN
    def knn_pipe(k, weights, metric):
        return Pipeline([
            ("prep", build_knn_preprocessor(num)),
            ("clf", KNeighborsClassifier(n_neighbors=k, weights=weights,
                                         metric=metric, n_jobs=-1)),
        ])

    knn_s1 = run_grid(
        "k-NN stage 1", knn_pipe,
        dict(k=[1, 3, 5, 7, 9, 11, 15, 21, 31, 51],
             weights=["uniform", "distance"],
             metric=["euclidean", "manhattan"]),
        X_tune, y_tune, TUNE_FOLDS)
    knn_s1.to_csv(f"{OUT_DIR}/tune_knn_stage1.csv", index=False)
    b = knn_s1.iloc[0]
    log(f"k-NN stage 1 best: k={b.k} weights={b.weights} metric={b.metric} "
        f"bal={b.bal_mean:.4f}", LOGP)

    kbest = int(b.k)
    neigh = sorted({max(1, kbest - 2), max(1, kbest - 1), kbest,
                    kbest + 1, kbest + 2, kbest + 4})
    knn_s2 = run_grid(
        "k-NN stage 2", knn_pipe,
        dict(k=neigh, weights=[b.weights], metric=[b.metric]),
        X_tune, y_tune, TUNE_FOLDS)
    knn_s2.to_csv(f"{OUT_DIR}/tune_knn_stage2.csv", index=False)
    b2 = knn_s2.iloc[0]
    knn_best = dict(k=int(b2.k), weights=str(b2.weights), metric=str(b2.metric),
                    bal_mean=float(b2.bal_mean), bal_std=float(b2.bal_std))
    log(f"k-NN final: {knn_best}", LOGP)

    # --- classification tree
    def ct_pipe(criterion, max_depth, min_samples_leaf, class_weight, ccp_alpha):
        # pandas stores a column of None mixed with ints as float, so both
        # fields get pushed back to their declared types
        return Pipeline([
            ("prep", build_ct_preprocessor(num)),
            ("clf", DecisionTreeClassifier(
                criterion=str(criterion), max_depth=as_depth(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                class_weight=as_weight(class_weight),
                ccp_alpha=float(ccp_alpha), random_state=RANDOM_SEED)),
        ])

    ct_s1 = run_grid(
        "tree stage 1", ct_pipe,
        dict(criterion=["gini", "entropy"],
             max_depth=[5, 10, 15, 20, 30, None],
             min_samples_leaf=[1, 5, 20],
             class_weight=[None, "balanced"],
             ccp_alpha=[0.0]),
        X_tune, y_tune, TUNE_FOLDS)
    ct_s1.to_csv(f"{OUT_DIR}/tune_ct_stage1.csv", index=False)
    c = ct_s1.iloc[0]
    log(f"tree stage 1 best: criterion={c.criterion} depth={c.max_depth} "
        f"leaf={c.min_samples_leaf} weight={c.class_weight} bal={c.bal_mean:.4f}", LOGP)

    depth0 = as_depth(c.max_depth)
    depths = sorted({d for d in
                     ([None] if depth0 is None else
                      [max(1, depth0 - 3), depth0, depth0 + 3, None])},
                    key=lambda d: (d is None, d))
    ct_s2 = run_grid(
        "tree stage 2", ct_pipe,
        dict(criterion=[str(c.criterion)],
             max_depth=depths,
             min_samples_leaf=[int(c.min_samples_leaf)],
             class_weight=[as_weight(c.class_weight)],
             ccp_alpha=[0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]),
        X_tune, y_tune, TUNE_FOLDS)
    ct_s2.to_csv(f"{OUT_DIR}/tune_ct_stage2.csv", index=False)
    c2 = ct_s2.iloc[0]
    ct_best = dict(criterion=str(c2.criterion),
                   max_depth=as_depth(c2.max_depth),
                   min_samples_leaf=int(c2.min_samples_leaf),
                   class_weight=as_weight(c2.class_weight),
                   ccp_alpha=float(c2.ccp_alpha),
                   bal_mean=float(c2.bal_mean), bal_std=float(c2.bal_std))
    log(f"tree final: {ct_best}", LOGP)

    with open(f"{OUT_DIR}/tune_best.json", "w") as f:
        json.dump(dict(knn=knn_best, tree=ct_best,
                       tune_fraction=TUNE_FRACTION, tune_folds=TUNE_FOLDS,
                       seed=RANDOM_SEED), f, indent=2)

    figure_knn_k(knn_s1)
    figure_ct_depth(ct_s1)
    log("tuning complete", LOGP)


def figure_knn_k(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    styles = {("uniform", "euclidean"): ("o-", "0.15"),
              ("distance", "euclidean"): ("s--", "0.15"),
              ("uniform", "manhattan"): ("^-", "0.55"),
              ("distance", "manhattan"): ("v--", "0.55")}
    for (w, m), (st, col) in styles.items():
        sub = df[(df.weights == w) & (df.metric == m)].sort_values("k")
        if len(sub):
            ax.errorbar(sub["k"], sub["bal_mean"], yerr=sub["bal_std"], fmt=st,
                        color=col, markersize=3.5, linewidth=1, capsize=2,
                        label=f"{w}, {m}")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 7, 9, 11, 15, 21, 31, 51])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of neighbours")
    ax.set_ylabel("balanced accuracy")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.savefig(f"{FIG_DIR}/fig_tune_knn_k.pdf")
    plt.close(fig)


def figure_ct_depth(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    d = df.copy()
    d["depth_plot"] = d["max_depth"].fillna(40)
    for cw, st, col in [(None, "o-", "0.15"), ("balanced", "s--", "0.55")]:
        sub = d[(d.class_weight.isna() if cw is None else d.class_weight == cw)]
        sub = sub[sub.min_samples_leaf == 1]
        sub = sub[sub.criterion == "gini"].sort_values("depth_plot")
        if len(sub):
            ax.errorbar(sub["depth_plot"], sub["bal_mean"], yerr=sub["bal_std"],
                        fmt=st, color=col, markersize=3.5, linewidth=1, capsize=2,
                        label=f"class weight: {cw if cw else 'none'}")
    ax.set_xlabel("maximum tree depth")
    ax.set_ylabel("balanced accuracy")
    ax.set_xticks([5, 10, 15, 20, 30, 40])
    ax.set_xticklabels(["5", "10", "15", "20", "30", "none"])
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.savefig(f"{FIG_DIR}/fig_tune_ct_depth.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
