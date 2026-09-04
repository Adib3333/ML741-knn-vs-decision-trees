"""Normalisation and redundancy study for k-NN.

The first ablation said min-max lost to z-score, and that dropping correlated
features hurt rather than helped. Both go against the textbook expectation, so
both are looked at directly here.

What is being tested: neither min-max nor z-score copes with a skewness of 173.2,
because both are anchored on statistics the upper tail controls. A rank transform
has no such anchor.

  results/scaling_study.csv, redundancy_study.csv
  figures/fig_scaling_study.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.metrics import balanced_accuracy_score, f1_score

from common import (RANDOM_SEED, OUT_DIR, FIG_DIR, NOMINAL,
                       MIN_LEVEL_FREQUENCY, CORR_THRESHOLD, ensure_dirs,
                       load_data, numeric_columns, RareLevelGrouper,
                       CorrelationFilter, ConstantFilter, Timer, log)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})

FOLDS = 5
FRACTION = 0.35
LOGP = f"{OUT_DIR}/scaling_log.txt"

# tuned k-NN control parameters from tune.py
K, WEIGHTS, METRIC = 6, "distance", "manhattan"


def signed_log(X):
    """Logarithmic compression that tolerates zero, applied feature-wise."""
    return np.log1p(np.clip(X, 0, None))


SCALERS = {
    "min-max": lambda: MinMaxScaler(),
    "z-score": lambda: StandardScaler(),
    "robust to outliers": lambda: RobustScaler(),
    "logarithmic, then min-max": lambda: Pipeline([
        ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
        ("scale", MinMaxScaler())]),
    "rank based": lambda: QuantileTransformer(
        output_distribution="uniform", n_quantiles=1000,
        subsample=200_000, random_state=RANDOM_SEED),
}


def make_prep(numeric, scaler, corr: bool):
    steps = [("constant", ConstantFilter())]
    if corr:
        steps.append(("corr", CorrelationFilter(threshold=CORR_THRESHOLD)))
    steps.append(("scale", scaler))
    return ColumnTransformer(
        [("num", Pipeline(steps), numeric),
         ("nom", Pipeline([
             ("rare", RareLevelGrouper(columns=NOMINAL,
                                       min_frequency=MIN_LEVEL_FREQUENCY)),
             ("onehot", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False))]), NOMINAL)],
        remainder="drop")


def score(prep, X, y) -> dict:
    pipe = Pipeline([("prep", prep),
                     ("clf", KNeighborsClassifier(n_neighbors=K, weights=WEIGHTS,
                                                  metric=METRIC, n_jobs=-1))])
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    bal, mf1, secs = [], [], []
    for tr, te in skf.split(X, y):
        with Timer() as t:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
        secs.append(t.seconds)
        bal.append(balanced_accuracy_score(y.iloc[te], pred))
        mf1.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
    return dict(bal_mean=float(np.mean(bal)), bal_std=float(np.std(bal, ddof=1)),
                f1_mean=float(np.mean(mf1)), f1_std=float(np.std(mf1, ddof=1)),
                seconds=float(np.mean(secs)))


def main() -> None:
    ensure_dirs()
    open(LOGP, "w").close()

    X, y = load_data()
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=0.25, stratify=y, random_state=RANDOM_SEED)
    Xa, _, ya, _ = train_test_split(
        X_tune.reset_index(drop=True), y_tune.reset_index(drop=True),
        train_size=FRACTION, stratify=y_tune, random_state=RANDOM_SEED)
    Xa, ya = Xa.reset_index(drop=True), ya.reset_index(drop=True)
    num = numeric_columns(Xa)
    log(f"sample {Xa.shape}", LOGP)

    rows = []
    for name, maker in SCALERS.items():
        r = score(make_prep(num, maker(), corr=False), Xa, ya)
        rows.append(dict(transformation=name, **r))
        log(f"scaling | {name:42s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
            f"f1={r['f1_mean']:.4f} ({r['seconds']:.1f}s)", LOGP)
    scaling = pd.DataFrame(rows).sort_values("bal_mean", ascending=False)
    scaling.to_csv(f"{OUT_DIR}/scaling_study.csv", index=False)
    winner = scaling.iloc[0]["transformation"]
    log(f"best transformation: {winner}", LOGP)

    rows = []
    for thr, label in [(None, "every feature retained"),
                       (0.99, "correlation filter at 0.99"),
                       (0.95, "correlation filter at 0.95"),
                       (0.90, "correlation filter at 0.90")]:
        prep = make_prep(num, SCALERS[winner](), corr=thr is not None)
        if thr is not None:
            prep.transformers[0][1].set_params(corr__threshold=thr)
        r = score(prep, Xa, ya)
        rows.append(dict(setting=label, threshold=thr, **r))
        log(f"redundancy | {label:32s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
            f"f1={r['f1_mean']:.4f}", LOGP)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/redundancy_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    d = scaling.iloc[::-1]
    ax.barh(range(len(d)), d["bal_mean"], xerr=d["bal_std"], color="0.45",
            edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.7, capsize=2))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["transformation"], fontsize=7.5)
    ax.set_xlabel("balanced accuracy")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    fig.savefig(f"{FIG_DIR}/fig_scaling_study.pdf")
    plt.close(fig)
    log("scaling study complete", LOGP)


if __name__ == "__main__":
    main()
