"""Step 3: preprocessing ablation.

The brief wants a justification for every step applied and every issue left
alone, so each transformation is removed or added one at a time and the effect on
balanced accuracy measured. Runs on the tuning partition only.

  results/ablation_knn.csv, ablation_ct.csv
  figures/fig_ablation.pdf
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer, StandardScaler)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from a2_common import (RANDOM_SEED, OUT_DIR, FIG_DIR, NOMINAL, ID_COL,
                       MIN_LEVEL_FREQUENCY, CORR_THRESHOLD, DATA_PATH,
                       ensure_dirs, load_data, numeric_columns,
                       build_knn_preprocessor, build_ct_preprocessor,
                       RareLevelGrouper, CorrelationFilter, ConstantFilter,
                       Timer, log)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})

ABLATION_FOLDS = 5
ABLATION_FRACTION = 0.35     # of the tuning partition, for runtime control
LOGP = f"{OUT_DIR}/ablation_log.txt"


def score(pipe, X, y, folds=ABLATION_FOLDS) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    bal, mf1, secs = [], [], []
    for tr, te in skf.split(X, y):
        with Timer() as t:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
        secs.append(t.seconds)
        bal.append(balanced_accuracy_score(y.iloc[te], pred))
        mf1.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
    return dict(bal_mean=float(np.mean(bal)), bal_std=float(np.std(bal, ddof=1)),
                f1_mean=float(np.mean(mf1)), seconds=float(np.mean(secs)))


# --- preprocessor variants
def knn_prep(numeric, scale="rank", corr=False, group=True):
    steps = [("constant", ConstantFilter())]
    if corr:
        steps.append(("corr", CorrelationFilter(threshold=CORR_THRESHOLD)))
    if scale == "rank":
        steps.append(("scale", QuantileTransformer(
            output_distribution="uniform", n_quantiles=1000,
            subsample=200_000, random_state=RANDOM_SEED)))
    elif scale == "minmax":
        steps.append(("scale", MinMaxScaler()))
    elif scale == "zscore":
        steps.append(("scale", StandardScaler()))
    numeric_branch = Pipeline(steps)

    nom_steps = []
    if group:
        nom_steps.append(("rare", RareLevelGrouper(columns=NOMINAL,
                                                   min_frequency=MIN_LEVEL_FREQUENCY)))
    nom_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False)))
    return ColumnTransformer([("num", numeric_branch, numeric),
                              ("nom", Pipeline(nom_steps), NOMINAL)],
                             remainder="drop")


def ct_prep(numeric, scale=None, corr=False, onehot=False):
    steps = []
    if corr:
        steps.append(("corr", CorrelationFilter(threshold=CORR_THRESHOLD)))
    if scale == "minmax":
        steps.append(("scale", MinMaxScaler()))
    numeric_branch = Pipeline(steps) if steps else "passthrough"

    if onehot:
        nom = Pipeline([("rare", RareLevelGrouper(columns=NOMINAL,
                                                  min_frequency=MIN_LEVEL_FREQUENCY)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
    else:
        nom = Pipeline([("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                   unknown_value=-1))])
    return ColumnTransformer([("num", numeric_branch, numeric),
                              ("nom", nom, NOMINAL)], remainder="drop")


def main() -> None:
    ensure_dirs()
    open(LOGP, "w").close()

    with open(f"{OUT_DIR}/tune_best.json") as f:
        best = json.load(f)
    kb, tb = best["knn"], best["tree"]
    log(f"tuned k-NN {kb}", LOGP)
    log(f"tuned tree {tb}", LOGP)

    # id is kept here only to measure what keeping it costs; every other
    # script drops it at load time
    X, y = load_data(keep_id=True)
    X_tune, _, y_tune, _ = train_test_split(
        X, y, train_size=0.25, stratify=y, random_state=RANDOM_SEED)
    X_tune, y_tune = X_tune.reset_index(drop=True), y_tune.reset_index(drop=True)
    Xa_id, _, ya, _ = train_test_split(
        X_tune, y_tune, train_size=ABLATION_FRACTION, stratify=y_tune,
        random_state=RANDOM_SEED)
    Xa_id, ya = Xa_id.reset_index(drop=True), ya.reset_index(drop=True)
    Xa = Xa_id.drop(columns=[ID_COL])
    num = numeric_columns(Xa)
    num_id = numeric_columns(Xa_id)
    log(f"ablation sample {Xa.shape}", LOGP)

    # --- k-NN
    def knn(prep):
        return Pipeline([("prep", prep),
                         ("clf", KNeighborsClassifier(n_neighbors=kb["k"],
                                                      weights=kb["weights"],
                                                      metric=kb["metric"],
                                                      n_jobs=-1))])

    knn_rows = []
    variants = [
        ("selected pipeline", knn_prep(num), Xa),
        ("no scaling", knn_prep(num, scale=None), Xa),
        ("min-max instead of rank", knn_prep(num, scale="minmax"), Xa),
        ("correlation filter added", knn_prep(num, corr=True), Xa),
        ("no rare-level grouping", knn_prep(num, group=False), Xa),
    ]
    for name, prep, data in variants:
        r = score(knn(prep), data, ya)
        knn_rows.append(dict(variant=name, **r))
        log(f"k-NN | {name:32s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
            f"f1={r['f1_mean']:.4f} ({r['seconds']:.1f}s)", LOGP)

    r = score(knn(knn_prep(num_id)), Xa_id, ya)
    knn_rows.append(dict(variant="identifier retained", **r))
    log(f"k-NN | {'identifier retained':32s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
        f"f1={r['f1_mean']:.4f} ({r['seconds']:.1f}s)", LOGP)

    pd.DataFrame(knn_rows).to_csv(f"{OUT_DIR}/ablation_knn.csv", index=False)

    # --- tree
    def ct(prep):
        return Pipeline([("prep", prep),
                         ("clf", DecisionTreeClassifier(
                             criterion=tb["criterion"], max_depth=tb["max_depth"],
                             min_samples_leaf=tb["min_samples_leaf"],
                             class_weight=tb["class_weight"],
                             ccp_alpha=tb["ccp_alpha"],
                             random_state=RANDOM_SEED))])

    ct_rows = []
    for name, prep in [
        ("selected pipeline", ct_prep(num)),
        ("min-max scaling added", ct_prep(num, scale="minmax")),
        ("correlation filter added", ct_prep(num, corr=True)),
        ("one-hot instead of ordinal", ct_prep(num, onehot=True)),
    ]:
        r = score(ct(prep), Xa, ya)
        ct_rows.append(dict(variant=name, **r))
        log(f"tree | {name:32s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
            f"f1={r['f1_mean']:.4f} ({r['seconds']:.1f}s)", LOGP)

    r = score(ct(ct_prep(num_id)), Xa_id, ya)
    ct_rows.append(dict(variant="identifier retained", **r))
    log(f"tree | {'identifier retained':32s} bal={r['bal_mean']:.4f}+-{r['bal_std']:.4f} "
        f"f1={r['f1_mean']:.4f} ({r['seconds']:.1f}s)", LOGP)

    pd.DataFrame(ct_rows).to_csv(f"{OUT_DIR}/ablation_ct.csv", index=False)

    figure_ablation(pd.DataFrame(knn_rows), pd.DataFrame(ct_rows))
    log("ablation complete", LOGP)


def figure_ablation(knn: pd.DataFrame, ct: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=False)
    for ax, df, title in ((axes[0], knn, "k-NN"), (axes[1], ct, "tree")):
        base = df.iloc[0]["bal_mean"]
        d = df.iloc[::-1]
        colours = ["0.15" if v == "selected pipeline" else "0.6"
                   for v in d["variant"]]
        ax.barh(range(len(d)), d["bal_mean"], xerr=d["bal_std"],
                color=colours, edgecolor="black", linewidth=0.4,
                error_kw=dict(lw=0.7, capsize=2))
        ax.axvline(base, color="black", linestyle=":", linewidth=0.8)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["variant"], fontsize=8)
        ax.set_xlabel("balanced accuracy")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
