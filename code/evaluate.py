"""Evaluation and statistical analysis.

Stratified ten-fold cross-validation of four configurations on the evaluation
partition, which no tuning decision has seen: k-NN, k-NN + SMOTE, tree, and
tree + SMOTE.

Resampling happens inside the cross-validation loop, on the training fold only,
through an imbalanced-learn pipeline. Resampling before the split would put
synthetic rows built from test-fold observations into training and inflate every
figure.

  results/final_folds.csv, final_summary.csv, final_perclass.csv
  results/final_confusion_*.csv
  results/stats_friedman.csv, stats_pairwise.csv, stats_normality.csv
  figures/fig_perfold.pdf, fig_perclass_recall.pdf, fig_confusion.pdf,
  figures/fig_final_comparison.pdf
"""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from common import (RANDOM_SEED, OUT_DIR, FIG_DIR, CLASS_NAMES, CLASS_ORDER,
                       ensure_dirs, load_data, numeric_columns,
                       build_knn_preprocessor, build_ct_preprocessor,
                       Timer, log)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})

FOLDS = 10
TUNE_FRACTION = 0.25
SMOTE_CAP = 0.25       # minority classes raised to at most this share of the majority
TRAIN_PROBE = 10_000   # stratified subsample used for the training-set estimate
LOGP = f"{OUT_DIR}/final_log.txt"


def capped_strategy(y) -> dict:
    """SMOTE targets, capped.

    Balancing fully would mean generating roughly 62000 synthetic Worms from
    157 real ones in every training fold, a factor above 390. At that point the
    synthetic rows are interpolations among a handful of points, which adds
    little and costs a lot of runtime. Every minority class is raised to at
    most SMOTE_CAP of the majority count instead.
    """
    counts = pd.Series(y).value_counts()
    target = int(math.ceil(SMOTE_CAP * counts.max()))
    return {c: max(int(n), target) for c, n in counts.items()}


class CappedSMOTE(SMOTE):
    """SMOTE whose sampling targets are recomputed for each training fold."""

    def fit_resample(self, X, y):
        self.sampling_strategy = capped_strategy(y)
        # k_neighbors cannot exceed the size of the smallest class minus one
        smallest = int(pd.Series(y).value_counts().min())
        self.k_neighbors = max(1, min(5, smallest - 1))
        return super().fit_resample(X, y)


def evaluate_config(name: str, make_pipe, X, y, folds: int = FOLDS) -> tuple:
    """Stratified cross-validation returning per-fold metrics and a confusion
    matrix.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    rows, cm_total = [], np.zeros((10, 10), dtype=np.int64)

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe = make_pipe()

        with Timer() as t:
            pipe.fit(Xtr, ytr)
        fit_s = t.seconds
        with Timer() as t:
            pred = pipe.predict(Xte)
        pred_s = t.seconds

        # Training score comes off a stratified subsample. Scoring the whole
        # training fold with k-NN costs about nine times a test fold and would
        # eat the runtime; TRAIN_PROBE rows already put the standard error
        # under half a point.
        if len(Xtr) > TRAIN_PROBE:
            probe_idx, _ = train_test_split(
                np.arange(len(Xtr)), train_size=TRAIN_PROBE, stratify=ytr,
                random_state=RANDOM_SEED)
        else:
            probe_idx = np.arange(len(Xtr))
        Xprobe, yprobe = Xtr.iloc[probe_idx], ytr.iloc[probe_idx]
        train_pred = pipe.predict(Xprobe)
        cm = confusion_matrix(yte, pred, labels=CLASS_ORDER)
        cm_total += cm
        per_class_recall = np.divide(np.diag(cm), cm.sum(axis=1),
                                     out=np.zeros(10), where=cm.sum(axis=1) > 0)

        row = dict(
            configuration=name, fold=fold,
            accuracy=accuracy_score(yte, pred),
            balanced_accuracy=balanced_accuracy_score(yte, pred),
            f1_macro=f1_score(yte, pred, average="macro", zero_division=0),
            precision_macro=precision_score(yte, pred, average="macro", zero_division=0),
            recall_macro=recall_score(yte, pred, average="macro", zero_division=0),
            mcc=matthews_corrcoef(yte, pred),
            kappa=cohen_kappa_score(yte, pred),
            train_accuracy=accuracy_score(yprobe, train_pred),
            train_balanced_accuracy=balanced_accuracy_score(yprobe, train_pred),
            fit_seconds=fit_s, predict_seconds=pred_s,
        )
        for c in CLASS_ORDER:
            row[f"recall_{CLASS_NAMES[c]}"] = per_class_recall[c]
        rows.append(row)
        log(f"{name} fold {fold:2d}: bal={row['balanced_accuracy']:.4f} "
            f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} "
            f"mcc={row['mcc']:.4f} ({fit_s:.1f}s fit, {pred_s:.1f}s predict)", LOGP)

    return pd.DataFrame(rows), cm_total


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    ensure_dirs()
    open(LOGP, "w").close()

    with open(f"{OUT_DIR}/tune_best.json") as f:
        best = json.load(f)
    kb, tb = best["knn"], best["tree"]
    log(f"tuned k-NN  : {kb}", LOGP)
    log(f"tuned tree  : {tb}", LOGP)

    X, y = load_data()
    _, X_eval, _, y_eval = train_test_split(
        X, y, train_size=TUNE_FRACTION, stratify=y, random_state=RANDOM_SEED)
    X_eval, y_eval = X_eval.reset_index(drop=True), y_eval.reset_index(drop=True)
    num = numeric_columns(X_eval)
    log(f"evaluation partition {X_eval.shape}, folds={FOLDS}", LOGP)

    def knn_clf():
        return KNeighborsClassifier(n_neighbors=kb["k"], weights=kb["weights"],
                                    metric=kb["metric"], n_jobs=-1)

    def ct_clf():
        return DecisionTreeClassifier(
            criterion=tb["criterion"], max_depth=tb["max_depth"],
            min_samples_leaf=tb["min_samples_leaf"],
            class_weight=tb["class_weight"], ccp_alpha=tb["ccp_alpha"],
            random_state=RANDOM_SEED)

    configs = {
        "k-NN": lambda: Pipeline([("prep", build_knn_preprocessor(num)),
                                  ("clf", knn_clf())]),
        "k-NN + SMOTE": lambda: ImbPipeline([("prep", build_knn_preprocessor(num)),
                                             ("smote", CappedSMOTE(random_state=RANDOM_SEED)),
                                             ("clf", knn_clf())]),
        "tree": lambda: Pipeline([("prep", build_ct_preprocessor(num)),
                                  ("clf", ct_clf())]),
        "tree + SMOTE": lambda: ImbPipeline([("prep", build_ct_preprocessor(num)),
                                             ("smote", CappedSMOTE(random_state=RANDOM_SEED)),
                                             ("clf", ct_clf())]),
    }

    all_folds, confusions = [], {}
    for name, maker in configs.items():
        log(f"=== {name} ===", LOGP)
        df, cm = evaluate_config(name, maker, X_eval, y_eval)
        all_folds.append(df)
        confusions[name] = cm
        pd.DataFrame(cm, index=[CLASS_NAMES[c] for c in CLASS_ORDER],
                     columns=[CLASS_NAMES[c] for c in CLASS_ORDER]).to_csv(
            f"{OUT_DIR}/final_confusion_{name.replace(' ', '_').replace('+', 'plus')}.csv")

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(f"{OUT_DIR}/final_folds.csv", index=False)

    metrics = ["balanced_accuracy", "f1_macro", "mcc", "kappa", "accuracy",
               "precision_macro", "recall_macro", "train_accuracy",
               "train_balanced_accuracy", "fit_seconds", "predict_seconds"]
    summary = folds.groupby("configuration")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex(list(configs))
    summary.to_csv(f"{OUT_DIR}/final_summary.csv")
    log("\n" + summary[[f"{m}_mean" for m in metrics[:5]]].round(4).to_string(), LOGP)

    perclass = folds.groupby("configuration")[
        [f"recall_{CLASS_NAMES[c]}" for c in CLASS_ORDER]].agg(["mean", "std"])
    perclass.columns = [f"{a}_{b}" for a, b in perclass.columns]
    perclass.reindex(list(configs)).to_csv(f"{OUT_DIR}/final_perclass.csv")

    # --- statistics
    statistical_analysis(folds, list(configs))

    # --- figures
    figure_perfold(folds, list(configs))
    figure_perclass(folds, list(configs))
    figure_confusion(confusions, list(configs))
    figure_final_comparison(folds, list(configs))
    log("evaluation complete", LOGP)


def statistical_analysis(folds: pd.DataFrame, names: list[str],
                         metric: str = "balanced_accuracy",
                         alpha: float = 0.05) -> None:
    wide = folds.pivot(index="fold", columns="configuration", values=metric)[names]

    chi2, p_fried = stats.friedmanchisquare(*[wide[c].values for c in names])
    pd.DataFrame([dict(metric=metric, statistic=chi2, p_value=p_fried,
                       alpha=alpha, reject_h0=bool(p_fried < alpha),
                       n_groups=len(names), n_folds=len(wide))]).to_csv(
        f"{OUT_DIR}/stats_friedman.csv", index=False)
    log(f"Friedman: chi2={chi2:.4f} p={p_fried:.3e} "
        f"{'reject' if p_fried < alpha else 'fail to reject'} H0", LOGP)

    norm_rows, pair_rows, raw_p = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = wide[a].values - wide[b].values
            w_stat, w_p = stats.wilcoxon(wide[a].values, wide[b].values)
            t_stat, t_p = stats.ttest_rel(wide[a].values, wide[b].values)
            sh_stat, sh_p = stats.shapiro(d)
            norm_rows.append(dict(comparison=f"{a} vs {b}", statistic=sh_stat,
                                  p_value=sh_p,
                                  normality_rejected=bool(sh_p < alpha)))
            pair_rows.append(dict(
                comparison=f"{a} vs {b}", mean_a=wide[a].mean(), mean_b=wide[b].mean(),
                mean_difference=float(np.mean(d)),
                wilcoxon_statistic=w_stat, wilcoxon_p=w_p,
                paired_t_statistic=t_stat, paired_t_p=t_p))
            raw_p.append(w_p)

    adj = holm(raw_p)
    for r, a_ in zip(pair_rows, adj):
        r["holm_p"] = a_
        r["significant"] = bool(a_ < alpha)
    pd.DataFrame(pair_rows).to_csv(f"{OUT_DIR}/stats_pairwise.csv", index=False)
    pd.DataFrame(norm_rows).to_csv(f"{OUT_DIR}/stats_normality.csv", index=False)
    for r in pair_rows:
        log(f"  {r['comparison']:24s} diff={r['mean_difference']:+.4f} "
            f"wilcoxon p={r['wilcoxon_p']:.4e} holm={r['holm_p']:.4e} "
            f"{'significant' if r['significant'] else 'not significant'}", LOGP)


MARKERS = {"k-NN": ("o-", "0.10"), "k-NN + SMOTE": ("s--", "0.10"),
           "tree": ("^-", "0.55"), "tree + SMOTE": ("v--", "0.55")}


def figure_perfold(folds: pd.DataFrame, names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for n in names:
        sub = folds[folds.configuration == n].sort_values("fold")
        st, col = MARKERS[n]
        ax.plot(sub["fold"], sub["balanced_accuracy"], st, color=col,
                markersize=3.5, linewidth=1, label=n)
    ax.set_xlabel("fold")
    ax.set_ylabel("balanced accuracy")
    ax.set_xticks(range(1, 11))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(f"{FIG_DIR}/fig_perfold.pdf")
    plt.close(fig)


def figure_perclass(folds: pd.DataFrame, names: list[str]) -> None:
    cols = [f"recall_{CLASS_NAMES[c]}" for c in CLASS_ORDER]
    order = [0, 9, 4, 6, 3, 1, 5, 2, 8, 7]        # descending class frequency
    labels = [CLASS_NAMES[c] for c in order]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.2
    xs = np.arange(len(order))
    shades = ["0.15", "0.40", "0.62", "0.82"]
    for i, n in enumerate(names):
        sub = folds[folds.configuration == n]
        means = [sub[f"recall_{CLASS_NAMES[c]}"].mean() for c in order]
        errs = [sub[f"recall_{CLASS_NAMES[c]}"].std(ddof=1) for c in order]
        ax.bar(xs + (i - 1.5) * width, means, width, yerr=errs, label=n,
               color=shades[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")
    fig.savefig(f"{FIG_DIR}/fig_perclass_recall.pdf")
    plt.close(fig)


def figure_confusion(confusions: dict, names: list[str]) -> None:
    show = [names[0], names[2]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    labels = [CLASS_NAMES[c] for c in CLASS_ORDER]
    for ax, n in zip(axes, show):
        cm = confusions[n].astype(float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("predicted class", fontsize=9)
        ax.set_ylabel("actual class", fontsize=9)
        ax.set_title(n, fontsize=10)
        for i in range(10):
            for j in range(10):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0"),
                            ha="center", va="center", fontsize=4.5,
                            color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="proportion of actual class")
    fig.savefig(f"{FIG_DIR}/fig_confusion.pdf")
    plt.close(fig)


def figure_final_comparison(folds: pd.DataFrame, names: list[str]) -> None:
    metrics = ["balanced_accuracy", "f1_macro", "mcc", "accuracy"]
    labels = ["balanced accuracy", "macro F1", "correlation coefficient", "accuracy"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = np.arange(len(metrics))
    width = 0.2
    shades = ["0.15", "0.40", "0.62", "0.82"]
    for i, n in enumerate(names):
        sub = folds[folds.configuration == n]
        means = [sub[m].mean() for m in metrics]
        errs = [sub[m].std(ddof=1) for m in metrics]
        ax.bar(xs + (i - 1.5) * width, means, width, yerr=errs, label=n,
               color=shades[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.savefig(f"{FIG_DIR}/fig_final_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
