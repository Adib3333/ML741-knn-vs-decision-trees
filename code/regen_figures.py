"""Redraw every figure at a larger text size from the saved result files.

Rule 25b wants figure text to match the body text at 10pt. No figure carrying 39
feature labels can do that literally, so every label goes as large as the space
allows and nothing falls below 7pt.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from common import FIG_DIR
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = os.environ.get("A2_DATA", "networkTraffic.csv")
NOMINAL = ["proto", "state", "service"]
ORDER = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
CLASS_SEQ = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
             "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})


def class_distribution() -> None:
    c = pd.read_csv("results/audit_classes.csv").sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.bar(range(len(c)), c["count"], color="0.35", edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c["class"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("number of instances")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(c["count"]):
        ax.text(i, v * 1.3, f"{v:,}", ha="center", fontsize=7.5, rotation=90)
    ax.set_ylim(top=c["count"].max() * 25)
    fig.savefig(f"{FIG_DIR}/fig_class_distribution.pdf")
    plt.close(fig)


def feature_ranges() -> None:
    s = pd.read_csv("results/audit_summary.csv")
    s = s[(s.kind == "numeric") & (s.feature != "id")].sort_values("rng")
    fig, ax = plt.subplots(figsize=(3.4, 5.4))
    ax.barh(range(len(s)), s["rng"].clip(lower=1e-3), color="0.35",
            edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["feature"], fontsize=7.5)
    ax.set_xlabel("range of observed values")
    ax.set_ylim(-0.8, len(s) - 0.2)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    fig.savefig(f"{FIG_DIR}/fig_feature_ranges.pdf")
    plt.close(fig)


def correlation_heatmap() -> None:
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, low_memory=False)
    cols = [c for c in raw.columns if c not in NOMINAL + ["attack_cat", "id"]]
    C = raw[cols].apply(pd.to_numeric).corr().abs()
    # full text width allows legible labels
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(C.to_numpy(), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(C)))
    ax.set_yticks(range(len(C)))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(C.columns, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("absolute correlation", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(f"{FIG_DIR}/fig_correlation_heatmap.pdf")
    plt.close(fig)


def tuning_knn() -> None:
    df = pd.read_csv("results/tune_knn_final.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    styles = {("uniform", "euclidean"): ("o-", "0.15"),
              ("distance", "euclidean"): ("s--", "0.15"),
              ("uniform", "manhattan"): ("^-", "0.55"),
              ("distance", "manhattan"): ("v--", "0.55")}
    for (w, m), (st, col) in styles.items():
        sub = df[(df.weights == w) & (df.metric == m)].sort_values("k")
        ax.errorbar(sub["k"], sub["bal_mean"], yerr=sub["bal_std"], fmt=st,
                    color=col, markersize=3.5, linewidth=1, capsize=2,
                    label=f"{w}, {m}")
    ax.set_xscale("log")
    # ticks come from the data, so the axis cannot disagree with the grid the
    # report describes
    ks = sorted(df["k"].unique())
    ax.set_xticks(ks)
    ax.set_xticklabels([str(int(k)) for k in ks], fontsize=7)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("number of neighbours")
    ax.set_ylabel("balanced accuracy")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(f"{FIG_DIR}/fig_tune_knn_k.pdf")
    plt.close(fig)


def tuning_tree() -> None:
    d = pd.read_csv("results/tune_ct_stage1.csv")
    d["depth_plot"] = d["max_depth"].fillna(40)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for cw, st, col, lab in [(None, "o-", "0.15", "no class weighting"),
                             ("balanced", "s--", "0.55", "class weighting")]:
        sub = d[(d.class_weight.isna() if cw is None else d.class_weight == cw)]
        sub = sub[(sub.min_samples_leaf == 1) & (sub.criterion == "gini")].sort_values("depth_plot")
        ax.errorbar(sub["depth_plot"], sub["bal_mean"], yerr=sub["bal_std"], fmt=st,
                    color=col, markersize=3.5, linewidth=1, capsize=2, label=lab)
    ax.set_xlabel("maximum tree depth")
    ax.set_ylabel("balanced accuracy")
    ax.set_xticks([5, 10, 15, 20, 30, 40])
    ax.set_xticklabels(["5", "10", "15", "20", "30", "none"], fontsize=9)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(f"{FIG_DIR}/fig_tune_ct_depth.pdf")
    plt.close(fig)


def per_fold() -> None:
    f = pd.read_csv("results/final_folds.csv")
    marks = {"k-NN": ("o-", "0.10"), "k-NN + SMOTE": ("s--", "0.10"),
             "tree": ("^-", "0.55"), "tree + SMOTE": ("v--", "0.55")}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for n in ORDER:
        sub = f[f.configuration == n].sort_values("fold")
        st, col = marks[n]
        ax.plot(sub["fold"], sub["balanced_accuracy"], st, color=col,
                markersize=3.5, linewidth=1, label=n)
    ax.set_xlabel("fold")
    ax.set_ylabel("balanced accuracy")
    ax.set_xticks(range(1, 11))
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="center right")
    fig.savefig(f"{FIG_DIR}/fig_perfold.pdf")
    plt.close(fig)


def per_class() -> None:
    f = pd.read_csv("results/final_folds.csv")
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    xs = np.arange(len(CLASS_SEQ))
    width = 0.2
    shades = ["0.15", "0.40", "0.62", "0.82"]
    for i, n in enumerate(ORDER):
        sub = f[f.configuration == n]
        means = [sub[f"recall_{c}"].mean() for c in CLASS_SEQ]
        errs = [sub[f"recall_{c}"].std(ddof=1) for c in CLASS_SEQ]
        ax.bar(xs + (i - 1.5) * width, means, width, yerr=errs, label=n,
               color=shades[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_SEQ, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")
    fig.savefig(f"{FIG_DIR}/fig_perclass_recall.pdf")
    plt.close(fig)


def confusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
    for ax, name, fn in zip(axes, ["k-NN", "tree"],
                            ["results/final_confusion_k-NN.csv",
                             "results/final_confusion_tree.csv"]):
        cm = pd.read_csv(fn, index_col=0).loc[CLASS_SEQ, CLASS_SEQ].to_numpy().astype(float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_SEQ, rotation=90, fontsize=7.5)
        ax.set_yticklabels(CLASS_SEQ, fontsize=7.5)
        ax.set_xlabel("predicted class", fontsize=9.5)
        ax.set_ylabel("actual class", fontsize=9.5)
        ax.set_title(name, fontsize=10)
        for i in range(10):
            for j in range(10):
                if norm[i, j] >= 0.02:
                    ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0"), ha="center",
                            va="center", fontsize=5.5,
                            color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="proportion of actual class")
    fig.savefig(f"{FIG_DIR}/fig_confusion.pdf")
    plt.close(fig)


def final_comparison() -> None:
    f = pd.read_csv("results/final_folds.csv")
    metrics = ["f1_macro", "balanced_accuracy", "f1_weighted", "mcc"]
    labels = ["macro\n$F_1$", "balanced\naccuracy", "weighted\n$F_1$",
              "Matthews\ncoefficient"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    xs = np.arange(len(metrics)); width = 0.2
    shades = ["0.15", "0.40", "0.62", "0.82"]
    for i, n in enumerate(ORDER):
        sub = f[f.configuration == n]
        means = [sub[m].mean() for m in metrics]
        errs = [sub[m].std(ddof=1) for m in metrics]
        ax.bar(xs + (i - 1.5) * width, means, width, yerr=errs, label=n,
               color=shades[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.savefig(f"{FIG_DIR}/fig_final_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    class_distribution(); print("class distribution")
    feature_ranges();     print("feature ranges")
    tuning_knn();         print("k-NN tuning")
    tuning_tree();        print("tree tuning")
    per_fold();           print("per fold")
    per_class();          print("per class recall")
    confusion();          print("confusion matrices")
    final_comparison();   print("final comparison")
    correlation_heatmap();print("correlation heatmap")
