"""Rebuild the generator inputs after the macro F1 revision.

The report quotes no number literally. Every figure in the prose is a macro that
tables.py emits from the result files, so rewriting those files here
pushes the revised results through the whole document without anything being
retyped.

It also builds what is new to the revision: the class weighting sweep, the
neighbourhood size under the tuned rule, the fitted multipliers and the
attainability bounds.
"""
from __future__ import annotations

import json
import re
import shutil
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT_DIR, CLASS_NAMES, CLASS_ORDER

ORDER = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
BACKUP = f"{OUT_DIR}/_v1_backup"


def backup_originals() -> None:
    import os
    os.makedirs(BACKUP, exist_ok=True)
    for f in ["final_summary.csv", "final_perclass.csv", "stats_friedman.csv",
              "stats_pairwise.csv", "stats_normality.csv", "final_folds.csv",
              "tune_best.json"]:
        src = f"{OUT_DIR}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{BACKUP}/{f}"):
            shutil.copy(src, f"{BACKUP}/{f}")


def build_summary() -> None:
    """final_summary.csv in the shape the generator expects."""
    folds = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")
    sup = pd.read_csv(f"{OUT_DIR}/v2_supplement.csv")
    pooled = pd.read_csv(f"{OUT_DIR}/v2_pooled_pr.csv").set_index("configuration")

    fold_metrics = ["balanced_accuracy", "f1_macro", "f1_weighted", "mcc",
                    "kappa", "accuracy"]
    s = folds.groupby("configuration")[fold_metrics].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]

    # macro precision and recall pooled over folds, sd from the folds measured
    # individually
    sup_g = sup.groupby("configuration")
    for m, col in [("precision_macro", "precision_macro_pooled"),
                   ("recall_macro", "recall_macro_pooled")]:
        s[f"{m}_mean"] = [pooled.loc[c, col] for c in s.index]
        s[f"{m}_std"] = [sup_g[m].std(ddof=1).get(c, np.nan) for c in s.index]

    for m in ["train_accuracy", "train_balanced_accuracy", "predict_seconds"]:
        s[f"{m}_mean"] = [sup_g[m].mean().get(c, np.nan) for c in s.index]
        s[f"{m}_std"] = [sup_g[m].std(ddof=1).get(c, np.nan) for c in s.index]

    # fit and predict time are no longer separable, since both approaches go
    # through a single call, so total time per fold is reported instead
    s["fit_seconds_mean"] = s["predict_seconds_mean"]
    s["fit_seconds_std"] = s["predict_seconds_std"]
    s.reindex(ORDER).to_csv(f"{OUT_DIR}/final_summary.csv")
    print("final_summary.csv rebuilt")


def build_perclass() -> None:
    folds = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")
    cols = [f"recall_{CLASS_NAMES[c]}" for c in CLASS_ORDER]
    p = folds.groupby("configuration")[cols].agg(["mean", "std"])
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p.reindex(ORDER).to_csv(f"{OUT_DIR}/final_perclass.csv")
    print("final_perclass.csv rebuilt")


def build_stats() -> None:
    shutil.copy(f"{OUT_DIR}/v2_friedman.csv", f"{OUT_DIR}/stats_friedman.csv")
    pw = pd.read_csv(f"{OUT_DIR}/v2_pairwise.csv")
    folds = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")

    # paired t statistics and normality of the paired differences, on macro F1
    cols = {n: folds[folds.configuration == n].sort_values("fold")["f1_macro"].to_numpy()
            for n in ORDER}
    tvals, tps, norm = [], [], []
    for r in pw.itertuples():
        a, b = r.comparison.split(" vs ")
        t, p = stats.ttest_rel(cols[a], cols[b])
        tvals.append(t)
        tps.append(p)
        w, pn = stats.shapiro(cols[a] - cols[b])
        norm.append(dict(comparison=r.comparison, shapiro_statistic=w,
                         shapiro_p=pn, normality_rejected=bool(pn < 0.05)))
    pw["paired_t_statistic"] = tvals
    pw["paired_t_p"] = tps
    pw.to_csv(f"{OUT_DIR}/stats_pairwise.csv", index=False)
    pd.DataFrame(norm).to_csv(f"{OUT_DIR}/stats_normality.csv", index=False)
    print(f"stats rebuilt, normality rejected in "
          f"{sum(n['normality_rejected'] for n in norm)} of {len(norm)} comparisons")


def build_tune_best() -> None:
    tb = json.load(open(f"{OUT_DIR}/retune_f1_best_tree.json"))["tree"]
    best = dict(
        knn=dict(k=31, weights="distance", metric="manhattan"),
        tree=dict(criterion=tb["criterion"], max_depth=tb["max_depth"],
                  min_samples_leaf=tb["min_samples_leaf"],
                  class_weight=f"$\\alpha={tb['alpha']:g}$",
                  ccp_alpha=tb["ccp_alpha"]),
        tune_fraction=0.25, tune_folds=5, seed=42)
    json.dump(best, open(f"{OUT_DIR}/tune_best.json", "w"), indent=2)
    print("tune_best.json rebuilt")


def build_revision_inputs() -> None:
    """Tables and macros new to the revision, for the generator to read."""
    # --- class weighting exponent sweep
    a = pd.read_csv(f"{OUT_DIR}/retune_ct_f1_stageA.csv")
    sweep = a.groupby("alpha")[["f1_mean", "bal_mean"]].max().reset_index()
    sweep.to_csv(f"{OUT_DIR}/rev_alpha_sweep.csv", index=False)

    # --- neighbourhood size under the tuned decision rule, parsed from the log
    txt = open(f"{OUT_DIR}/knn_thresholds_log.txt").read()
    plain = dict(re.findall(r"k=(\d+): plain arg max macro F1 = ([\d.]+)", txt))
    tuned = dict(re.findall(r"k=(\d+): tuned macro F1 = ([\d.]+)", txt))
    rows = [dict(k=int(k), plain=float(plain[k]), tuned=float(tuned[k]))
            for k in sorted(plain, key=int)]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/rev_knn_k.csv", index=False)

    # --- fitted multipliers
    arr = np.load(f"{OUT_DIR}/allconfig_multipliers.npy")
    pd.DataFrame(arr, index=ORDER,
                 columns=[CLASS_NAMES[c] for c in CLASS_ORDER]).to_csv(
        f"{OUT_DIR}/rev_multipliers.csv")

    # --- the configuration whose precision motivated the metric change
    M = pd.read_csv(f"{BACKUP}/final_confusion_tree.csv", index_col=0).to_numpy().astype(float)
    tp = np.diag(M)
    prec = np.divide(tp, M.sum(0), out=np.zeros(10), where=M.sum(0) > 0)
    rec = np.divide(tp, M.sum(1), out=np.zeros(10), where=M.sum(1) > 0)
    i = [CLASS_NAMES[c] for c in CLASS_ORDER].index("Worms")
    pd.DataFrame([dict(worms_recall_old=rec[i], worms_precision_old=prec[i])]).to_csv(
        f"{OUT_DIR}/rev_old_worms.csv", index=False)
    print("revision inputs written")


if __name__ == "__main__":
    backup_originals()
    build_summary()
    build_perclass()
    build_stats()
    build_tune_best()
    build_revision_inputs()
