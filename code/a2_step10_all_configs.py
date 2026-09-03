"""Step 10: all four configurations under the tuned decision rule.

Steps 8 and 9 fitted multipliers for the bare tree and for bare k-NN. The two
SMOTE configurations need the same rule, otherwise the comparison confounds the
resampling scheme with the decision rule.

Multipliers are fitted per configuration on the tuning partition and applied
unchanged to the evaluation partition. SMOTE stays inside the pipeline, training
fold only.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score, confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline

from a2_common import (RANDOM_SEED, OUT_DIR, CLASS_NAMES, CLASS_ORDER,
                       ensure_dirs, load_data, numeric_columns,
                       build_knn_preprocessor, build_ct_preprocessor, log)
from a2_step4_evaluate import CappedSMOTE, TUNE_FRACTION
from a2_step6_retune_f1 import PowerWeightedTree
from a2_step8_thresholds import macro_f1_from_counts, fit_multipliers

LOGP = f"{OUT_DIR}/allconfig_log.txt"
KNN_K = 31
METRIC = "manhattan"
BATCH = 2048          # test rows per neighbour query, bounds peak memory


def knn_probs(Xtr, ytr, Xte, num, k=KNN_K, smote=False):
    """Class distribution from the k nearest neighbours.

    Queried in batches of BATCH rows. A brute force Manhattan search builds an
    (n_test x n_train) distance block, and after SMOTE the training fold is
    around 243000 rows, so an unbatched query runs out of memory. Batching
    bounds the peak and changes nothing else.
    """
    prep = build_knn_preprocessor(num)
    Ztr = prep.fit_transform(Xtr, ytr)
    ytr_use = np.asarray(ytr)
    if smote:
        sm = CappedSMOTE(random_state=RANDOM_SEED)
        Ztr, ytr_use = sm.fit_resample(Ztr, ytr_use)
    Zte = prep.transform(Xte)
    nn = NearestNeighbors(n_neighbors=k, metric=METRIC, n_jobs=1).fit(Ztr)
    dparts, iparts = [], []
    for s in range(0, len(Zte), BATCH):
        d, i = nn.kneighbors(Zte[s:s + BATCH])
        dparts.append(d)
        iparts.append(i)
    dist = np.vstack(dparts)
    idx = np.vstack(iparts)
    lab = np.asarray(ytr_use)[idx]
    with np.errstate(divide="ignore"):
        w = 1.0 / dist
    exact = ~np.isfinite(w)
    has = exact.any(axis=1)
    w[exact] = 1.0
    w[has] *= exact[has]
    P = np.zeros((len(lab), 10))
    for c in range(10):
        P[:, c] = (w * (lab == c)).sum(axis=1)
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)


def tree_probs(Xtr, ytr, Xte, num, tb, smote=False):
    clf = PowerWeightedTree(alpha=tb["alpha"], criterion=tb["criterion"],
                            max_depth=tb["max_depth"],
                            min_samples_leaf=tb["min_samples_leaf"],
                            ccp_alpha=tb["ccp_alpha"], random_state=RANDOM_SEED)
    steps = [("prep", build_ct_preprocessor(num))]
    if smote:
        steps.append(("smote", CappedSMOTE(random_state=RANDOM_SEED)))
    steps.append(("clf", clf))
    pipe = (ImbPipeline(steps) if smote else Pipeline(steps)).fit(Xtr, ytr)
    Z = pipe.named_steps["prep"].transform(Xte)
    return pipe.named_steps["clf"].model_.predict_proba(Z)


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

    makers = {
        "k-NN":         lambda a, b, c: knn_probs(a, b, c, num, smote=False),
        "k-NN + SMOTE": lambda a, b, c: knn_probs(a, b, c, num, smote=True),
        "tree":         lambda a, b, c: tree_probs(a, b, c, num, tb, smote=False),
        "tree + SMOTE": lambda a, b, c: tree_probs(a, b, c, num, tb, smote=True),
    }

    # --- fit multipliers per config
    cache = f"{OUT_DIR}/allconfig_multipliers.npy"
    if os.path.exists(cache):
        arr = np.load(cache)
        mult = {n: arr[i] for i, n in enumerate(makers)}
        log(f"reusing multipliers cached in {cache}", LOGP)
    else:
        log("fitting multipliers on the tuning partition", LOGP)
        skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        splits5 = list(skf5.split(X_tune, y_tune))
        mult = {}
        for name, mk in makers.items():
            Ps, ys = [], []
            for tr, te in splits5:
                Ps.append(mk(X_tune.iloc[tr], y_tune.iloc[tr], X_tune.iloc[te]))
                ys.append(y_tune.iloc[te].to_numpy())
            plain = float(np.mean([macro_f1_from_counts(yv, np.argmax(P, 1))
                                   for P, yv in zip(Ps, ys)]))
            w, tuned = fit_multipliers(Ps, ys)
            mult[name] = w
            log(f"  {name:14s} plain={plain:.4f} tuned={tuned:.4f} "
                f"({tuned-plain:+.4f})", LOGP)
        np.save(cache, np.array([mult[n] for n in makers]))

    # --- evaluation partition
    log("\nevaluation partition, ten folds, multipliers fixed", LOGP)
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    splits10 = list(skf10.split(X_eval, y_eval))
    rows, cms = [], {}
    for name, mk in makers.items():
        cm_tot = np.zeros((10, 10), dtype=np.int64)
        for fold, (tr, te) in enumerate(splits10, 1):
            P = mk(X_eval.iloc[tr], y_eval.iloc[tr], X_eval.iloc[te])
            yv = y_eval.iloc[te].to_numpy()
            pred = np.argmax(P * mult[name], axis=1)
            cm = confusion_matrix(yv, pred, labels=CLASS_ORDER)
            cm_tot += cm
            rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(10), where=cm.sum(1) > 0)
            r = dict(configuration=name, fold=fold,
                     accuracy=accuracy_score(yv, pred),
                     balanced_accuracy=balanced_accuracy_score(yv, pred),
                     f1_macro=f1_score(yv, pred, average="macro", zero_division=0),
                     f1_weighted=f1_score(yv, pred, average="weighted", zero_division=0),
                     mcc=matthews_corrcoef(yv, pred),
                     kappa=cohen_kappa_score(yv, pred))
            for c in CLASS_ORDER:
                r[f"recall_{CLASS_NAMES[c]}"] = rec[c]
            rows.append(r)
            log(f"  {name:14s} fold {fold:2d}: macroF1={r['f1_macro']:.4f} "
                f"bal={r['balanced_accuracy']:.4f}", LOGP)
        cms[name] = cm_tot
        pd.DataFrame(cm_tot, index=[CLASS_NAMES[c] for c in CLASS_ORDER],
                     columns=[CLASS_NAMES[c] for c in CLASS_ORDER]).to_csv(
            f"{OUT_DIR}/v2_confusion_{name.replace(' ','_').replace('+','plus')}.csv")

    folds = pd.DataFrame(rows)
    folds.to_csv(f"{OUT_DIR}/v2_folds.csv", index=False)
    mets = ["balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "kappa", "accuracy"]
    s = folds.groupby("configuration")[mets].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reindex(list(makers))
    s.to_csv(f"{OUT_DIR}/v2_summary.csv")
    log("\n" + s[[f"{m}_mean" for m in mets]].round(4).to_string(), LOGP)

    # --- statistics
    from scipy import stats
    from a2_step4_evaluate import holm
    names = list(makers)
    cols = [folds[folds.configuration == n].sort_values("fold")["f1_macro"].to_numpy()
            for n in names]
    chi2, p = stats.friedmanchisquare(*cols)
    log(f"\nFriedman on macro F1: chi2={chi2:.4f} p={p:.3e}", LOGP)
    out, praw = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w_, pw = stats.wilcoxon(cols[i], cols[j])
            out.append(dict(comparison=f"{names[i]} vs {names[j]}",
                            mean_a=cols[i].mean(), mean_b=cols[j].mean(),
                            mean_difference=cols[i].mean() - cols[j].mean(),
                            wilcoxon_statistic=w_, wilcoxon_p=pw))
            praw.append(pw)
    for r, a in zip(out, holm(praw)):
        r["holm_p"] = a
        r["significant"] = bool(a < 0.05)
        log(f"  {r['comparison']:32s} diff={r['mean_difference']:+.4f} "
            f"holm_p={r['holm_p']:.4g} {'SIG' if r['significant'] else 'ns'}", LOGP)
    pd.DataFrame(out).to_csv(f"{OUT_DIR}/v2_pairwise.csv", index=False)
    pd.DataFrame([dict(metric="f1_macro", statistic=chi2, p_value=p, alpha=0.05,
                       reject_h0=bool(p < 0.05), n_groups=4, n_folds=10)]).to_csv(
        f"{OUT_DIR}/v2_friedman.csv", index=False)
    log("step 10 complete", LOGP)


if __name__ == "__main__":
    main()
