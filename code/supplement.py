"""The leftover secondary metrics.

Appends each fold to disk as it finishes, because the first attempt died on the
last fold and lost the lot.

The tree is cheap, so it runs over all ten folds. The k-NN configurations run on
one fold, which is enough for the timing comparison and for the training
estimate. Pooled macro precision and recall come off the confusion matrices step
10 already wrote.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score)

from common import (RANDOM_SEED, OUT_DIR, ensure_dirs, load_data,
                       numeric_columns, log)
from evaluate import TUNE_FRACTION, TRAIN_PROBE
from final_eval import knn_probs, tree_probs

LOGP = f"{OUT_DIR}/supplement_log.txt"
OUTP = f"{OUT_DIR}/v2_supplement.csv"


def append_row(row: dict) -> None:
    header = not os.path.exists(OUTP)
    pd.DataFrame([row]).to_csv(OUTP, mode="a", header=header, index=False)


def main():
    ensure_dirs()
    if os.path.exists(OUTP):
        os.remove(OUTP)
    tb = json.load(open(f"{OUT_DIR}/retune_f1_best_tree.json"))["tree"]
    arr = np.load(f"{OUT_DIR}/allconfig_multipliers.npy")
    names = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
    mult = {n: arr[i] for i, n in enumerate(names)}

    X, y = load_data()
    _, X_eval, _, y_eval = train_test_split(X, y, train_size=TUNE_FRACTION,
                                            stratify=y, random_state=RANDOM_SEED)
    X_eval = X_eval.reset_index(drop=True)
    y_eval = y_eval.reset_index(drop=True)
    num = numeric_columns(X_eval)

    makers = {
        "tree":         (lambda a, b, c: tree_probs(a, b, c, num, tb, smote=False), 10),
        "tree + SMOTE": (lambda a, b, c: tree_probs(a, b, c, num, tb, smote=True), 10),
        "k-NN":         (lambda a, b, c: knn_probs(a, b, c, num, smote=False), 1),
        "k-NN + SMOTE": (lambda a, b, c: knn_probs(a, b, c, num, smote=True), 1),
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(X_eval, y_eval))

    for name, (mk, nfolds) in makers.items():
        for fold, (tr, te) in enumerate(splits[:nfolds], 1):
            t0 = time.perf_counter()
            P = mk(X_eval.iloc[tr], y_eval.iloc[tr], X_eval.iloc[te])
            pred_s = time.perf_counter() - t0
            yv = y_eval.iloc[te].to_numpy()
            pred = np.argmax(P * mult[name], axis=1)

            probe, _ = train_test_split(np.arange(len(tr)), train_size=TRAIN_PROBE,
                                        stratify=y_eval.iloc[tr],
                                        random_state=RANDOM_SEED)
            Ptr = mk(X_eval.iloc[tr], y_eval.iloc[tr], X_eval.iloc[tr].iloc[probe])
            ytr_probe = y_eval.iloc[tr].iloc[probe].to_numpy()
            tpred = np.argmax(Ptr * mult[name], axis=1)

            row = dict(
                configuration=name, fold=fold,
                precision_macro=precision_score(yv, pred, average="macro",
                                                zero_division=0),
                recall_macro=recall_score(yv, pred, average="macro",
                                          zero_division=0),
                train_accuracy=accuracy_score(ytr_probe, tpred),
                train_balanced_accuracy=balanced_accuracy_score(ytr_probe, tpred),
                predict_seconds=pred_s)
            append_row(row)
            log(f"  {name:14s} fold {fold:2d}: prec={row['precision_macro']:.4f} "
                f"trainBal={row['train_balanced_accuracy']:.4f} "
                f"predict={pred_s:.1f}s", LOGP)

    # pooled macro precision and recall, no further prediction required
    pooled = []
    for n in names:
        f = f"{OUT_DIR}/v2_confusion_{n.replace(' ','_').replace('+','plus')}.csv"
        M = pd.read_csv(f, index_col=0).to_numpy().astype(float)
        tp = np.diag(M)
        p = np.divide(tp, M.sum(0), out=np.zeros(10), where=M.sum(0) > 0).mean()
        r = np.divide(tp, M.sum(1), out=np.zeros(10), where=M.sum(1) > 0).mean()
        pooled.append(dict(configuration=n, precision_macro_pooled=float(p),
                           recall_macro_pooled=float(r)))
        log(f"  pooled {n:14s} precision={p:.4f} recall={r:.4f}", LOGP)
    pd.DataFrame(pooled).to_csv(f"{OUT_DIR}/v2_pooled_pr.csv", index=False)
    log("supplement complete", LOGP)
    print("DONE")


if __name__ == "__main__":
    main()
