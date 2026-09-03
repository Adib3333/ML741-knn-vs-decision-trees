"""Attainability bounds, exported as CSV so none of them is hand-typed into prose.

Predicting the most common class of each repeated feature vector is optimal for
accuracy, not for macro F1, so the reweighting search at the end establishes how
far above that a deterministic function can actually reach.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RAW = os.environ.get("A2_DATA", "networkTraffic.csv")
NAMES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS", 4: "Exploits",
         5: "Analysis", 6: "Fuzzers", 7: "Worms", 8: "Shellcode", 9: "Generic"}
SEED, FOLDS = 42, 10


def per_class_f1(true, pred, K=10):
    M = np.zeros((K, K), dtype=np.int64)
    np.add.at(M, (true, pred), 1)
    tp = np.diag(M).astype(float)
    p = tp / np.maximum(M.sum(0), 1)
    r = tp / np.maximum(M.sum(1), 1)
    return 2 * p * r / np.maximum(p + r, 1e-12)


def main():
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, low_memory=False)
    y_all = raw["attack_cat"].astype(int).to_numpy()
    feat = [c for c in raw.columns if c not in ("attack_cat", "id")]
    key_all = pd.factorize(raw[feat].agg("\x1f".join, axis=1))[0]

    # --- resubstitution bound, full data
    d = pd.DataFrame({"k": key_all, "y": y_all})
    maj = d.groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
    resub_pred = d["k"].map(maj).to_numpy()
    f1_resub = per_class_f1(y_all, resub_pred)
    n_distinct = int(maj.size)
    acc_resub = float((resub_pred == y_all).mean())

    # The majority rule is optimal for accuracy, not for an average over
    # classes, so its macro F1 is a reference point and not a ceiling. The
    # search below finds how far above it you can actually get.
    counts = np.zeros((n_distinct, 10), dtype=np.int64)
    np.add.at(counts, (key_all, y_all), 1)
    w = np.ones(10)
    grid = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.1, 40.0, 120)])
    best = float(per_class_f1(y_all, counts.argmax(axis=1)[key_all]).mean())
    for _ in range(8):
        improved = False
        for c in range(10):
            keep = w[c]
            for gval in grid:
                w[c] = gval
                s = float(per_class_f1(
                    y_all, (counts * w).argmax(axis=1)[key_all]).mean())
                if s > best + 1e-12:
                    best, keep, improved = s, gval, True
            w[c] = keep
        if not improved:
            break
    alt = (counts * w).argmax(axis=1)[key_all]
    f1_resub_best = best
    acc_resub_best = float((alt == y_all).mean())

    # identifier retained: every row unique
    key_id = raw[[c for c in raw.columns if c != "attack_cat"]].agg("\x1f".join, axis=1)
    id_unique = bool(key_id.nunique() == len(raw))

    # --- cross-validated bounds, evaluation partition
    idx = np.arange(len(y_all))
    _, ev = train_test_split(idx, train_size=0.25, stratify=y_all, random_state=SEED)
    key, y = key_all[ev], y_all[ev]

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    mem, gift, matched = [], [], []
    for tr, te in skf.split(np.zeros(len(y)), y):
        dd = pd.DataFrame({"k": key[tr], "y": y[tr]})
        mj = dd.groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
        fb = int(pd.Series(y[tr]).value_counts().idxmax())
        look = pd.Series(key[te]).map(mj)
        hit = look.notna().to_numpy()
        matched.append(hit.mean())
        pred = np.where(hit, look.fillna(fb).to_numpy(), fb).astype(int)
        mem.append(per_class_f1(y[te], pred))
        pp = pred.copy()
        pp[~hit] = y[te][~hit]
        gift.append(per_class_f1(y[te], pp))
    f1_mem = np.mean(mem, axis=0)
    f1_gift = np.mean(gift, axis=0)

    pd.DataFrame({
        "class": [NAMES[i] for i in range(10)],
        "ceiling_resubstitution": f1_resub,
        "ceiling_memorisation_cv": f1_mem,
        "ceiling_gifted_cv": f1_gift,
    }).to_csv("results/ceiling_perclass.csv", index=False)

    pd.DataFrame([dict(
        distinct_feature_vectors=n_distinct,
        total_rows=int(len(y_all)),
        identifier_makes_every_row_unique=id_unique,
        matched_fraction=float(np.mean(matched)),
        macro_f1_resubstitution=float(f1_resub.mean()),
        accuracy_resubstitution=acc_resub,
        macro_f1_resubstitution_reweighted=f1_resub_best,
        accuracy_resubstitution_reweighted=acc_resub_best,
        macro_f1_memorisation_cv=float(f1_mem.mean()),
        macro_f1_gifted_cv=float(f1_gift.mean()),
    )]).to_csv("results/ceiling_summary.csv", index=False)

    print(pd.read_csv("results/ceiling_summary.csv").T.to_string())


if __name__ == "__main__":
    main()
