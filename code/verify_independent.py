"""Independent check of every dataset claim in the report.

Shares no code with the pipeline, on purpose. Every quantity is recomputed from
the raw CSV with plain pandas and compared against what the report says. A
disagreement means one of the two is wrong.
"""
from __future__ import annotations

import re
import subprocess
import os
import numpy as np
import pandas as pd

RAW = os.environ.get("A2_DATA", "networkTraffic.csv")
PDF = "report/26243881RW741assignment2.pdf"
NOMINAL = ["proto", "state", "service"]

results: list[tuple[str, str, str, bool]] = []


def check(name: str, claimed, computed, tol: float = 0.0) -> None:
    if isinstance(claimed, (int, float)) and isinstance(computed, (int, float)):
        ok = abs(float(claimed) - float(computed)) <= tol
    else:
        ok = str(claimed) == str(computed)
    results.append((name, str(claimed), str(computed), ok))


def main() -> None:
    # --- load raw
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, low_memory=False)
    n_rows, n_cols = raw.shape

    check("instances", 257673, n_rows)
    check("total columns", 44, n_cols)
    check("descriptive features", 43, n_cols - 1)
    check("nominal features", 3, len(NOMINAL))
    check("numeric features (incl. id)", 40, n_cols - 1 - len(NOMINAL))

    # --- missingness
    miss = {c: int((raw[c] == "?").sum()) for c in raw.columns}
    with_missing = {c: v for c, v in miss.items() if v > 0}
    check("features with missing values", 1, len(with_missing))
    check("missing feature name", "service", list(with_missing)[0])
    check("service missing count", 141321, with_missing["service"])
    check("service missing percent", 54.85,
          round(100 * with_missing["service"] / n_rows, 2), tol=0.01)
    check("target missing values", 0, int((raw["attack_cat"] == "?").sum()))

    # --- class skew
    y = raw["attack_cat"].astype(int)
    vc = y.value_counts()
    check("number of classes", 10, int(y.nunique()))
    check("largest class count", 93000, int(vc.max()))
    check("smallest class count", 174, int(vc.min()))
    check("imbalance ratio (rounded)", 534, round(vc.max() / vc.min()))

    # --- numeric ops
    num_cols = [c for c in raw.columns if c not in NOMINAL + ["attack_cat"]]
    num = raw[num_cols].apply(pd.to_numeric)
    nid = num.drop(columns=["id"])

    rng = nid.max() - nid.min()
    check("widest range feature", "sload", rng.idxmax())
    check("widest range value", 5988000256.0, float(rng.max()), tol=1.0)
    nonzero = rng[rng > 0]
    check("narrowest non-zero range feature", "is_sm_ips_ports", nonzero.idxmin())
    check("narrowest non-zero range value", 1.0, float(nonzero.min()))
    check("range ratio mantissa", 5.99,
          round(float(rng.max() / nonzero.min()) / 1e9, 2), tol=0.005)

    sk = nid.skew()
    check("most skewed feature", "trans_depth", sk.idxmax())
    check("maximum skewness", 173.2, round(float(sk.max()), 1), tol=0.05)

    # outliers by the 1.5 interquartile range rule
    pct_out = {}
    for c in nid.columns:
        s = nid[c]
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        if iqr > 0:
            pct_out[c] = 100 * (((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) / len(s)
    po = pd.Series(pct_out)
    check("features with >10% outliers", 16, int((po >= 10).sum()))
    check("worst outlier feature", "dload", po.idxmax())
    check("worst outlier percent", 22.3, round(float(po.max()), 1), tol=0.05)

    # --- correlations
    C = nid.corr().abs()
    iu = np.triu(np.ones(C.shape, dtype=bool), k=1)
    vals = C.to_numpy()[iu]
    check("pairs with |r| >= 0.90", 18, int((vals >= 0.90).sum()))
    check("pairs with |r| >= 0.95", 12, int((vals >= 0.95).sum()))
    check("strongest correlation", 0.9989, round(float(vals.max()), 4), tol=0.0001)
    i, j = np.unravel_index(np.argmax(C.to_numpy() * iu), C.shape)
    pair = sorted([C.columns[i], C.columns[j]])
    check("strongest pair", "['ct_ftp_cmd', 'is_ftp_login']", str(pair))

    # --- duplicates
    dup_all = int(raw.drop(columns=["id"]).duplicated().sum())
    dup_x = int(raw.drop(columns=["id", "attack_cat"]).duplicated().sum())
    check("duplicate rows ignoring id", 94928, dup_all)
    check("duplicate percent", 36.84, round(100 * dup_all / n_rows, 2), tol=0.01)
    check("conflicting label rows", 9061, dup_x - dup_all)

    # --- nominal features
    check("proto cardinality", 133, int(raw["proto"].nunique()))
    check("state cardinality", 11, int(raw["state"].nunique()))
    check("service cardinality", 13, int(raw["service"].nunique()))
    check("id unique per row", True, bool(raw["id"].nunique() == n_rows))
    check("id monotonically increasing", True,
          bool(raw["id"].astype(int).is_monotonic_increasing))

    # --- claim: adjacent records share a class more
    same = float((y.values[1:] == y.values[:-1]).mean())
    chance = float((vc / n_rows).pow(2).sum())
    check("adjacent-class agreement exceeds chance", True, bool(same > chance * 2))

    # --- claim about sbytes and median
    check("sbytes maximum over median", 27189,
          round(float(nid["sbytes"].max() / nid["sbytes"].median())), tol=1)

    # --- report
    print(f"{'CLAIM':<44} {'REPORT':>22} {'RECOMPUTED':>22}   OK")
    print("-" * 96)
    bad = 0
    for name, claimed, computed, ok in results:
        if not ok:
            bad += 1
        print(f"{name:<44} {claimed:>22} {computed:>22}   {'yes' if ok else 'NO'}")
    print("-" * 96)
    print(f"{len(results) - bad}/{len(results)} verified against the raw data")
    if bad:
        print("\nMISMATCHES REQUIRE INVESTIGATION")


if __name__ == "__main__":
    main()
