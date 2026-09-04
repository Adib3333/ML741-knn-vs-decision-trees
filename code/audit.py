"""Data quality audit.

Produces the evidence behind the data quality section of the report, plus three
figures.

  results/audit_summary.csv        per-feature summary
  results/audit_correlations.csv   pairs with |r| >= 0.90
  results/audit_classes.csv        class distribution
  results/audit_facts.txt          the headline numbers
  figures/fig_class_distribution.pdf, fig_feature_ranges.pdf,
  figures/fig_correlation_heatmap.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (DATA_PATH, OUT_DIR, FIG_DIR, TARGET, ID_COL, NOMINAL,
                       MISSING_TOKEN, CLASS_NAMES, CLASS_ORDER, ensure_dirs, log)

# figure text at 10pt to match the 10pt IEEE body text (writing rule 25b)
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "serif",
})

FACTS: dict[str, object] = {}


def main() -> None:
    ensure_dirs()
    logpath = f"{OUT_DIR}/audit_facts.txt"
    open(logpath, "w").close()

    raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False, low_memory=False)
    n, p = raw.shape
    FACTS["n_instances"] = n
    FACTS["n_columns"] = p
    FACTS["n_descriptive"] = p - 1
    log(f"instances={n}  columns={p}  descriptive={p-1}", logpath)

    # --- missing
    miss = {c: int((raw[c] == MISSING_TOKEN).sum()) for c in raw.columns}
    miss = {c: v for c, v in miss.items() if v > 0}
    FACTS["missing"] = miss
    for c, v in miss.items():
        log(f"missing '{MISSING_TOKEN}' in {c}: {v} ({100*v/n:.2f}%)", logpath)
    FACTS["n_features_with_missing"] = len(miss)

    # --- target
    y = raw[TARGET].astype(int)
    vc = y.value_counts().reindex(CLASS_ORDER)
    classes = pd.DataFrame({
        "code": CLASS_ORDER,
        "class": [CLASS_NAMES[c] for c in CLASS_ORDER],
        "count": vc.values,
        "percent": (100 * vc.values / n).round(3),
    })
    classes.to_csv(f"{OUT_DIR}/audit_classes.csv", index=False)
    imb = int(vc.max()) / int(vc.min())
    FACTS["imbalance_ratio"] = round(imb, 1)
    FACTS["majority"] = (CLASS_NAMES[int(vc.idxmax())], int(vc.max()))
    FACTS["minority"] = (CLASS_NAMES[int(vc.idxmin())], int(vc.min()))
    FACTS["n_target_missing"] = int((raw[TARGET] == MISSING_TOKEN).sum())
    log(f"classes=10  imbalance={imb:.1f}:1  "
        f"majority={FACTS['majority']}  minority={FACTS['minority']}", logpath)
    log(f"missing target values: {FACTS['n_target_missing']}", logpath)

    # --- numeric
    X = raw.drop(columns=[TARGET])
    numeric_cols = [c for c in X.columns if c not in NOMINAL]
    num = X[numeric_cols].apply(pd.to_numeric)

    rows = []
    for c in numeric_cols:
        s = num[c]
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        pct_out = (100 * (((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) / n
                   if iqr > 0 else np.nan)
        rows.append(dict(
            feature=c, kind="numeric", cardinality=int(s.nunique()),
            minimum=float(s.min()), maximum=float(s.max()),
            rng=float(s.max() - s.min()), mean=float(s.mean()),
            std=float(s.std()), median=float(s.median()),
            skewness=float(s.skew()), pct_outlier=pct_out,
            pct_zero=float(100 * (s == 0).mean()),
        ))
    for c in NOMINAL:
        s = X[c]
        rows.append(dict(
            feature=c, kind="nominal", cardinality=int(s.nunique()),
            minimum=np.nan, maximum=np.nan, rng=np.nan, mean=np.nan,
            std=np.nan, median=np.nan, skewness=np.nan, pct_outlier=np.nan,
            pct_zero=np.nan,
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/audit_summary.csv", index=False)

    # scale spread, excluding the identifier which is dropped for both models
    rng = summary.set_index("feature")["rng"].drop(labels=[ID_COL], errors="ignore").dropna()
    rmax, rmin = rng.max(), rng[rng > 0].min()
    FACTS["range_max"] = (rng.idxmax(), float(rmax))
    FACTS["range_min"] = (rng[rng > 0].idxmin(), float(rmin))
    FACTS["range_ratio"] = float(rmax / rmin)
    log(f"largest range: {FACTS['range_max']}  smallest non-zero: "
        f"{FACTS['range_min']}  ratio={FACTS['range_ratio']:.3e}", logpath)

    # outliers and skewness
    FACTS["max_skew"] = (summary.loc[summary.skewness.idxmax(), "feature"],
                         float(summary.skewness.max()))
    hi_out = summary[summary.pct_outlier >= 10].sort_values("pct_outlier", ascending=False)
    FACTS["n_features_gt10pct_outliers"] = int(len(hi_out))
    FACTS["max_outlier_feature"] = (hi_out.iloc[0]["feature"],
                                    float(hi_out.iloc[0]["pct_outlier"])) if len(hi_out) else None
    log(f"max skewness: {FACTS['max_skew']}", logpath)
    log(f"features with >10% outliers by 1.5 IQR: {FACTS['n_features_gt10pct_outliers']}; "
        f"worst {FACTS['max_outlier_feature']}", logpath)

    # --- nominal
    for c in NOMINAL:
        FACTS[f"cardinality_{c}"] = int(X[c].nunique())
        log(f"cardinality {c}: {X[c].nunique()}", logpath)
    FACTS["id_unique"] = bool(raw[ID_COL].nunique() == n)
    log(f"id unique for every instance: {FACTS['id_unique']}", logpath)

    # --- correlation
    corr_src = num.drop(columns=[ID_COL], errors="ignore")
    C = corr_src.corr().abs()
    cols = list(C.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iloc[i, j]
            if np.isfinite(v) and v >= 0.90:
                pairs.append(dict(feature_a=cols[i], feature_b=cols[j],
                                  abs_r=round(float(v), 4)))
    cp = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    cp.to_csv(f"{OUT_DIR}/audit_correlations.csv", index=False)
    FACTS["n_pairs_090"] = int(len(cp))
    FACTS["n_pairs_095"] = int((cp.abs_r >= 0.95).sum())
    FACTS["top_pairs"] = cp.head(5).values.tolist()
    log(f"numeric pairs with |r|>=0.90: {len(cp)};  >=0.95: {FACTS['n_pairs_095']}", logpath)
    for r in cp.head(6).itertuples():
        log(f"   {r.feature_a} - {r.feature_b}: |r|={r.abs_r}", logpath)

    # --- duplicates
    dup_all = int(raw.drop(columns=[ID_COL]).duplicated().sum())
    dup_x = int(raw.drop(columns=[ID_COL, TARGET]).duplicated().sum())
    FACTS["dup_rows"] = dup_all
    FACTS["dup_rows_pct"] = round(100 * dup_all / n, 2)
    FACTS["dup_descriptive"] = dup_x

    # dup_x - dup_all counts extra (vector, label) pairs, not records, so it
    # answers a different question from the one the report asks. Count the
    # records directly: those sitting in a feature vector that carries more
    # than one label, and those whose own label is not the most frequent label
    # of that vector. The second count is what no deterministic rule can fix.
    key = raw.drop(columns=[ID_COL, TARGET]).agg("\x1f".join, axis=1)
    grp = pd.DataFrame({"k": pd.factorize(key)[0], "y": raw[TARGET]})
    mixed = grp.groupby("k")["y"].nunique()
    FACTS["conflict_group_records"] = int(grp["k"].isin(
        mixed[mixed > 1].index).sum())
    maj = grp.groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
    FACTS["unresolvable"] = int((grp["k"].map(maj) != grp["y"]).sum())
    FACTS["unresolvable_pct"] = round(100 * FACTS["unresolvable"] / n, 2)
    log(f"duplicate rows ignoring id: {dup_all} ({100*dup_all/n:.2f}%)", logpath)
    log(f"records in mixed-label vectors: {FACTS['conflict_group_records']}; "
        f"records the majority rule cannot place: {FACTS['unresolvable']} "
        f"({FACTS['unresolvable_pct']}%)", logpath)

    # --- irregular cardinality
    for c in ["is_ftp_login", "ct_ftp_cmd"]:
        vals = sorted(num[c].unique().tolist())
        bad = int((~num[c].isin([0, 1])).sum())
        FACTS[f"{c}_values"] = vals
        FACTS[f"{c}_nonbinary"] = bad
        log(f"{c} declared binary, observed values {vals}, "
            f"non-binary rows {bad}", logpath)

    # --- figures
    figure_class_distribution(classes)
    figure_feature_ranges(summary)
    figure_correlation_heatmap(corr_src)

    with open(logpath, "a") as f:
        f.write("\n--- FACTS ---\n")
        for k, v in FACTS.items():
            f.write(f"{k}: {v}\n")
    log("audit complete", logpath)


def figure_class_distribution(classes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    order = classes.sort_values("count", ascending=False)
    ax.bar(range(len(order)), order["count"], color="0.35", edgecolor="black",
           linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["class"], rotation=55, ha="right")
    ax.set_ylabel("number of instances")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(order["count"]):
        ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=6.5, rotation=90)
    ax.set_ylim(top=order["count"].max() * 12)
    fig.savefig(f"{FIG_DIR}/fig_class_distribution.pdf")
    plt.close(fig)


def figure_feature_ranges(summary: pd.DataFrame) -> None:
    s = summary[(summary.kind == "numeric") & (summary.feature != ID_COL)]
    s = s.sort_values("rng", ascending=True)
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    ax.barh(range(len(s)), s["rng"].clip(lower=1e-3), color="0.35",
            edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["feature"], fontsize=6)
    ax.set_xlabel("range of observed values")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    fig.savefig(f"{FIG_DIR}/fig_feature_ranges.pdf")
    plt.close(fig)


def figure_correlation_heatmap(num: pd.DataFrame) -> None:
    C = num.corr().abs()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(C.to_numpy(), cmap="Greys", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(C)))
    ax.set_yticks(range(len(C)))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=4)
    ax.set_yticklabels(C.columns, fontsize=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("absolute correlation", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(f"{FIG_DIR}/fig_correlation_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
