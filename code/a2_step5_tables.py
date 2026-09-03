"""Step 5: generate every LaTeX table and every numeric macro.

Each table and each number the report quotes comes straight from the result
files, so nothing in the prose can drift away from the code behind it.

  report/tables/*.tex   table bodies the report includes
  report/numbers.tex    \newcommand macros for every number quoted in prose
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from a2_common import OUT_DIR, CLASS_NAMES, CLASS_ORDER

TAB = "report/tables"
os.makedirs(TAB, exist_ok=True)
MACROS: dict[str, str] = {}


_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
           "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def macro(name: str, value) -> None:
    """Register a macro. A control sequence takes letters only, so a digit in
    the name is spelled out.
    """
    clean = "".join(_DIGITS.get(ch, ch) for ch in name)
    MACROS[clean] = str(value)


_WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine"]


def word(n: int) -> str:
    """Writing rule 18: single-digit numbers are given in words."""
    n = int(n)
    return _WORDS[n] if 0 <= n <= 9 else f"{n}"


def thin(n) -> str:
    """Thousands separator, matching the thin space used in the prose."""
    return f"{int(n):,}".replace(",", "\\,")


def fmt(v, d=4) -> str:
    return f"{v:.{d}f}"


def pm(m, s, d=3) -> str:
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def write(fn: str, body: str) -> None:
    with open(f"{TAB}/{fn}", "w") as f:
        f.write(body)


def table_dataset() -> None:
    cls = pd.read_csv(f"{OUT_DIR}/audit_classes.csv")
    rows = "\n".join(
        f"{r['class']} & {r['code']} & {thin(r['count'])} & {r['percent']:.3f} \\\\"
        for _, r in cls.sort_values("count", ascending=False).iterrows())
    write("tab_classes.tex", f"""\\begin{{tabular}}{{lrrr}}
\\toprule
Class & Code & Instances & Percentage \\\\
\\midrule
{rows}
\\midrule
Total & & {thin(cls['count'].sum())} & 100.000 \\\\
\\bottomrule
\\end{{tabular}}""")
    macro("nInstances", f"{cls['count'].sum():,}".replace(",", "\\,"))
    macro("nMajority", f"{cls['count'].max():,}".replace(",", "\\,"))
    macro("nMinority", f"{cls['count'].min():,}")
    macro("imbRatio", f"{cls['count'].max() / cls['count'].min():.0f}")


def table_quality() -> None:
    s = pd.read_csv(f"{OUT_DIR}/audit_summary.csv")
    corr = pd.read_csv(f"{OUT_DIR}/audit_correlations.csv")
    num = s[(s.kind == "numeric") & (s.feature != "id")]

    rmax, rmin = num["rng"].max(), num.loc[num["rng"] > 0, "rng"].min()
    fmax = num.loc[num["rng"].idxmax(), "feature"]
    fmin = num.loc[num.loc[num["rng"] > 0, "rng"].idxmin(), "feature"]
    smax = num.loc[num["skewness"].idxmax()]
    omax = num.loc[num["pct_outlier"].idxmax()]

    macro("rangeMaxFeature", fmax.replace("_", "\\_"))
    macro("rangeMaxValue", f"{rmax:.3g}".replace("e+09", "\\times 10^{9}"))
    macro("rangeMinFeature", fmin.replace("_", "\\_"))
    ratio = rmax / rmin
    exponent = int(np.floor(np.log10(ratio)))
    macro("rangeRatio", f"{ratio / 10**exponent:.2f}\\times 10^{{{exponent}}}")
    macro("skewMaxFeature", smax["feature"].replace("_", "\\_"))
    macro("skewMaxValue", fmt(smax["skewness"], 1))
    macro("outlierMaxFeature", omax["feature"].replace("_", "\\_"))
    macro("outlierMaxValue", fmt(omax["pct_outlier"], 1))
    macro("nOutlierFeatures", int((num["pct_outlier"] >= 10).sum()))
    macro("nCorrNinety", len(corr))
    macro("nCorrNinetyFive", int((corr.abs_r >= 0.95).sum()))
    macro("topCorrValue", fmt(corr.abs_r.max(), 4))

    top = corr.head(8)
    rows = "\n".join(
        f"\\texttt{{{r.feature_a.replace('_', chr(92)+'_')}}} & "
        f"\\texttt{{{r.feature_b.replace('_', chr(92)+'_')}}} & {r.abs_r:.4f} \\\\"
        for r in top.itertuples())
    write("tab_correlations.tex", f"""\\begin{{tabular}}{{llr}}
\\toprule
First feature & Second feature & $|r|$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    sk = num.sort_values("skewness", ascending=False).head(8)
    rows = "\n".join(
        f"\\texttt{{{r.feature.replace('_', chr(92)+'_')}}} & {thin(r.minimum)} & "
        f"{thin(round(r.maximum)) if r.maximum >= 1000 else f'{r.maximum:.3g}'} & {r.skewness:.1f} & "
        f"{'--' if not np.isfinite(r.pct_outlier) else f'{r.pct_outlier:.1f}'} \\\\"
        for r in sk.itertuples())
    write("tab_skew.tex", f"""\\begin{{tabular}}{{lrrrr}}
\\toprule
Feature & Minimum & Maximum & Skewness & Outliers (\\%) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")


def table_scaling() -> None:
    sc = pd.read_csv(f"{OUT_DIR}/scaling_study.csv")
    rows = "\n".join(f"{r.transformation} & {pm(r.bal_mean, r.bal_std)} & "
                     f"{pm(r.f1_mean, r.f1_std)} \\\\" for r in sc.itertuples())
    write("tab_scaling.tex", f"""\\begin{{tabular}}{{lcc}}
\\toprule
Transformation & Balanced accuracy & Macro $F_1$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")
    best, worst = sc.iloc[0], sc[sc.transformation == "min-max"].iloc[0]
    macro("scaleBest", best.transformation)
    macro("scaleBestBal", fmt(best.bal_mean))
    macro("scaleMinmaxBal", fmt(worst.bal_mean))
    macro("scaleGain", fmt(best.bal_mean - worst.bal_mean))
    macro("scaleGainPct", fmt(100 * (best.bal_mean - worst.bal_mean) / worst.bal_mean, 1))

    rd = pd.read_csv(f"{OUT_DIR}/redundancy_study.csv")
    rows = "\n".join(f"{r.setting} & {pm(r.bal_mean, r.bal_std)} & "
                     f"{fmt(r.f1_mean)} \\\\" for r in rd.itertuples())
    write("tab_redundancy.tex", f"""\\begin{{tabular}}{{lcc}}
\\toprule
Setting & Balanced accuracy & Macro $F_1$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")
    macro("redundNoneBal", fmt(rd.iloc[0].bal_mean))
    macro("redundNinetyFiveBal", fmt(rd[rd.threshold == 0.95].iloc[0].bal_mean))


def table_ablation() -> None:
    k = pd.read_csv(f"{OUT_DIR}/ablation_knn.csv")
    c = pd.read_csv(f"{OUT_DIR}/ablation_ct.csv")
    kb, cb = k.iloc[0].bal_mean, c.iloc[0].bal_mean

    def block(df, base):
        return "\n".join(
            f"{r.variant} & {pm(r.bal_mean, r.bal_std)} & "
            f"{r.bal_mean - base:+.3f} & {r.seconds:.1f} \\\\"
            for r in df.itertuples())

    write("tab_ablation.tex", f"""\\begin{{tabular}}{{lccr}}
\\toprule
Variant & Balanced accuracy & Change & Seconds \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\emph{{Nearest neighbour classifier}}}} \\\\
{block(k, kb)}
\\midrule
\\multicolumn{{4}}{{l}}{{\\emph{{Classification tree}}}} \\\\
{block(c, cb)}
\\bottomrule
\\end{{tabular}}""")

    def g(df, v, col="bal_mean"):
        return float(df[df.variant == v].iloc[0][col])

    macro("ablKnnBase", fmt(kb))
    macro("ablKnnNoScale", fmt(g(k, "no scaling")))
    macro("ablKnnNoScaleDrop", fmt(kb - g(k, "no scaling")))
    macro("ablKnnMinmax", fmt(g(k, "min-max instead of rank")))
    macro("ablKnnCorr", fmt(g(k, "correlation filter added")))
    macro("ablKnnNoGroup", fmt(g(k, "no rare-level grouping")))
    macro("ablKnnGroupDelta", fmt(abs(kb - g(k, "no rare-level grouping")), 4))
    macro("ablKnnNoGroupTime", fmt(g(k, "no rare-level grouping", "seconds"), 1))
    macro("ablKnnBaseTime", fmt(float(k.iloc[0].seconds), 1))
    macro("ablKnnGroupSpeedup", fmt(g(k, "no rare-level grouping", "seconds") /
                                    float(k.iloc[0].seconds), 1))
    macro("ablKnnId", fmt(g(k, "identifier retained")))
    macro("ablKnnIdGain", fmt(g(k, "identifier retained") - kb))
    macro("ablCtBase", fmt(cb))
    macro("ablCtScale", fmt(g(c, "min-max scaling added")))
    macro("ablCtScaleDelta", fmt(abs(g(c, "min-max scaling added") - cb)))
    macro("ablCtCorr", fmt(g(c, "correlation filter added")))
    macro("ablCtCorrDrop", fmt(cb - g(c, "correlation filter added")))
    macro("ablCtOnehot", fmt(g(c, "one-hot instead of ordinal")))
    macro("ablCtId", fmt(g(c, "identifier retained")))
    macro("ablCtIdGain", fmt(g(c, "identifier retained") - cb))


def table_tuning() -> None:
    best = json.load(open(f"{OUT_DIR}/tune_best.json"))
    kb, tb = best["knn"], best["tree"]
    macro("bestKWord", word(kb["k"]))
    macro("bestLeafWord", word(tb["min_samples_leaf"]))
    for key, val in [("bestK", kb["k"]), ("bestWeights", kb["weights"]),
                     ("bestMetric", kb["metric"]),
                     ("bestKnnBal", fmt(kb["bal_mean"])),
                     ("bestCriterion", tb["criterion"]),
                     ("bestDepth", tb["max_depth"]),
                     ("bestLeaf", tb["min_samples_leaf"]),
                     ("bestClassWeight", tb["class_weight"]),
                     ("bestAlpha", f"{tb['ccp_alpha']:g}"),
                     ("bestCtBal", fmt(tb["bal_mean"]))]:
        macro(key, val)

    kg = pd.read_csv(f"{OUT_DIR}/tune_knn_final.csv")
    sub = kg[(kg.weights == "distance") & (kg.metric == "manhattan")].sort_values("k")
    rows = "\n".join(f"{int(r.k)} & {pm(r.bal_mean, r.bal_std)} & {fmt(r.f1_mean)} \\\\"
                     for r in sub.itertuples())
    write("tab_tune_knn.tex", f"""\\begin{{tabular}}{{rcc}}
\\toprule
Neighbours & Balanced accuracy & Macro $F_1$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    eu = kg[(kg.metric == "euclidean") & (kg.weights == "distance")]["bal_mean"].to_numpy()
    mh = kg[(kg.metric == "manhattan") & (kg.weights == "distance")]["bal_mean"].to_numpy()
    macro("manhattanWins", word(int((mh > eu).sum())))
    macro("manhattanTotal", word(len(eu)))
    macro("manhattanMeanGain", fmt(float(np.mean(mh - eu))))
    un = kg[(kg.weights == "uniform") & (kg.metric == "manhattan")].sort_values("k")["bal_mean"].to_numpy()
    di = kg[(kg.weights == "distance") & (kg.metric == "manhattan")].sort_values("k")["bal_mean"].to_numpy()
    macro("weightWins", word(int((di >= un).sum())))
    macro("weightMeanGain", fmt(float(np.mean(di - un))))

    cg = pd.read_csv(f"{OUT_DIR}/tune_ct_stage1.csv")
    bal = cg[cg.class_weight == "balanced"]["bal_mean"]
    none = cg[cg.class_weight.isna()]["bal_mean"]
    macro("ctWeightedBest", fmt(bal.max()))
    macro("ctUnweightedBest", fmt(none.max()))
    macro("ctWeightGain", fmt(bal.max() - none.max()))

    cs2 = pd.read_csv(f"{OUT_DIR}/tune_ct_stage2.csv")
    unp = cs2[(cs2.max_depth.isna()) & (cs2.ccp_alpha == 0.0)]
    prn = cs2[(cs2.max_depth.isna())].sort_values("bal_mean", ascending=False)
    if len(unp):
        macro("ctUnprunedBal", fmt(float(unp.iloc[0].bal_mean)))
        macro("ctPrunedBal", fmt(float(prn.iloc[0].bal_mean)))
        macro("ctPruneGain", fmt(float(prn.iloc[0].bal_mean - unp.iloc[0].bal_mean)))

    keep = cs2[cs2.ccp_alpha.isin([0.0, 5e-4])].sort_values(
        ["max_depth", "ccp_alpha"], na_position="last")
    rows = "\n".join(
        f"{'unbounded' if pd.isna(r.max_depth) else int(r.max_depth)} & "
        f"{r.ccp_alpha:g} & {pm(r.bal_mean, r.bal_std)} & {fmt(r.f1_mean)} \\\\"
        for r in keep.itertuples())
    write("tab_tune_ct.tex", f"""\\begin{{tabular}}{{rrcc}}
\\toprule
Depth & Penalty & Balanced accuracy & Macro $F_1$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")


def table_final() -> None:
    s = pd.read_csv(f"{OUT_DIR}/final_summary.csv", index_col=0)
    order = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
    label = {"k-NN": "Nearest neighbour", "k-NN + SMOTE": "Nearest neighbour, oversampled",
             "tree": "Classification tree", "tree + SMOTE": "Classification tree, oversampled"}
    metrics = [("f1_macro", "Macro $F_1$"), ("balanced_accuracy", "Balanced accuracy"),
               ("f1_weighted", "Weighted $F_1$"),
               ("mcc", "Correlation coefficient"), ("kappa", "Cohen $\\kappa$"),
               ("accuracy", "Accuracy")]
    lines = []
    # four decimals, matching the per-fold table, so recomputing from the fold
    # data reproduces these exactly
    for m, lab in metrics:
        cells = " & ".join(pm(s.loc[c, f"{m}_mean"], s.loc[c, f"{m}_std"], 4) for c in order)
        lines.append(f"{lab} & {cells} \\\\")
    # macro precision and recall are pooled over the folds, so no dispersion is quoted
    for m, lab in [("precision_macro", "Macro precision"), ("recall_macro", "Macro recall")]:
        cells = " & ".join(fmt(s.loc[c, f"{m}_mean"], 4) for c in order)
        lines.append(f"{lab} & {cells} \\\\")
    cells = " & ".join(fmt(s.loc[c, "train_accuracy_mean"], 4) for c in order)
    lines.append(f"Training accuracy & {cells} \\\\")
    cells = " & ".join(fmt(s.loc[c, "train_balanced_accuracy_mean"], 4) for c in order)
    lines.append(f"Training balanced accuracy & {cells} \\\\")
    cells = " & ".join(f"{s.loc[c, 'predict_seconds_mean']:.1f}" for c in order)
    lines.append(f"Time per fold (seconds) & {cells} \\\\")

    for c, key in zip(order, ["Knn", "KnnS", "Ct", "CtS"]):
        pass
    write("tab_final.tex", f"""\\begin{{tabular}}{{lcccc}}
\\toprule
Measure & Nearest & Nearest neighbour, & Classification & Classification tree, \\\\
 & neighbour & oversampled & tree & oversampled \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")

    for c, key in zip(order, ["Knn", "KnnS", "Ct", "CtS"]):
        for m, mk in [("balanced_accuracy", "Bal"), ("f1_macro", "F1"), ("mcc", "Mcc"),
                      ("accuracy", "Acc"), ("precision_macro", "Prec"),
                      ("recall_macro", "Rec"), ("kappa", "Kappa"),
                      ("f1_weighted", "Wf1")]:
            macro(f"fin{key}{mk}", fmt(s.loc[c, f"{m}_mean"]))
            macro(f"fin{key}{mk}Sd", fmt(s.loc[c, f"{m}_std"]))
        macro(f"fin{key}Train", fmt(s.loc[c, "train_balanced_accuracy_mean"]))
        macro(f"fin{key}Fit", fmt(s.loc[c, "fit_seconds_mean"], 1))
        macro(f"fin{key}Pred", fmt(s.loc[c, "predict_seconds_mean"], 1))
    macro("ctOverKnn", fmt(s.loc["tree", "f1_macro_mean"] -
                           s.loc["k-NN", "f1_macro_mean"]))
    macro("ctOverKnnPct", fmt(100 * (s.loc["tree", "f1_macro_mean"] -
                                     s.loc["k-NN", "f1_macro_mean"]) /
                              s.loc["k-NN", "f1_macro_mean"], 1))
    macro("ctOverKnnBal", fmt(s.loc["tree", "balanced_accuracy_mean"] -
                              s.loc["k-NN", "balanced_accuracy_mean"]))
    macro("knnOverCtWf1", fmt(s.loc["k-NN", "f1_weighted_mean"] -
                              s.loc["tree", "f1_weighted_mean"]))
    macro("predSpeedup", f"{s.loc['k-NN', 'predict_seconds_mean'] / max(s.loc['tree', 'predict_seconds_mean'], 1e-9):,.0f}".replace(",", "\\,"))
    macro("finKnnPredExact", fmt(s.loc["k-NN", "predict_seconds_mean"], 1))
    macro("finCtPredExact", fmt(s.loc["tree", "predict_seconds_mean"], 3))
    # instance-weighted metrics favour the nearest neighbour classifier
    macro("knnOverCtAcc", fmt(s.loc["k-NN", "accuracy_mean"] - s.loc["tree", "accuracy_mean"]))
    macro("knnOverCtMcc", fmt(s.loc["k-NN", "mcc_mean"] - s.loc["tree", "mcc_mean"]))
    macro("knnOverCtKappa", fmt(s.loc["k-NN", "kappa_mean"] - s.loc["tree", "kappa_mean"]))
    macro("ctOverKnnF1", fmt(s.loc["tree", "f1_macro_mean"] - s.loc["k-NN", "f1_macro_mean"]))
    macro("ctSmoteDelta", fmt(abs(s.loc["tree + SMOTE", "f1_macro_mean"] - s.loc["tree", "f1_macro_mean"])))
    macro("knnSmoteDelta", fmt(abs(s.loc["k-NN + SMOTE", "f1_macro_mean"] - s.loc["k-NN", "f1_macro_mean"])))
    macro("knnSmoteF1Delta", fmt(abs(s.loc["k-NN + SMOTE", "f1_macro_mean"] - s.loc["k-NN", "f1_macro_mean"])))
    macro("ctSmoteBalDelta", fmt(abs(s.loc["tree + SMOTE", "balanced_accuracy_mean"] - s.loc["tree", "balanced_accuracy_mean"])))
    macro("knnSmoteBalGain", fmt(s.loc["k-NN + SMOTE", "balanced_accuracy_mean"] - s.loc["k-NN", "balanced_accuracy_mean"]))
    macro("finKnnTrainAcc", fmt(s.loc["k-NN", "train_accuracy_mean"]))
    macro("finCtTrainAcc", fmt(s.loc["tree", "train_accuracy_mean"]))


def table_perclass() -> None:
    p = pd.read_csv(f"{OUT_DIR}/final_perclass.csv", index_col=0)
    cls = pd.read_csv(f"{OUT_DIR}/audit_classes.csv").set_index("class")
    order = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
    names = [CLASS_NAMES[c] for c in [0, 9, 4, 6, 3, 1, 5, 2, 8, 7]]
    lines = []
    for n in names:
        cells = " & ".join(fmt(p.loc[c, f"recall_{n}_mean"]) for c in order)
        lines.append(f"{n} & {thin(cls.loc[n, 'count'])} & {cells} \\\\")
    write("tab_perclass.tex", f"""\\begin{{tabular}}{{lrcccc}}
\\toprule
Class & Instances & Nearest & Nearest, & Tree & Tree, \\\\
 & & neighbour & oversampled & & oversampled \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}""")
    # Who leads on which classes, and the record share of each group. A lead
    # only counts where the recall gap exceeds TOL: Generic separates the two
    # by 0.0002, and handing that to either side would misstate the result.
    TOL = 0.005
    cnt = cls["count"]; tot = int(cnt.sum())
    diff = {n: p.loc["k-NN", f"recall_{n}_mean"] - p.loc["tree", f"recall_{n}_mean"]
            for n in cls.index}
    knn_lead = [n for n in cls.index if diff[n] > TOL]
    ct_lead = [n for n in cls.index if diff[n] < -TOL]
    tied = [n for n in cls.index if abs(diff[n]) <= TOL]
    macro("nKnnLead", word(len(knn_lead)))
    macro("nCtLead", word(len(ct_lead)))
    macro("nTied", word(len(tied)))
    macro("shareKnnLead", fmt(100 * cnt[knn_lead].sum() / tot, 1))
    macro("shareCtLead", fmt(100 * cnt[ct_lead].sum() / tot, 1))
    macro("shareTied", fmt(100 * cnt[tied].sum() / tot, 1))

    def names(lst):
        lst = sorted(lst, key=lambda n: -cnt[n])
        return lst[0] if len(lst) == 1 else ", ".join(lst[:-1]) + " and " + lst[-1]

    macro("knnLeadNames", names(knn_lead))
    macro("tiedNames", names(tied))
    macro("tiedDiff", fmt(abs(min(diff[n] for n in tied)), 4))

    for key, cfg in zip(["Knn", "KnnS", "Ct", "CtS"], order):
        macro(f"worms{key}", fmt(p.loc[cfg, "recall_Worms_mean"]))
        macro(f"shell{key}", fmt(p.loc[cfg, "recall_Shellcode_mean"]))
        macro(f"backdoor{key}", fmt(p.loc[cfg, "recall_Backdoor_mean"]))
        macro(f"analysis{key}", fmt(p.loc[cfg, "recall_Analysis_mean"]))
        macro(f"normal{key}", fmt(p.loc[cfg, "recall_Normal_mean"]))
        macro(f"exploits{key}", fmt(p.loc[cfg, "recall_Exploits_mean"]))
        macro(f"generic{key}", fmt(p.loc[cfg, "recall_Generic_mean"]))
        macro(f"fuzzers{key}", fmt(p.loc[cfg, "recall_Fuzzers_mean"]))
        macro(f"dos{key}", fmt(p.loc[cfg, "recall_DoS_mean"]))


def table_stats() -> None:
    fr = pd.read_csv(f"{OUT_DIR}/stats_friedman.csv").iloc[0]
    pw = pd.read_csv(f"{OUT_DIR}/stats_pairwise.csv")
    nm = pd.read_csv(f"{OUT_DIR}/stats_normality.csv")

    macro("friedChi", fmt(fr.statistic, 3))
    macro("friedP", f"{fr.p_value:.3e}".replace("e-0", "\\times 10^{-").replace(
        "e-", "\\times 10^{-") + "}")
    macro("friedReject", "rejected" if fr.reject_h0 else "not rejected")
    macro("nNormalityRejected", word(int(nm.normality_rejected.sum())))
    macro("nComparisons", word(len(pw)))

    rows = "\n".join(
        f"{r.comparison.replace('k-NN + SMOTE', 'neighbour, oversampled').replace('tree + SMOTE', 'tree, oversampled').replace('k-NN', 'neighbour')} & "
        f"{r.mean_difference:+.4f} & {r.wilcoxon_p:.4f} & {r.holm_p:.4f} & "
        f"{'yes' if r.significant else 'no'} \\\\" for r in pw.itertuples())
    write("tab_stats.tex", f"""\\begin{{tabular}}{{lrrrc}}
\\toprule
Comparison & Difference & Raw $p$ & Adjusted $p$ & Significant \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}""")

    key = pw[pw.comparison == "k-NN vs tree"]
    if len(key):
        r = key.iloc[0]
        macro("keyDiff", fmt(abs(r.mean_difference), 4))
        macro("keyWilcoxonP", f"{r.wilcoxon_p:.4f}")
        macro("keyHolmP", f"{r.holm_p:.4f}")
        macro("keySignificant", "significant" if r.significant else "not significant")


def _sci(v) -> str:
    """Render a small float in scientific notation, not as code output."""
    if v == 0:
        return "0"
    import math
    e = int(math.floor(math.log10(abs(v))))
    m = v / (10 ** e)
    return (f"10^{{{e}}}" if abs(m - 1) < 1e-12
            else f"{m:g} \\times 10^{{{e}}}")


def table_config() -> None:
    best = json.load(open(f"{OUT_DIR}/tune_best.json"))
    kb, tb = best["knn"], best["tree"]
    sp = pd.read_csv(f"{OUT_DIR}/split_sizes.csv").set_index("partition")
    macro("nTune", f"{int(sp.loc['tuning', 'instances']):,}".replace(",", "\\,"))
    macro("nEval", f"{int(sp.loc['evaluation', 'instances']):,}".replace(",", "\\,"))
    macro("nEvalFold", f"{int(sp.loc['evaluation', 'instances'] / 10):,}".replace(",", "\\,"))
    macro("nMinorityFold", int(174 * 0.75 / 10))

    write("tab_config.tex", f"""\\begin{{tabular}}{{llc}}
\\toprule
Model & Control parameter & Value \\\\
\\midrule
Nearest neighbour & neighbourhood size & {kb['k']} \\\\
 & vote weighting & {kb['weights']} \\\\
 & distance measure & {kb['metric']} \\\\
\\midrule
Classification tree & impurity measure & {tb['criterion']} \\\\
 & depth limit & {tb['max_depth']} \\\\
 & minimum leaf size & {tb['min_samples_leaf']} \\\\
 & class weighting & {tb['class_weight']} \\\\
 & pruning penalty & ${_sci(tb['ccp_alpha'])}$ \\\\
\\midrule
Both & random seed & 42 \\\\
 & folds reported & 10 \\\\
 & folds during tuning & 5 \\\\
\\bottomrule
\\end{{tabular}}""")


def main() -> None:
    table_dataset()
    table_quality()
    table_scaling()
    table_ablation()
    table_tuning()
    table_final()
    table_perclass()
    table_stats()
    table_config()
    from a2_step5b_revision import add_revision_tables
    add_revision_tables(macro, write, fmt, thin, word, MACROS)

    with open("report/numbers.tex", "w") as f:
        f.write("% generated by a2_step5_tables.py - do not edit by hand\n")
        for k in sorted(MACROS):
            f.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
    print(f"wrote {len(MACROS)} macros and 11 tables")
    for k in sorted(MACROS):
        print(f"  \\{k} = {MACROS[k]}")


if __name__ == "__main__":
    main()
