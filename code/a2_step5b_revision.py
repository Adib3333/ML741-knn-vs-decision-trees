"""Tables and macros new to the macro F1 revision.

Called from a2_step5_tables.main, so everything the report quotes still comes out
of code rather than out of prose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a2_common import OUT_DIR, CLASS_NAMES, CLASS_ORDER

ORDER = ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]
SHORT = {"k-NN": "Neighbour", "k-NN + SMOTE": "Neighbour, os.",
         "tree": "Tree", "tree + SMOTE": "Tree, os."}
# classes in descending frequency, matching the other per-class tables
SEQ = [CLASS_NAMES[c] for c in [0, 9, 4, 6, 3, 1, 5, 2, 8, 7]]


def add_revision_tables(macro, write, fmt, thin, word, MACROS) -> None:

    # --- class weighting exponent
    sw = pd.read_csv(f"{OUT_DIR}/rev_alpha_sweep.csv")
    rows = []
    for r in sw.itertuples():
        lab = {0.0: "$0$ (no weighting)", 1.0: "$1$ (inverse frequency)"}.get(
            r.alpha, f"${r.alpha:g}$")
        rows.append(f"{lab} & {fmt(r.f1_mean)} & {fmt(r.bal_mean)} \\\\")
    write("tab_alpha.tex", f"""\\begin{{tabular}}{{lcc}}
\\toprule
Exponent $\\alpha$ & Macro $F_1$ & Balanced accuracy \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")

    best = sw.loc[sw.f1_mean.idxmax()]
    macro("alphaBest", f"{best.alpha:g}")
    macro("alphaBestFone", fmt(best.f1_mean))
    macro("alphaBestBal", fmt(best.bal_mean))
    one = sw[sw.alpha == 1.0].iloc[0]
    zero = sw[sw.alpha == 0.0].iloc[0]
    macro("alphaOneFone", fmt(one.f1_mean))
    macro("alphaOneBal", fmt(one.bal_mean))
    macro("alphaZeroFone", fmt(zero.f1_mean))
    macro("alphaGainOverOne", fmt(best.f1_mean - one.f1_mean))
    macro("alphaBalCost", fmt(one.bal_mean - best.bal_mean))

    # --- neighbourhood size under the tuned rule
    kk = pd.read_csv(f"{OUT_DIR}/rev_knn_k.csv")
    rows = [f"${int(r.k)}$ & {fmt(r.plain)} & {fmt(r.tuned)} & {fmt(r.tuned - r.plain)} \\\\"
            for r in kk.itertuples()]
    write("tab_knnk.tex", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Neighbourhood $k$ & Arg max & Tuned rule & Gain \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")
    bp = kk.loc[kk.plain.idxmax()]
    bt = kk.loc[kk.tuned.idxmax()]
    macro("knnKArgmax", str(int(bp.k)))
    macro("knnKArgmaxWord", word(int(bp.k)))
    macro("knnKTuned", str(int(bt.k)))
    macro("knnKTunedFone", fmt(bt.tuned))
    macro("knnKArgmaxFone", fmt(bp.plain))
    macro("knnKTunedGain", fmt(bt.tuned - bt.plain))
    macro("knnKArgmaxGain", fmt(bp.tuned - bp.plain))

    # --- multipliers table
    mp = pd.read_csv(f"{OUT_DIR}/rev_multipliers.csv", index_col=0)
    rows = []
    for n in SEQ:
        cells = " & ".join(f"{mp.loc[c, n]:.2f}" for c in ORDER)
        rows.append(f"{n} & {cells} \\\\")
    write("tab_multipliers.tex", f"""\\begin{{tabular}}{{lcccc}}
\\toprule
Class & {' & '.join(SHORT[c] for c in ORDER)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")
    macro("multWormsKnn", f"{mp.loc['k-NN', 'Worms']:.2f}")
    macro("multShellCt", f"{mp.loc['tree', 'Shellcode']:.2f}")
    macro("multNormalCt", f"{mp.loc['tree', 'Normal']:.2f}")
    macro("multExploitsCt", f"{mp.loc['tree', 'Exploits']:.2f}")

    # --- attainability bounds
    cs = pd.read_csv(f"{OUT_DIR}/ceiling_summary.csv").iloc[0]
    cp = pd.read_csv(f"{OUT_DIR}/ceiling_perclass.csv").set_index("class")
    s = pd.read_csv(f"{OUT_DIR}/final_summary.csv", index_col=0)
    ours = float(s.loc["tree", "f1_macro_mean"])

    rows = []
    for n in SEQ:
        rows.append(f"{n} & {fmt(cp.loc[n, 'ceiling_memorisation_cv'])} & "
                    f"{fmt(cp.loc[n, 'ceiling_gifted_cv'])} & "
                    f"{fmt(cp.loc[n, 'ceiling_resubstitution'])} \\\\")
    write("tab_ceiling.tex", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Class & Memorisation & Conceded & Resubstitution \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
Macro mean & {fmt(cs.macro_f1_memorisation_cv)} & {fmt(cs.macro_f1_gifted_cv)} & {fmt(cs.macro_f1_resubstitution)} \\\\
\\bottomrule
\\end{{tabular}}""")

    macro("ceilResub", fmt(cs.macro_f1_resubstitution))
    macro("ceilMemorise", fmt(cs.macro_f1_memorisation_cv))
    macro("ceilGifted", fmt(cs.macro_f1_gifted_cv))
    macro("ceilMatched", fmt(100 * cs.matched_fraction, 1))
    macro("ceilUnmatched", fmt(100 * (1 - cs.matched_fraction), 1))
    macro("nDistinctVectors", thin(int(cs.distinct_feature_vectors)))
    macro("ceilProgress", fmt(100 * (ours - cs.macro_f1_memorisation_cv) /
                              (cs.macro_f1_gifted_cv - cs.macro_f1_memorisation_cv), 0))
    macro("ceilOverMemorise", fmt(100 * (ours - cs.macro_f1_memorisation_cv) /
                                  cs.macro_f1_memorisation_cv, 0))
    for n in ["Backdoor", "Analysis", "DoS", "Shellcode", "Worms", "Generic"]:
        macro(f"ceil{n}", fmt(cp.loc[n, "ceiling_resubstitution"]))
    three = sum(cp.loc[n, "ceiling_resubstitution"] for n in ["Backdoor", "Analysis", "DoS"])
    macro("ceilThreeSum", fmt(three / 10))
    macro("ceilSevenNeeded", fmt((0.70 * 10 - three) / 7))
    seven = [n for n in SEQ if n not in ("Backdoor", "Analysis", "DoS")]
    macro("ceilSevenMean", fmt(np.mean([cp.loc[n, "ceiling_resubstitution"] for n in seven])))

    # the normalisation study was scored on balanced accuracy; under macro F1
    # the top two transformations are not separated by the evidence
    sc = pd.read_csv(f"{OUT_DIR}/scaling_study.csv")
    rank = sc[sc.transformation == "rank based"].iloc[0]
    logm = sc[sc.transformation == "logarithmic, then min-max"].iloc[0]
    macro("scaleRankFone", fmt(rank.f1_mean))
    macro("scaleLogFone", fmt(logm.f1_mean))
    macro("scaleFoneGap", fmt(abs(logm.f1_mean - rank.f1_mean), 4))
    macro("scaleRankBalSd", fmt(rank.bal_std, 4))
    macro("scaleRankFoneSd", fmt(rank.f1_std, 4))
    macro("ctWeightGainFone", fmt(
        float(sw.f1_mean.max()) - float(sw[sw.alpha == 0.0].iloc[0].f1_mean)))

    dup = pd.read_csv(f"{OUT_DIR}/rev_duplicates.csv").iloc[0]
    macro("nDupConflict", thin(int(dup.conflicting_label_rows)))


    # --- per-fold results, WR spec 5
    pf = pd.read_csv(f"{OUT_DIR}/v2_folds.csv")
    rows = []
    for f in sorted(pf.fold.unique()):
        r = pf[pf.fold == f].set_index("configuration")
        cells = " & ".join(f"{r.loc[c, 'f1_macro']:.4f}" for c in ORDER)
        cells2 = " & ".join(f"{r.loc[c, 'balanced_accuracy']:.4f}" for c in ORDER)
        rows.append(f"{int(f)} & {cells} & {cells2} \\\\")
    g = pf.groupby("configuration")
    mean_row = " & ".join(f"{g['f1_macro'].mean()[c]:.4f}" for c in ORDER) + " & " + \
               " & ".join(f"{g['balanced_accuracy'].mean()[c]:.4f}" for c in ORDER)
    sd_row = " & ".join(f"{g['f1_macro'].std(ddof=1)[c]:.4f}" for c in ORDER) + " & " + \
             " & ".join(f"{g['balanced_accuracy'].std(ddof=1)[c]:.4f}" for c in ORDER)
    hdr = " & ".join(SHORT[c] for c in ORDER)
    write("tab_perfold.tex", f"""\\begin{{tabular}}{{r cccc cccc}}
\\toprule
& \\multicolumn{{4}}{{c}}{{Macro $F_1$}} & \\multicolumn{{4}}{{c}}{{Balanced accuracy}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
Fold & {hdr} & {hdr} \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
Mean & {mean_row} \\\\
Standard deviation & {sd_row} \\\\
\\bottomrule
\\end{{tabular}}""")

    # how many folds each configuration wins, for the stability claim
    win = pf.loc[pf.groupby("fold")["f1_macro"].idxmax(), "configuration"]
    macro("nFoldsTreeWins", word(int((win == "tree").sum())))
    macro("nFoldsTotal", word(int(pf.fold.nunique())))
    sd = g["f1_macro"].std(ddof=1)
    macro("sdSpreadMin", fmt(sd.min(), 4))
    macro("sdSpreadMax", fmt(sd.max(), 4))
    gapmin = pf.pivot(index="fold", columns="configuration", values="f1_macro")
    margin = (gapmin["tree"] - gapmin["k-NN"])
    macro("foldMarginMin", fmt(margin.min(), 4))
    macro("foldMarginMax", fmt(margin.max(), 4))

    # --- training partition size, spec question 3
    ts = pd.read_csv(f"{OUT_DIR}/rev_trainsize.csv")
    tss = pd.read_csv(f"{OUT_DIR}/rev_trainsize_summary.csv").iloc[0]
    rows = [f"{r.configuration} & {r.tuning_f1:.4f} & {r.evaluation_f1:.4f} & "
            f"{r.gain:+.4f} \\\\" for r in ts.itertuples()]
    write("tab_trainsize.tex", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Configuration & Smaller partition & Larger partition & Change \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")
    macro("trainSmall", thin(int(tss.train_records_tuning)))
    macro("trainLarge", thin(int(tss.train_records_evaluation)))
    macro("trainRatio", fmt(tss.ratio, 1))
    macro("trainMeanGain", fmt(tss.mean_gain, 4))
    macro("trainRankingSame", "unchanged" if bool(tss.ranking_identical) else "changed")

    # --- correlations the specification calls perfect
    scr = pd.read_csv(f"{OUT_DIR}/rev_speccorr.csv")
    scs = pd.read_csv(f"{OUT_DIR}/rev_speccorr_summary.csv").iloc[0]
    macro("specCorrMax", f"{scs.max_correlation:.6f}")
    macro("specCorrExact", word(int(scs.pairs_exactly_one)))
    for r in scr.itertuples():
        macro(f"corr{r.feature_a.replace('_','').capitalize()}",
              f"{r.correlation:.6f}")


    # gap in balanced accuracy before and after oversampling, for the sensitivity
    fs = pd.read_csv(f"{OUT_DIR}/final_summary.csv", index_col=0)["balanced_accuracy_mean"]
    macro("gapBalPlain", fmt(fs["tree"] - fs["k-NN"], 4))
    macro("gapBalSmote", fmt(fs["tree"] - fs["k-NN + SMOTE"], 4))

    # --- scale of the control parameter search
    import glob, os as _os
    def _n(f):
        p = f"{OUT_DIR}/{f}"
        return len(pd.read_csv(p)) if _os.path.exists(p) else 0
    knn_cp = _n("tune_knn_stage1.csv") + _n("tune_knn_stage2.csv") + \
             _n("tune_knn_final.csv") + _n("retune_knn_f1.csv")
    ct_cp  = _n("tune_ct_stage1.csv") + _n("tune_ct_stage2.csv") + \
             _n("retune_ct_f1_stageA.csv") + _n("retune_ct_f1_stageB.csv")
    knn_pp = _n("scaling_study.csv") + _n("redundancy_study.csv") + _n("ablation_knn.csv")
    ct_pp  = _n("ablation_ct.csv")
    rule   = _n("rev_knn_k.csv")
    macro("nCfgKnnControl", str(knn_cp))
    macro("nCfgCtControl", str(ct_cp))
    macro("nCfgKnn", str(knn_cp + knn_pp + rule))
    macro("nCfgCt", str(ct_cp + ct_pp))
    macro("nCfgTotal", str(knn_cp + ct_cp + knn_pp + ct_pp + rule))
    macro("nCfgPreproc", str(knn_pp + ct_pp))
    k1 = pd.read_csv(f"{OUT_DIR}/tune_knn_stage1.csv")
    macro("nKnnGridCells", str(len(k1)))
    macro("nKnnGridComplete",
          "complete" if len(k1) == k1.k.nunique() * k1.weights.nunique() * k1.metric.nunique()
          else "partial")

    # --- the eight statements the specification makes about the supplied file
    dq = [
        ("Missing values marked \\texttt{?}",
         "Harmful", "Neutral",
         "Distance is undefined for a missing value; a tree sends the level down one branch"),
        ("Many features with correlations",
         "Harmful", "Neutral",
         "A repeated construct is summed twice in the distance; a tree tests one member only"),
        ("Features with outliers",
         "Harmful", "Neutral",
         "An outlier fixes the anchors of min-max scaling; a split point depends on rank, so no treatment is required"),
        ("Numeric ranges differing significantly",
         "Severe", "Neutral",
         "The widest feature dominates the sum; threshold tests are scale invariant"),
        ("Numerical and categorical features",
         "Harmful", "Mild",
         "Categorical values have no numeric distance; a tree needs only an arbitrary code"),
        ("\\texttt{proto}, \\texttt{state}, \\texttt{service} nominal",
         "Harmful", "Mild",
         "One-hot encoding of 133 levels adds dimensions; a tree isolates levels by splitting"),
        ("\\texttt{id} unique per observation",
         "Harmful", "Severe",
         "A unique value adds noise to every distance; a deep tree memorises capture order"),
        ("Skew class distribution",
         "Severe", "Severe",
         "A rare class rarely holds a neighbourhood majority; a rare leaf is pruned away"),
    ]
    rows = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in dq]
    write("tab_dqexpect.tex", f"""\\begin{{tabular}}{{p{{0.24\\textwidth}} cc p{{0.44\\textwidth}}}}
\\toprule
Data quality issue & k-NN & Tree & Reason \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")
    macro("nDqIssues", word(len(dq)))

    # --- the precision that motivated the metric change
    ow = pd.read_csv(f"{OUT_DIR}/rev_old_worms.csv").iloc[0]
    macro("wormsCtOld", fmt(ow.worms_recall_old))
    macro("wormsCtPrecOld", fmt(ow.worms_precision_old))
