"""Check the revised report against the result files.

Every macro the generator emits is checked to appear in the rendered PDF text,
and every derived arithmetic claim is recomputed independently of the generator.
"""
from __future__ import annotations

import os
import re
import subprocess
import numpy as np
import pandas as pd

PDF = os.environ.get("A2_PDF", "report/26243881RW741assignment2.pdf")
OUT = "results"

txt = subprocess.run(["pdftotext", "-layout", PDF, "-"],
                     capture_output=True, text=True).stdout
flat = " ".join(txt.split())

fails, checks = [], 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{name}: {detail}")


# --- macros in PDF
macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}",
                         open("report/numbers.tex").read()))
numeric = {k: v for k, v in macros.items()
           if re.fullmatch(r"[-+]?[\d.,\\ ]+", v) and any(c.isdigit() for c in v)}
missing = []
for k, v in numeric.items():
    probe = v.replace("\\,", "\u2009").replace("\\,", " ")
    variants = {probe, probe.replace("\u2009", " "), probe.replace("\u2009", ""),
                probe.replace("\u2009", ",")}
    if not any(w in flat for w in variants):
        missing.append(f"{k}={v}")
check("every numeric macro appears in the PDF", not missing,
      f"{len(missing)} absent: {missing[:6]}")

# --- headline consistency
s = pd.read_csv(f"{OUT}/final_summary.csv", index_col=0)
folds = pd.read_csv(f"{OUT}/v2_folds.csv")

for cfg in ["k-NN", "k-NN + SMOTE", "tree", "tree + SMOTE"]:
    sub = folds[folds.configuration == cfg]
    for m in ["balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "kappa",
              "accuracy"]:
        check(f"{cfg} {m} summary equals fold mean",
              abs(sub[m].mean() - s.loc[cfg, f"{m}_mean"]) < 1e-9)

# macro recall must equal balanced accuracy by definition
for cfg in s.index:
    check(f"{cfg}: macro recall equals balanced accuracy",
          abs(s.loc[cfg, "recall_macro_mean"] - s.loc[cfg, "balanced_accuracy_mean"]) < 5e-4,
          f"{s.loc[cfg,'recall_macro_mean']:.5f} vs {s.loc[cfg,'balanced_accuracy_mean']:.5f}")

# --- derived quantities
f = lambda x: float(x)
check("ctOverKnn equals tree minus k-NN macro F1",
      abs(f(macros["ctOverKnn"]) -
          (s.loc["tree", "f1_macro_mean"] - s.loc["k-NN", "f1_macro_mean"])) < 5e-4)
check("ctOverKnnPct consistent",
      abs(f(macros["ctOverKnnPct"]) - 100 * (s.loc["tree", "f1_macro_mean"] -
          s.loc["k-NN", "f1_macro_mean"]) / s.loc["k-NN", "f1_macro_mean"]) < 0.1)
check("knnOverCtAcc equals k-NN minus tree accuracy",
      abs(f(macros["knnOverCtAcc"]) -
          (s.loc["k-NN", "accuracy_mean"] - s.loc["tree", "accuracy_mean"])) < 5e-4)

# --- ceiling bounds
cs = pd.read_csv(f"{OUT}/ceiling_summary.csv").iloc[0]
cp = pd.read_csv(f"{OUT}/ceiling_perclass.csv").set_index("class")
check("resubstitution bound exceeds the conceded bound",
      cs.macro_f1_resubstitution > cs.macro_f1_gifted_cv)
check("conceded bound exceeds memorisation",
      cs.macro_f1_gifted_cv > cs.macro_f1_memorisation_cv)
check("our tree lies between memorisation and the conceded bound",
      cs.macro_f1_memorisation_cv < s.loc["tree", "f1_macro_mean"] < cs.macro_f1_gifted_cv)
check("resubstitution bound equals the mean of its per-class values",
      abs(cs.macro_f1_resubstitution - cp.ceiling_resubstitution.mean()) < 1e-6)
prog = 100 * (s.loc["tree", "f1_macro_mean"] - cs.macro_f1_memorisation_cv) / \
       (cs.macro_f1_gifted_cv - cs.macro_f1_memorisation_cv)
check("ceilProgress consistent", abs(f(macros["ceilProgress"]) - prog) < 1.0,
      f"macro {macros['ceilProgress']} vs {prog:.1f}")

three = sum(cp.loc[n, "ceiling_resubstitution"] for n in ["Backdoor", "Analysis", "DoS"])
check("ceilThreeSum consistent", abs(f(macros["ceilThreeSum"]) - three / 10) < 5e-4)
check("ceilSevenNeeded consistent",
      abs(f(macros["ceilSevenNeeded"]) - (7.0 - three) / 7) < 5e-4)

# --- statistics coherence
pw = pd.read_csv(f"{OUT}/stats_pairwise.csv")
fr = pd.read_csv(f"{OUT}/stats_friedman.csv").iloc[0]
check("Friedman statistic is the maximum for four groups over ten folds",
      abs(fr.statistic - 30.0) < 1e-9, f"{fr.statistic}")
check("all six comparisons significant after Holm", bool(pw.significant.all()))
for r in pw.itertuples():
    a, b = r.comparison.split(" vs ")
    d = folds[folds.configuration == a].sort_values("fold").f1_macro.mean() - \
        folds[folds.configuration == b].sort_values("fold").f1_macro.mean()
    check(f"{r.comparison}: difference matches folds", abs(d - r.mean_difference) < 1e-9)

# --- lead/tie claims
p = pd.read_csv(f"{OUT}/final_perclass.csv", index_col=0)
cls = pd.read_csv(f"{OUT}/audit_classes.csv").set_index("class")
diff = {n: p.loc["k-NN", f"recall_{n}_mean"] - p.loc["tree", f"recall_{n}_mean"]
        for n in cls.index}
knn = [n for n in cls.index if diff[n] > 0.005]
ct = [n for n in cls.index if diff[n] < -0.005]
tie = [n for n in cls.index if abs(diff[n]) <= 0.005]
tot = cls["count"].sum()
words = {1: "one", 2: "two", 3: "three", 7: "seven", 8: "eight"}
check("nKnnLead matches", macros["nKnnLead"] == words[len(knn)])
check("nCtLead matches", macros["nCtLead"] == words[len(ct)])
check("nTied matches", macros["nTied"] == words[len(tie)])
check("shares sum to 100",
      abs(f(macros["shareKnnLead"]) + f(macros["shareCtLead"]) +
          f(macros["shareTied"]) - 100.0) < 0.15,
      f"{macros['shareKnnLead']}+{macros['shareCtLead']}+{macros['shareTied']}")
check("shareKnnLead matches counts",
      abs(f(macros["shareKnnLead"]) - 100 * cls.loc[knn, "count"].sum() / tot) < 0.05)

# --- report
print(f"{checks - len(fails)}/{checks} checks passed")
for x in fails:
    print("  FAIL:", x)
if not fails:
    print("no discrepancies")
