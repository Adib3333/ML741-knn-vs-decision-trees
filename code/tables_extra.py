"""Macros added by the audit corrections.

Every value is read from a result file. Appends to report/numbers.tex, replacing
any macro of the same name already there.
"""
from __future__ import annotations

import re
import pandas as pd

from common import OUT_DIR

NUMBERS = "report/numbers.tex"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def fmt(v, d=4):
    return f"{v:.{d}f}"


def thin(n):
    """Thousands separator, matching the thin space used in the prose."""
    return f"{int(n):,}".replace(",", "\\,")


def main() -> None:
    ceil = pd.read_csv(f"{OUT_DIR}/ceiling_summary.csv").iloc[0]
    knn = pd.read_csv(f"{OUT_DIR}/ablation_knn.csv").set_index("variant")["bal_mean"]
    ct = pd.read_csv(f"{OUT_DIR}/ablation_ct.csv").set_index("variant")
    rule = pd.read_csv(f"{OUT_DIR}/rev_knn_k.csv")
    dup = pd.read_csv(f"{OUT_DIR}/rev_duplicates.csv").iloc[0]
    red = pd.read_csv(f"{OUT_DIR}/redundancy_study.csv").set_index("setting")["bal_mean"]

    new = {
        # attainability, corrected
        "ceilResubAcc": fmt(ceil["accuracy_resubstitution"]),
        "ceilResubBest": fmt(ceil["macro_f1_resubstitution_reweighted"]),
        "ceilResubBestAcc": fmt(ceil["accuracy_resubstitution_reweighted"]),
        # the size of the loss, as distinct from the resulting value
        "ablKnnCorrDrop": fmt(knn["selected pipeline"]
                              - knn["correlation filter added"]),
        # the variation against which the invariance residual is judged
        "ablCtBaseSd": fmt(ct.loc["selected pipeline", "bal_std"]),
        # configurations spent on the joint re-examination of the rule
        "nCfgRuleWord": WORDS[len(rule)],
        # records the majority rule cannot place, as distinct from the count
        # of extra (vector, label) pairs the first audit reported
        "nConflictRecords": thin(dup["conflict_group_records"]),
        "nUnresolvable": thin(dup["unresolvable_records"]),
        "pctUnresolvable": f"{dup['unresolvable_percent']:.2f}",
        # what each correlation threshold cost, against full retention
        "redNineNine": fmt(red["every feature retained"]
                           - red["correlation filter at 0.99"]),
        "redNineFive": fmt(red["every feature retained"]
                           - red["correlation filter at 0.95"]),
        # one-hot against ordinal encoding in the tree pipeline
        "ablCtOnehotDelta": fmt(abs(ct.loc["selected pipeline", "bal_mean"]
                                    - ct.loc["one-hot instead of ordinal",
                                             "bal_mean"])),
    }

    text = open(NUMBERS).read()
    for key, val in new.items():
        line = "\\newcommand{\\%s}{%s}" % (key, val)
        pat = re.compile(r"\\newcommand\{\\%s\}\{[^}]*\}" % key)
        # lambda, not a template string: the replacement contains backslashes
        text = pat.sub(lambda _: line, text) if pat.search(text) \
            else text + line + "\n"
    open(NUMBERS, "w").write(text)

    for key, val in new.items():
        print(f"  {key:20s} {val}")


if __name__ == "__main__":
    main()
