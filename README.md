# ML741 Assignment 2: Nearest Neighbours and Decision Trees

N.H. Chowdhury, 26243881. Stellenbosch University.

A comparison of k-nearest neighbours against a classification tree on a modified
UNSW-NB15 network traffic dataset: 257 673 records, 43 descriptive features, ten
classes at a 534:1 imbalance.

The point of the study is that the two algorithms rest on opposing assumptions,
so a single preprocessing scheme cannot serve both. k-NN needs scaling and
suffers from redundancy and dimensionality; the tree is invariant to all three
and suffers from class imbalance instead. Each pipeline is derived from those
assumptions and then checked by ablation.

## Contents

- `code/` — the whole pipeline, numbered in run order
- `results/` — every result file the report draws on
- `ML741_A2_walkthrough.ipynb` — the study end to end, section by section, with
  the saved results and figures already in place
- `CODE_GUIDE.md` — plain-English summary of what each script does

The report itself is not in this repository.

## Dataset

`networkTraffic.csv` is supplied with the assignment and is not redistributed
here. Put it wherever you like and point `A2_DATA` at it.

## Reproducing

```powershell
$env:A2_DATA = "networkTraffic.csv"

python code\a2_step1_audit.py            # dataset audit
python code\a2_step2_tune.py             # two-stage grid search
python code\a2_step3_ablation.py         # preprocessing ablation
python code\a2_step3b_scaling.py         # normalisation and redundancy studies
python code\a2_step6_retune_f1.py        # control parameters under macro F1
python code\a2_step8_thresholds.py       # decision multipliers, tree
python code\a2_step9_knn_thresholds.py   # decision multipliers, k-NN
python code\a2_step10_all_configs.py     # four-configuration evaluation
python code\a2_step11b_supplement.py     # secondary metrics
python code\ceiling_export.py            # attainability analysis
python code\a2_step12_rebuild_tables.py
python code\a2_step13_audit_additions.py
python code\a2_step5_tables.py           # tables and numeric macros
python code\a2_step5c_audit_macros.py
python code\regen_figures.py             # figures
```

Checks:

```powershell
python code\verify_independent.py        # dataset claims, recomputed from the raw file
python code\verify_revision.py           # report numbers against the result files
python code\check_writing_rules.py       # automated writing-rule check
```

Seed 42 throughout. The full pipeline takes a few hours; the notebook runs on a
sample by default and finishes in a couple of minutes.

## Headline results

Ten-fold cross-validation on a 75 per cent evaluation partition that no tuning
decision touched.

| | macro F1 | balanced accuracy | accuracy |
|---|---|---|---|
| classification tree | 0.5893 | 0.6516 | 0.7830 |
| k-NN | 0.5436 | 0.5480 | 0.7949 |

The tree wins on the class-averaged metrics and k-NN wins on the record-averaged
ones, which traces to a crossover in per-class recall rather than to one model
being better everywhere. Every pairwise difference is significant under Wilcoxon
with Holm correction, and the ordering holds in all ten folds.

The absolute numbers look low because the dataset caps them. 22 191 records carry
a label other than the most frequent label of their own feature vector, and three
classes
(Backdoor, Analysis, DoS) are largely inseparable from Exploits. Memorising the
whole dataset reaches a macro F1 of about 0.78, so a score near 0.70 on this data
is a sign of a leak rather than a better model.

## Licence

MIT, see `LICENSE`.
