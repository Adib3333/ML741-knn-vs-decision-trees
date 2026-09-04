# ML741 Assignment 2: Nearest Neighbours and Decision Trees

N.H. Chowdhury, 26243881. Stellenbosch University.

A comparison of k-nearest neighbours against a classification tree on a modified
UNSW-NB15 network traffic dataset: 257 673 records, 43 descriptive features, ten
classes at a 534:1 imbalance.

The point of the study is that the two algorithms rest on opposing assumptions,
so a single preprocessing scheme cannot serve both. k-NN needs scaling and
suffers from redundancy and dimensionality; the tree is invariant to scale, far
less sensitive to the other two, and suffers from class imbalance instead. Each
pipeline is derived from those assumptions and then checked by ablation.

## Contents

- `code/` - the whole pipeline, listed below in run order
- `results/` - every result file the report draws on

The report itself is not in this repository. It is uploaded in STEMlearn.

## Dataset

`networkTraffic.csv` is supplied with the assignment on STEMlearn and is not
included here. Download it and put it in `code/` before running anything. The
scripts read the path from the `A2_DATA` environment variable, set in the block
below.

## Reproducing

```powershell
$env:A2_DATA = "code\networkTraffic.csv"

python code\audit.py             # look at the data
python code\tune.py              # find good settings
python code\ablation.py          # check the preprocessing
python code\scaling.py           # scaling and redundancy studies
python code\retune_f1.py         # re-tune for macro F1
python code\multipliers_tree.py  # decision multipliers, tree
python code\multipliers_knn.py   # decision multipliers, k-NN
python code\final_eval.py        # final evaluation
python code\supplement.py        # remaining metrics
python code\ceiling_export.py    # attainability analysis
python code\rebuild_inputs.py    # reshape results
python code\extra_results.py     # per-fold and sensitivity data
python code\tables.py            # tables and numbers
python code\tables_extra.py      # audit-correction macros
python code\regen_figures.py     # charts
```

Checks:

```powershell
python code\verify_independent.py   # dataset claims, recomputed from the raw file
python code\verify_revision.py      # report numbers against the result files
```

Seed 42 throughout. The full pipeline takes a few hours.

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
classes (Backdoor, Analysis, DoS) are largely inseparable from Exploits.
Memorising the whole dataset reaches a macro F1 of about 0.78, so a score near
0.70 on this data is a sign of a leak rather than a better model.

## What each file in `code/` does

### Shared

**`common.py`**
Loads the dataset and holds the two preprocessing steps, one for k-NN and one
for the tree. Everything else imports from here, so changing something here
changes the whole study at once. The random seed of 42 lives here too, which is
what makes the results repeatable.

### Looking at the data

**`audit.py`**
Measures what is wrong with the raw data before any modelling: how lopsided the
classes are, how many values are missing, how skewed the numbers are, how many
rows are duplicates, and which features are near-copies of each other. The data
quality section of the report comes from here.

### Finding the best settings

**`tune.py`**
Grid search. For k-NN that is the number of neighbours, the vote weighting and
the distance measure. For the tree it is depth, leaf size, split rule and
pruning. Runs coarse first, then finer around the winner.

**`tune_resume.py`**
Restart helper. The search takes hours, so if it dies partway this picks up from
the finished results instead of starting over.

**`retune_knn.py`**
Runs the k-NN search again after the scaling method changed. Different scaling
means a different shape of space, which means a different best k. Re-searching
rather than assuming is the point.

### Checking the preprocessing was worth it

**`ablation.py`**
Switches off one preprocessing step at a time and measures the damage. If
turning a step off changes nothing, that step was not earning its place. This is
where the evidence comes from that scaling is critical for k-NN and pointless
for the tree.

**`scaling.py`**
Two studies. First compares five ways of squashing numbers onto a common range,
which is how the rank transform beat min-max. Second tests whether dropping
near-duplicate features helps. It did not.

### Evaluating

**`evaluate.py`**
First full evaluation. All four setups through stratified 10-fold
cross-validation, every metric recorded per fold. Also holds the capped SMOTE
code and the Holm correction used later.

**`retune_f1.py`**
Redoes the search with macro F1 as the target instead of balanced accuracy.
Introduces the tunable class weighting, so the push towards rare classes becomes
a dial rather than an on/off switch.

**`multipliers_tree.py`**
Fits the decision multipliers for the tree. Instead of predicting whichever
class has the highest probability, each class is scaled first by a number tuned
for macro F1. Fitted on tuning data only, then applied unchanged.

**`multipliers_knn.py`**
Same for k-NN, and re-checks the neighbourhood size at the same time because the
two interact. Runs the neighbour search once at the largest k and reuses it for
the smaller ones, so three settings cost one search.

**`final_eval.py`**
Final evaluation. All four setups under the new decision rule, ten folds, plus
the Friedman and Wilcoxon tests. These are the numbers in the report.

**`supplement.py`**
Fills in what `final_eval.py` did not record: precision, training-set scores and
timings. Writes each fold to disk as it finishes, so a crash near the end does
not lose everything.

### Turning results into the report

**`rebuild_inputs.py`**
Reshapes the result files into the format the table generator expects, so new
results flow through to the report without anything being retyped.

**`tables.py`**
Turns the result files into every LaTeX table and every number the report
quotes. Nothing in the report is typed by hand; each figure is a macro generated
here. That is what stops a number in the text drifting away from the data.

**`tables_revision.py`**
The extra tables and numbers added in the revision: the weighting dial, the
neighbourhood comparison, the multipliers and the ceiling analysis.

**`extra_results.py`**
Three more result files the final version of the report needed: per-fold
results, the effect of training partition size, and the correlations the brief
calls perfect.

**`tables_extra.py`**
The handful of macros added after the final audit, all read from result files
rather than typed in.

**`regen_figures.py`**
Draws every chart at font sizes large enough to satisfy the writing rules.
Writes into `report/figures`, which is where the LaTeX source includes them
from, so a regenerated chart reaches the report without a manual copy.

### Checking the work

**`ceiling_export.py`**
Works out how well anything could possibly do on this dataset, by imagining a
model that memorises every row. Also searches for a reweighting that beats the
plain majority rule on macro F1, since the majority rule is only optimal for
accuracy.

**`verify_independent.py`**
Re-checks 38 facts about the dataset using separate code that shares nothing
with the pipeline. If the two disagree, one of them has a bug. All 38 agreed.

**`verify_revision.py`**
Checks that every number printed in the report matches the result files and that
the derived numbers add up.

## Licence

MIT
