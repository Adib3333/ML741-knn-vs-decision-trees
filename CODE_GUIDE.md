# What each file in `code/` does

Plain-English guide, in roughly the order the files were run.

## Shared

**`a2_common.py`**
Loads the dataset and holds the two preprocessing recipes, one for k-NN and one
for the tree. Everything else imports from here, so changing something here
changes the whole study at once. The random seed of 42 lives here too, which is
what makes the results repeatable.

## Looking at the data

**`a2_step1_audit.py`**
Measures what is wrong with the raw data before any modelling: how lopsided the
classes are, how many values are missing, how skewed the numbers are, how many
rows are duplicates, and which features are near-copies of each other. The data
quality section of the report comes from here.

## Finding the best settings

**`a2_step2_tune.py`**
Grid search. For k-NN that is the number of neighbours, the vote weighting and
the distance measure. For the tree it is depth, leaf size, split rule and
pruning. Runs coarse first, then finer around the winner.

**`a2_step2b_resume.py`**
Restart helper. The search takes hours, so if it dies partway this picks up from
the finished results instead of starting over.

**`a2_step2c_retune_knn.py`**
Runs the k-NN search again after the scaling method changed. Different scaling
means a different shape of space, which means a different best k. Re-searching
rather than assuming is the point.

## Checking the preprocessing was worth it

**`a2_step3_ablation.py`**
Switches off one preprocessing step at a time and measures the damage. If
turning a step off changes nothing, that step was not earning its place. This is
where the evidence comes from that scaling is critical for k-NN and pointless
for the tree.

**`a2_step3b_scaling.py`**
Two studies. First compares five ways of squashing numbers onto a common range,
which is how the rank transform beat min-max. Second tests whether dropping
near-duplicate features helps. It did not.

## Evaluating

**`a2_step4_evaluate.py`**
First full evaluation. All four setups through stratified 10-fold
cross-validation, every metric recorded per fold. Also holds the capped SMOTE
code and the Holm correction used later.

**`a2_step6_retune_f1.py`**
Redoes the search with macro F1 as the target instead of balanced accuracy.
Introduces the tunable class weighting, so the push towards rare classes becomes
a dial rather than an on/off switch.

**`a2_step8_thresholds.py`**
Fits the decision multipliers for the tree. Instead of predicting whichever
class has the highest probability, each class is scaled first by a number tuned
for macro F1. Fitted on tuning data only, then applied unchanged.

**`a2_step9_knn_thresholds.py`**
Same for k-NN, and re-checks the neighbourhood size at the same time because the
two interact. Runs the neighbour search once at the largest k and reuses it for
the smaller ones, so three settings cost one search.

**`a2_step10_all_configs.py`**
Final evaluation. All four setups under the new decision rule, ten folds, plus
the Friedman and Wilcoxon tests. These are the numbers in the report.

**`a2_step11b_supplement.py`**
Fills in what step 10 did not record: precision, training-set scores and
timings. Writes each fold to disk as it finishes, so a crash near the end does
not lose everything.

## Turning results into the report

**`a2_step12_rebuild_tables.py`**
Reshapes the result files into the format the table generator expects, so new
results flow through to the report without anything being retyped.

**`a2_step5_tables.py`**
Turns the result files into every LaTeX table and every number the report
quotes. Nothing in the report is typed by hand; each figure is a macro generated
here. That is what stops a number in the text drifting away from the data.

**`a2_step5b_revision.py`**
The extra tables and numbers added in the revision: the weighting dial, the
neighbourhood comparison, the multipliers and the ceiling analysis.

**`a2_step13_audit_additions.py`**
Three more result files the final version of the report needed: per-fold
results, the effect of training partition size, and the correlations the brief
calls perfect.

**`a2_step5c_audit_macros.py`**
The handful of macros added after the final audit, all read from result files
rather than typed in.

**`regen_figures.py`**
Draws every chart at font sizes large enough to satisfy the writing rules.
Writes into `report/figures`, which is where the LaTeX source includes them
from, so a regenerated chart reaches the report without a manual copy.

## Checking the work

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

**`check_writing_rules.py`**
Checks the report against all 34 writing rules: sentence length, tense,
pronouns, digits, captions, references and the rest.

## Order to run them

```powershell
$env:A2_DATA = "networkTraffic.csv"

python code\a2_step1_audit.py            # look at the data
python code\a2_step2_tune.py             # find good settings
python code\a2_step3_ablation.py         # check the preprocessing
python code\a2_step3b_scaling.py         # scaling and redundancy studies
python code\a2_step6_retune_f1.py        # re-tune for macro F1
python code\a2_step8_thresholds.py       # multipliers, tree
python code\a2_step9_knn_thresholds.py   # multipliers, k-NN
python code\a2_step10_all_configs.py     # final evaluation
python code\a2_step11b_supplement.py     # remaining metrics
python code\ceiling_export.py            # attainability analysis
python code\a2_step12_rebuild_tables.py  # reshape results
python code\a2_step13_audit_additions.py # per-fold and sensitivity data
python code\a2_step5_tables.py           # tables and numbers
python code\a2_step5c_audit_macros.py    # audit-correction macros
python code\regen_figures.py             # charts

python code\verify_independent.py        # checks
python code\verify_revision.py
python code\check_writing_rules.py
```

Note: `a2_step12_rebuild_tables.py` rewrites `tune_best.json` without the two
score fields `a2_step5_tables.py` reads, so on a clean re-run from scratch step 5
needs those fields put back first.
