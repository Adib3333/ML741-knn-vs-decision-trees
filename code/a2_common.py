"""Shared config, data loading and the two preprocessing pipelines.

networkTraffic.csv is 257673 x 44. Target is attack_cat, ten classes, 534:1 skew.

k-NN gets one-hot nominals and rank-scaled numerics. The tree gets ordinal
nominals and no scaling. Everything is a sklearn transformer, so it refits on the
training fold inside cross-validation and nothing leaks.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer)

# --- configuration
RANDOM_SEED = 42

DATA_PATH = os.environ.get("A2_DATA", "networkTraffic.csv")
OUT_DIR = os.environ.get("A2_OUT", "results")
# the report includes from here, so figures are written where LaTeX reads them
FIG_DIR = os.environ.get("A2_FIG", "report/figures")

TARGET = "attack_cat"
ID_COL = "id"
NOMINAL = ["proto", "state", "service"]
MISSING_TOKEN = "?"

CLASS_NAMES = {
    0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Analysis", 6: "Fuzzers", 7: "Worms", 8: "Shellcode", 9: "Generic",
}
CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# correlation threshold above which one member of a pair is dropped for k-NN
CORR_THRESHOLD = 0.95
# minimum relative frequency for a nominal level to survive one-hot encoding
MIN_LEVEL_FREQUENCY = 0.001


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


# --- data loading
def load_data(path: str = DATA_PATH,
              keep_id: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load the data, turn '?' into a missing marker, split off the target.

    id is dropped here rather than in the pipelines. Both models drop it anyway
    and leaving it lying around is asking for a leak. keep_id exists only for
    the ablation that measures what happens when you keep it.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)

    y = df[TARGET].astype(int)
    drop = [TARGET] if keep_id else [TARGET, ID_COL]
    X = df.drop(columns=drop)

    # nominal columns keep the '?' as an explicit level named 'unknown'
    for c in NOMINAL:
        X[c] = X[c].replace(MISSING_TOKEN, "unknown").astype(str)

    # everything else is numeric
    numeric_cols = [c for c in X.columns if c not in NOMINAL]
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="raise")

    return X, y


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in NOMINAL]


# --- custom transformers
class RareLevelGrouper(BaseEstimator, TransformerMixin):
    """Fold rare levels of a nominal feature into a single 'other' level.

    Fits on the training fold. A level survives if it appears in at least
    min_frequency of the training rows. Anything rarer, or anything unseen at
    fit time, becomes 'other'.

    proto has 133 levels. One-hot encoding all of them would let the nominal
    block carry more than three quarters of every distance, for levels that
    mostly appear in well under one row in a thousand.
    """

    def __init__(self, columns: list[str] | None = None,
                 min_frequency: float = MIN_LEVEL_FREQUENCY):
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        self.keep_: dict[str, set] = {}
        for c in cols:
            freq = X[c].value_counts(normalize=True)
            keep = set(freq.index[freq >= self.min_frequency])
            if not keep:                       # degenerate guard
                keep = {freq.index[0]}
            self.keep_[c] = keep
        self.columns_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns_:
            keep = self.keep_[c]
            X[c] = np.where(X[c].isin(keep), X[c], "other")
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one member of every numeric pair with |r| above a threshold.

    Fits on the training fold. Of a pair, the one with the higher mean absolute
    correlation against everything else goes, so the more distinctive feature
    stays. Ties break on column order to keep the result deterministic.

    Duplicated information gets summed twice inside a distance, which quietly
    double-weights whatever those two features measure.
    """

    def __init__(self, threshold: float = CORR_THRESHOLD):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        C = X.corr(numeric_only=True).abs()
        cols = list(C.columns)
        A = C.to_numpy(copy=True)
        np.fill_diagonal(A, np.nan)
        redundancy = np.nanmean(A, axis=1)          # mean |r| against all others
        rank = {c: (redundancy[i], i) for i, c in enumerate(cols)}

        drop: set[str] = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if a in drop or b in drop:
                    continue
                v = A[i, j]
                if np.isfinite(v) and v >= self.threshold:
                    drop.add(a if rank[a] > rank[b] else b)

        self.dropped_ = sorted(drop)
        self.kept_ = [c for c in cols if c not in drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_]


class ConstantFilter(BaseEstimator, TransformerMixin):
    """Drop numeric columns with zero variance on the training fold."""

    def fit(self, X: pd.DataFrame, y=None):
        nun = X.nunique()
        self.kept_ = [c for c in X.columns if nun[c] > 1]
        self.dropped_ = [c for c in X.columns if nun[c] <= 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_]


# --- pipelines
def build_knn_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Preprocessing for k-NN.

    numeric: drop constants, then map each feature onto its empirical quantile
    nominal: group rare levels, then one-hot encode

    Min-max and z-score are both anchored on statistics the upper tail
    controls, and skewness here reaches 173.2, with sbytes alone topping out at
    27189 times its median. Under either one almost every observation ends up
    squashed into a sliver near zero. The rank transform has no tail
    dependence, keeps the ordering, and puts the numeric block on the same
    [0, 1] footing as the one-hot block. The normalisation study backs this up.

    No correlation filter. Three thresholds were tried and none beat keeping
    every feature. CorrelationFilter stays here for that study.
    """
    numeric_branch = Pipeline([
        ("constant", ConstantFilter()),
        ("scale", QuantileTransformer(output_distribution="uniform",
                                      n_quantiles=1000, subsample=200_000,
                                      random_state=RANDOM_SEED)),
    ])
    nominal_branch = Pipeline([
        ("rare", RareLevelGrouper(columns=NOMINAL, min_frequency=MIN_LEVEL_FREQUENCY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_branch, numeric_cols),
         ("nom", nominal_branch, NOMINAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_ct_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Preprocessing for the tree.

    numeric: untouched
    nominal: ordinal encoded

    No scaling, because a threshold test is invariant under any monotone
    rescaling. No correlation filter, because at each node the tree takes
    whichever member of a pair gives the higher gain and ignores the other.
    """
    nominal_branch = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])
    return ColumnTransformer(
        [("num", "passthrough", numeric_cols),
         ("nom", nominal_branch, NOMINAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --- helpers
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0


def stratified_subsample(X: pd.DataFrame, y: pd.Series, frac: float,
                         seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample preserving the original class proportions."""
    from sklearn.model_selection import train_test_split
    if frac >= 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=seed)
    return Xs.reset_index(drop=True), ys.reset_index(drop=True)


def log(msg: str, path: str | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")
