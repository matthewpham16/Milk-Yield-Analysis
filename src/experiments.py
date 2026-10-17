"""Leak-safe re-implementations of the v1 notebook experiments. Each variant is the final v2 pipeline
with one step swapped/added, so every one is scored by the same OOF CV (src/cv.py)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .models import get_models
from .pipeline import build_pipeline
from .transformers import ColumnDropper, FarmMedianImputer

FEED = "Feed_Quantity_kg"
FEED_FEATURES = ["Age_Months", "Weight_kg", "Parity", "Days_in_Milk", "Water_Intake_L",
                 "Body_Condition_Score", FEED]


class ModelFeedImputer(BaseEstimator, TransformerMixin):
    """v1 feed imputers (KNN k=5 / RandomForest) re-done in-fold. Runs after FarmMedianImputer and uses
    its `feed_missing` flag to find the rows to overwrite; fit() only sees the training fold."""

    def __init__(self, kind="knn"):
        self.kind = kind

    def _masked(self, X):
        tmp = X[FEED_FEATURES].copy()
        tmp.loc[X["feed_missing"] == 1, FEED] = np.nan
        return tmp

    def fit(self, X, y=None):
        tmp = self._masked(X)
        if self.kind == "knn":
            self.model_ = KNNImputer(n_neighbors=5, weights="distance").fit(tmp)
        else:
            ok = tmp[FEED].notna()
            self.model_ = RandomForestRegressor(100, n_jobs=-1, random_state=SEED).fit(
                tmp.loc[ok, FEED_FEATURES[:-1]], tmp.loc[ok, FEED])
        return self

    def transform(self, X):
        X = X.copy()
        miss = X["feed_missing"] == 1
        if not miss.any():
            return X
        if self.kind == "knn":
            X[FEED] = self.model_.transform(self._masked(X))[:, -1]
        else:
            X.loc[miss, FEED] = self.model_.predict(X.loc[miss, FEED_FEATURES[:-1]])
        return X


class GlobalMedianImputer(FarmMedianImputer):
    """Housing_Score -> global median too (tests whether the farm median in v1 actually helps)."""

    def fit(self, X, y=None):
        super().fit(X, y)
        self.farm_housing_ = self.farm_housing_.iloc[:0]
        return self


class MonthOneHot(BaseEstimator, TransformerMixin):
    """v1 date handling: month dummies instead of cyclic features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        m = pd.to_datetime(X.pop("Date")).dt.month
        for k in range(1, 13):
            X[f"month_{k}"] = (m == k).astype(int)
        return X


class Interactions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Feed_x_Weight"] = X[FEED] * X["Weight_kg"]
        X["Temp_x_Humidity"] = X["Ambient_Temperature_C"] * X["Humidity_percent"]
        X["Age_x_Parity_prod"] = X["Age_Months"] * X["Parity"]
        X["DIM_x_PrevYield"] = X["Days_in_Milk"] * X["Previous_Week_Avg_Yield"]
        return X


class Polynomials(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Age_sq"], X["Age_cu"] = X["Age_Months"] ** 2, X["Age_Months"] ** 3
        X["Weight_sq"] = X["Weight_kg"] ** 2
        X["DIM_cu"] = X["Days_in_Milk"] ** 3
        return X


class LogSkewed(BaseEstimator, TransformerMixin):
    """v1 log transform, but which columns are skewed (|skew|>threshold) is learned on the training fold."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        num = X.select_dtypes("number")
        num = num.loc[:, (num.min() >= 0) & (num.nunique() > 2)]  # log1p needs x >= 0; skip flags
        self.cols_ = list(num.columns[num.skew().abs() > self.threshold])
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}_log"] = np.log1p(X[c].clip(lower=0))
        return X


class OutlierTrimmedRegressor(BaseEstimator, RegressorMixin):
    """Row-wise outlier removal done correctly: detector and trimming use TRAINING rows only (and may
    use y), at predict time every row is predicted. Needs numeric input (place after the encoder)."""

    def __init__(self, estimator=None, contamination=0.01):
        self.estimator = estimator
        self.contamination = contamination

    def fit(self, X, y):
        Z = StandardScaler().fit_transform(np.column_stack([X, y]))
        keep = IsolationForest(contamination=self.contamination, random_state=SEED, n_jobs=-1
                               ).fit_predict(Z) == 1
        self.n_removed_ = int((~keep).sum())
        self.estimator_ = clone(self.estimator).fit(X[keep], np.asarray(y)[keep])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def _swap(pipe, name, new, after=None):
    """Replace step `name` (new=None removes it) or, if absent, insert `new` right after `after`."""
    steps = list(pipe.steps)
    names = [n for n, _ in steps]
    if name in names:
        i = names.index(name)
        steps[i:i + 1] = [] if new is None else [(name, new)]
    else:
        steps.insert(names.index(after) + 1, (name, new))
    return Pipeline(steps)


def variants():
    """name -> unfitted pipeline. All use the tuned HistGB; 'v2 final' is the reference."""
    hgb = get_models()["HistGB tuned"][0]
    base = lambda **kw: build_pipeline(clone(hgb), **kw)
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation="logistic", solver="sgd", alpha=0.05,
                       learning_rate_init=0.005, momentum=0.925, max_iter=1000, early_stopping=True,
                       random_state=16)  # best params found by v1 GridSearchCV
    out = {
        "v2 final (reference)": base(),
        "no winsorizing": _swap(base(), "winsor", None),
        "housing: global median (no farm)": _swap(base(), "impute", GlobalMedianImputer()),
        "no domain features": base(engineer=False),
        "feed: KNN imputer (v1)": _swap(base(), "feed_model", ModelFeedImputer("knn"), after="impute"),
        "feed: RandomForest imputer (v1)": _swap(base(), "feed_model", ModelFeedImputer("rf"), after="impute"),
        "date: month one-hot (v1)": _swap(base(), "date", MonthOneHot()),
        "date: dropped": _swap(base(), "date", ColumnDropper(["Date"])),
        "+ interactions (v1)": _swap(base(), "extra", Interactions(), after="features"),
        "+ log of skewed (v1)": _swap(base(), "extra", LogSkewed(), after="features"),
        "+ polynomials (v1)": _swap(base(), "extra", Polynomials(), after="features"),
    }
    trim = base()
    trim.steps[-1] = ("model", OutlierTrimmedRegressor(clone(hgb)))
    out["in-fold outlier trim (IsolationForest)"] = trim
    out["MLP (v1 best params, no features)"] = _swap(base(engineer=False), "model", make_pipeline(
        StandardScaler(), mlp))
    return out


def main(only=None):
    """python -m src.experiments [name ...] -> reports/experiments.csv (OOF RMSE of every variant);
    with names, only those rows are re-run and updated in the existing csv."""
    from .config import REPORT_DIR
    from .cv import run_cv
    from .data import load_test, load_train

    X, y, _ = load_train()
    X_test, _ = load_test()
    rows = []
    for name, pipe in variants().items():
        if only and name not in only:
            continue
        r = run_cv(pipe, X, y, X_test)
        rows.append({"variant": name, "oof_rmse": r["rmse"], "fold_std": float(np.std(r["fold_rmse"])),
                     "secs": r["secs"]})
        print(f"{name:42s} {r['rmse']:.4f} ±{rows[-1]['fold_std']:.4f} ({r['secs']:.0f}s)", flush=True)
    new = pd.DataFrame(rows)
    path = REPORT_DIR / "experiments.csv"
    if only:
        new = pd.concat([pd.read_csv(path).query("variant not in @only"), new])
    new.to_csv(path, index=False)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
