"""Stateful steps learn only in fit() so CV folds never see validation/test statistics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FARM_COL


class BreedCleaner(BaseEstimator, TransformerMixin):
    """Strip whitespace and fix the 'Holstien' typo."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Breed"] = X["Breed"].str.strip().replace({"Holstien": "Holstein"})
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class DateFeatures(BaseEstimator, TransformerMixin):
    """Cyclic month/day-of-year (target is strongly seasonal); drops the raw date."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        d = pd.to_datetime(X.pop("Date"))
        for name, frac in (("month", (d.dt.month - 1) / 12), ("doy", (d.dt.dayofyear - 1) / 366)):
            X[f"{name}_sin"] = np.sin(2 * np.pi * frac)
            X[f"{name}_cos"] = np.cos(2 * np.pi * frac)
        return X


class FarmMedianImputer(BaseEstimator, TransformerMixin):
    """Housing_Score -> farm median (low within-farm variance); Feed_Quantity_kg -> global median
    (feed is uncorrelated with farm). Adds missing flags, then drops Farm_ID.
    Medians come from fit() data only; unseen farms fall back to the global median."""

    def fit(self, X, y=None):
        self.farm_housing_ = X.groupby(FARM_COL)["Housing_Score"].median()
        self.housing_ = X["Housing_Score"].median()
        self.feed_ = X["Feed_Quantity_kg"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["housing_missing"] = X["Housing_Score"].isna().astype(int)
        X["feed_missing"] = X["Feed_Quantity_kg"].isna().astype(int)
        farm = X[FARM_COL].map(self.farm_housing_)
        X["Housing_Score"] = X["Housing_Score"].fillna(farm).fillna(self.housing_)
        X["Feed_Quantity_kg"] = X["Feed_Quantity_kg"].fillna(self.feed_)
        return X.drop(columns=[FARM_COL])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip features to quantiles learned on the training fold (replaces row-wise outlier removal)."""

    def __init__(self, columns=(), lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        cols = list(self.columns)
        self.lo_ = X[cols].quantile(self.lower)
        self.hi_ = X[cols].quantile(self.upper)
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.columns)
        X[cols] = X[cols].clip(self.lo_, self.hi_, axis=1)
        return X
