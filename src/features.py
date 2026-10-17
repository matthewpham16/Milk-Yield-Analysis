import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatures(BaseEstimator, TransformerMixin):
    """Row-wise (stateless) biologically motivated features, so no fold leakage is possible."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dim = X["Days_in_Milk"]
        X["DIM_sq"] = dim ** 2
        X["DIM_log"] = np.log1p(dim)
        X["DIM_band"] = pd.cut(dim, [-np.inf, 100, 200, np.inf], labels=False).astype(int)
        t, rh = X["Ambient_Temperature_C"], X["Humidity_percent"]
        X["THI"] = 0.8 * t + rh / 100 * (t - 14.4) + 46.4  # temperature-humidity heat-stress index
        X["Feed_per_kg_bw"] = X["Feed_Quantity_kg"] / X["Weight_kg"]
        X["Water_per_kg_bw"] = X["Water_Intake_L"] / X["Weight_kg"]
        X["Age_x_Parity"] = X["Age_Months"] / X["Parity"].clip(lower=1)
        return X
