"""Model zoo. Only numpy/pandas/scikit-learn. LightGBM/XGBoost are NOT used until approved."""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .pipeline import build_pipeline


def get_models():
    """name -> (estimator, engineer_features)"""
    hgb = lambda **kw: HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.05, early_stopping=True, n_iter_no_change=30,
        random_state=SEED, **kw)
    return {
        "Mean (dummy)": (DummyRegressor(), False),
        "Ridge (baseline)": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), False),
        "Ridge + features": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), True),
        "RandomForest": (RandomForestRegressor(200, min_samples_leaf=5, max_features=0.4,
                                               n_jobs=-1, random_state=SEED), True),
        "HistGB": (hgb(), False),
        "HistGB + features": (hgb(), True),
        "HistGB tuned": (hgb(max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0), True),
    }


def get_pipelines():
    return {n: build_pipeline(est, engineer=eng) for n, (est, eng) in get_models().items()}
