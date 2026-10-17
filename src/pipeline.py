from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CAT_COLS, DROP_COLS, WINSOR_COLS
from .features import DomainFeatures
from .transformers import BreedCleaner, ColumnDropper, DateFeatures, FarmMedianImputer, Winsorizer


def build_pipeline(model, engineer=True):
    """Raw dataframe -> clean -> (features) -> one-hot -> model. Everything fits inside CV folds."""
    steps = [
        ("breed", BreedCleaner()),
        ("drop", ColumnDropper(DROP_COLS)),
        ("date", DateFeatures()),
        ("impute", FarmMedianImputer()),
        ("winsor", Winsorizer(WINSOR_COLS)),
    ]
    if engineer:
        steps.append(("features", DomainFeatures()))
    steps.append(("encode", ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
        remainder="passthrough", verbose_feature_names_out=False)))
    steps.append(("model", model))
    return Pipeline(steps)
