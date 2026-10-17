import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from .config import N_FOLDS, SEED


def run_cv(pipe, X, y, X_test):
    """Regression -> shuffled KFold (rows are independent cows, dates interleave train/test, so no
    time or group split). A fresh clone of the whole pipeline is fit per fold, so every stateful
    step sees only that fold's training rows. Returns OOF preds, fold-averaged test preds, fold RMSEs."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_rmse = []
    t0 = time.time()
    for tr, va in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / N_FOLDS
        fold_rmse.append(root_mean_squared_error(y.iloc[va], oof[va]))
    return {"oof": oof, "test": test_pred, "fold_rmse": fold_rmse,
            "rmse": root_mean_squared_error(y, oof), "secs": time.time() - t0}
