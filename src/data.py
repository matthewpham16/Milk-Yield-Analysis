import pandas as pd

from .config import ID_COL, TARGET, TEST_CSV, TRAIN_CSV


def load_train():
    """Return X (features incl. Farm_ID/Date), y, and ids."""
    df = pd.read_csv(TRAIN_CSV)
    return df.drop(columns=[TARGET, ID_COL]), df[TARGET], df[ID_COL]


def load_test():
    df = pd.read_csv(TEST_CSV)
    return df.drop(columns=[ID_COL]), df[ID_COL]
