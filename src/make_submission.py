"""python -m src.make_submission  (run src.benchmark first) -> submission.csv from the best OOF model."""
import numpy as np
import pandas as pd

from .config import ID_COL, REPORT_DIR, ROOT, SAMPLE_SUB, TARGET
from .data import load_test


def main():
    best = pd.read_csv(REPORT_DIR / "results.csv").iloc[0]["model"]
    pred = np.load(REPORT_DIR / "preds" / f"{best.replace(' ', '_')}.npz")["test"]
    _, ids = load_test()
    sub = pd.DataFrame({ID_COL: ids, TARGET: pred})
    ref = pd.read_csv(SAMPLE_SUB)
    assert len(sub) == len(ref) and (sub[ID_COL].values == ref[ID_COL].values).all(), "ids differ from sample"
    assert sub[TARGET].notna().all()
    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"wrote submission.csv from '{best}' ({len(sub)} rows)")


if __name__ == "__main__":
    main()
