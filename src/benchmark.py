"""python -m src.benchmark  -> reports/results.csv, reports/preds/*.npz, reports/performance_report.md"""
import numpy as np
import pandas as pd

from .config import METRIC, N_FOLDS, REPORT_DIR, SEED, TARGET
from .cv import run_cv
from .data import load_test, load_train
from .models import get_pipelines


def main():
    X, y, _ = load_train()
    X_test, _ = load_test()
    (REPORT_DIR / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, pipe in get_pipelines().items():
        res = run_cv(pipe, X, y, X_test)
        np.savez(REPORT_DIR / "preds" / f"{name.replace(' ', '_')}.npz", oof=res["oof"], test=res["test"])
        rows.append({"model": name, "oof_rmse": res["rmse"], "fold_std": np.std(res["fold_rmse"]),
                     "secs": res["secs"]})
        print(f"{name:20s} OOF {METRIC}={res['rmse']:.4f} ±{rows[-1]['fold_std']:.4f} ({res['secs']:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values("oof_rmse").reset_index(drop=True)
    df.to_csv(REPORT_DIR / "results.csv", index=False)
    write_report(df, y)


def write_report(df, y):
    best = df.iloc[0]
    table = "\n".join(f"| {r.model} | {r.oof_rmse:.4f} | {r.fold_std:.4f} | {r.secs:.0f} |"
                      for r in df.itertuples())
    (REPORT_DIR / "performance_report.md").write_text(f"""# Performance report

## Problem type
**Tabular regression.** `{TARGET}` is continuous (210,000 unique values, mean {y.mean():.2f}, sd {y.std():.2f},
range {y.min():.1f}..{y.max():.1f}). It is **not time-series**: `Date` (2022-2024) is spread across both train and
test, each row is a different cow, and there is no per-entity sequence; date only provides yearly seasonality
(Q3 mean ~14.5 L vs Q1 ~16.3 L, same every year), so it is used as cyclic features. `Mastitis` is an input, not the label.

## Validation
{N_FOLDS}-fold shuffled KFold (seed {SEED}); metric {METRIC} on out-of-fold predictions. The whole preprocessing
pipeline is cloned and fit inside each fold (see `src/cv.py`), so imputation medians, winsor quantiles and
encoders never see validation rows. Test predictions are the average of the {N_FOLDS} fold models.

## Results (lower is better)
| Model | OOF {METRIC} | Fold std | Seconds |
|---|---|---|---|
{table}

**Best: {best.model} ({best.oof_rmse:.4f}).**

## Notes
- Removed from the old notebook: row-wise outlier removal (fit on full train incl. target, leaky in CV) -> in-fold winsorizing.
- Dropped duplicate/derivable columns: `Feed_Quantity_lb`, `Rumination_Time_hrs`, `Lactation_Stage`.
- LightGBM / XGBoost / CatBoost were not used (not approved/installed); the GBM is scikit-learn `HistGradientBoostingRegressor`.
""")


if __name__ == "__main__":
    main()
