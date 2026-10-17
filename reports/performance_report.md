# Performance report

## Problem type
**Tabular regression.** `Milk_Yield_L` is continuous (210,000 unique values, mean 15.59, sd 5.35,
range -5.7..44.6). It is **not time-series**: `Date` (2022-2024) is spread across both train and
test, each row is a different cow, and there is no per-entity sequence; date only provides yearly seasonality
(Q3 mean ~14.5 L vs Q1 ~16.3 L, same every year), so it is used as cyclic features. `Mastitis` is an input, not the label.

## Validation
5-fold shuffled KFold (seed 42); metric RMSE on out-of-fold predictions. The whole preprocessing
pipeline is cloned and fit inside each fold (see `src/cv.py`), so imputation medians, winsor quantiles and
encoders never see validation rows. Test predictions are the average of the 5 fold models.

## Results (lower is better)
| Model | OOF RMSE | Fold std | Seconds |
|---|---|---|---|
| HistGB tuned | 4.1433 | 0.0132 | 18 |
| HistGB | 4.1458 | 0.0131 | 16 |
| HistGB + features | 4.1461 | 0.0128 | 18 |
| RandomForest | 4.2171 | 0.0137 | 125 |
| Ridge + features | 4.2597 | 0.0136 | 4 |
| Ridge (baseline) | 4.2613 | 0.0133 | 4 |
| Mean (dummy) | 5.3521 | 0.0075 | 2 |

**Best: HistGB tuned (4.1433).**

## Notes
- Removed from the old notebook: row-wise outlier removal (fit on full train incl. target, leaky in CV) -> in-fold winsorizing.
- Dropped duplicate/derivable columns: `Feed_Quantity_lb`, `Rumination_Time_hrs`, `Lactation_Stage`.
- LightGBM / XGBoost / CatBoost were not used (not approved/installed); the GBM is scikit-learn `HistGradientBoostingRegressor`.
