# Cattle Milk Yield Prediction (CS 363M)

Predict the daily milk yield (`Milk_Yield_L`) of individual dairy cattle from breed, climate, feeding, health, housing and date information.

## Team
| Name |
|---|
| Matthew Pham |
| Thai |
| Jack |
| Issay |

The first version of the project (`ML-project.ipynb`) was written by the team. The refactor into `src/`, the leakage-safe pipeline, the re-test of every experiment and `ML-project-v2.ipynb` were done with following an Explore -> Plan -> Code workflow.

## At a glance
- **Problem type: tabular regression** (not classification, not time-series; evidence below).
- **Data:** 210,000 training rows x 36 columns, 40,000 test rows x 35 columns, 1,000 farms, dates 2022-2024.
- **Metric:** RMSE (what v1 optimized; please check it against the competition page).
- **Validation:** 5-fold shuffled KFold, out-of-fold (OOF) predictions, the whole preprocessing pipeline re-fit in every fold.
- **Best model:** tuned scikit-learn `HistGradientBoostingRegressor`, **OOF RMSE 4.1433** (fold std 0.013). Predicting the mean gives 5.3521, Ridge 4.2613, RandomForest 4.2171, v1's MLP 4.1759 (same CV).
- **Allowed libraries:** numpy, pandas, scikit-learn (matplotlib for plots, approved because v1 already used it). LightGBM / XGBoost / CatBoost were *not* used (not approved, not installed).

## Repository layout
```
ML-project.ipynb      original notebook (v1), untouched
ML-project-v2.ipynb   v2: the full investigation + decisions, driving the code in src/ (outputs saved)
src/
  config.py           paths, seed (42), 5 folds, column groups
  data.py             load train / test
  transformers.py     BreedCleaner, ColumnDropper, DateFeatures, FarmMedianImputer, Winsorizer (sklearn transformers)
  features.py         DomainFeatures (row-wise engineered features)
  pipeline.py         build_pipeline(model): clean -> impute -> clip -> features -> one-hot -> model
  cv.py               run_cv: out-of-fold CV, fold-averaged test predictions
  models.py           dummy, Ridge, RandomForest, HistGradientBoosting variants
  benchmark.py        python -m src.benchmark -> reports/results.csv + performance_report.md
  experiments.py      leak-safe versions of every v1 idea; python -m src.experiments -> reports/experiments.csv
  make_submission.py  python -m src.make_submission -> submission.csv
reports/              results.csv, experiments.csv, performance_report.md, preds/*.npz
```
Every Python file is under 250 lines.

## How to run
```bash
python -m src.benchmark          # ~3 min: all models, writes reports/
python -m src.make_submission    # writes submission.csv from the best model's fold-averaged test predictions
python -m src.experiments        # ~15 min, optional: re-runs the v1-ideas study
jupyter nbconvert --execute --inplace ML-project-v2.ipynb   # or open the notebook (RERUN=False reads saved experiment results)
```
Run everything from the project folder. Requires numpy, pandas, scikit-learn (tested with scikit-learn 1.9, pandas 3.0).

## The development process, start to finish

### 1. Looking at the data (v1)
We first opened the CSVs in a spreadsheet. We noticed: many non-numeric columns (breed, climate zone, management system, date), some numbers starting with a `'`, negative values that make no sense (rumination time), blanks, and feed quantity present twice (kg and lb).

### 2. What kind of problem is it?
It is **regression**:
- `Milk_Yield_L` is continuous (all 210,000 values are different) and the submission asks for a decimal per `Cattle_ID`; `Mastitis` is 0/1 but is an input column.
- It is **not time-series**: `Date` runs 2022-01-01 to 2024-12-30 in *both* train and test, so test rows are interleaved with training rows; each row is a different cow (34 repeated IDs); there are no per-cow sequences. Date only gives yearly seasonality (mean yield in Q3 is ~14.5 L vs ~16.3 L in Q1, identically every year), so it is used as a feature and we validate with shuffled KFold instead of a time split.
- All 1,000 farms appear in train and test, so a farm-grouped split is unnecessary.

### 3. Cleaning and imputation (v1 investigation)
- Fixed breed typos/whitespace (`Holstien`, ` Brown Swiss`).
- Missing values only in `Feed_Quantity_kg` (10,481), `Feed_Quantity_lb` (same rows) and `Housing_Score` (6,279). We checked whether whole farms were missing them (no: at most 22 per farm) and whether a farm average would impute better. It would not: within-farm variance equals whole-data variance for both columns (housing 0.0410 vs 0.0410, feed 15.749 vs 15.755).
- Correlation analysis: `Rumination_Time_hrs` is a 1-to-1 copy of `Previous_Week_Avg_Yield` (corr 1.0; more than half its values are also negative) so it is dropped. Strongest correlations with the target: Age -0.31, Weight +0.30, Parity -0.24, Feed +0.22, Water +0.13, Mastitis -0.12, Previous-week yield +0.09, Days-in-milk -0.06. Inputs are nearly independent of each other (apart from kg/lb feed).
- `Lactation_Stage` is defined from `Days_in_Milk`, so it was dropped; IDs were dropped.
- v1 imputed housing with the farm median and feed with a 5-nearest-neighbour imputer (a RandomForest imputer was tried as an alternative).

### 4. Outliers and feature ideas (v1)
We ran four outlier detectors (EllipticEnvelope, IsolationForest, LocalOutlierFactor, OneClassSVM) and removed 1,974 rows flagged by at least two. We also tried capping at percentiles, interactions, log transforms, polynomials and biologically motivated features (days-in-milk stages, feed per kg body weight, heat-stress index).

### 5. Models (v1)
Kernel ridge regression did not work (n x n kernel matrix at 210k rows crashed our machines). We grid-searched an sklearn MLP (81 candidates x 5 folds): best (10, 10) logistic/SGD, reported RMSE 4.1459.

### 6. Audit and refactor (v2)
We audited v1 for leakage (see Challenges), planned a `src/` architecture, and rebuilt the pipeline from scikit-learn transformers so that every step is fit on the training fold only. We then:
1. built the OOF CV harness and a model benchmark (baseline vs gradient boosting);
2. re-implemented *each* v1 idea as an in-fold variant and scored it under identical conditions (`src/experiments.py`);
3. wrote `ML-project-v2.ipynb`, which keeps the whole investigation, the recorded v1 results and the new evidence;
4. generated the performance report and the submission script.

## Results

### Model benchmark (5-fold OOF RMSE, lower is better)
| Model | OOF RMSE | Fold std |
|---|---|---|
| **HistGB tuned** | **4.1433** | 0.0132 |
| HistGB | 4.1458 | 0.0131 |
| HistGB + features | 4.1461 | 0.0128 |
| RandomForest | 4.2171 | 0.0137 |
| Ridge + features | 4.2597 | 0.0136 |
| Ridge (baseline) | 4.2613 | 0.0133 |
| Mean (dummy) | 5.3521 | 0.0075 |

### Re-test of the v1 ideas (all in-fold, tuned HistGB; reference 4.1433)
| Variant | OOF RMSE |
|---|---|
| Housing: global median instead of farm median | 4.1433 |
| No winsorizing / no domain features | 4.1433 / 4.1433 |
| Log of skewed / polynomials | 4.1433 / 4.1433 (identical: trees only use order) |
| Feed: RandomForest imputer | 4.1436 |
| In-fold outlier trimming (IsolationForest) | 4.1438 |
| Month one-hot instead of cyclic month | 4.1440 |
| Interactions | 4.1441 |
| Feed: KNN imputer (6 min per CV run vs 18 s) | 4.1447 |
| **Date dropped** | **4.2734** |
| MLP with v1's best parameters | 4.1759 |

### Findings
1. Gradient boosting beats Ridge, RandomForest and the MLP; the tuned HistGB is ~22.6% better than predicting the mean and ~2.8% better than Ridge.
2. **Seasonality (`Date`) is the only preprocessing choice that matters** (+0.13 RMSE when removed). Imputation method, outlier handling, capping and engineered features all change the score by less than the fold-to-fold noise (~0.013).
3. Farm identity carries almost no information about `Housing_Score` or `Feed_Quantity_kg`.
4. A large share of the target looks like irreducible noise: the best single correlation is only |0.31| and every sensible model ends at RMSE ~4.1-4.3.
5. v1's 4.1459 for the MLP was optimistic; with all steps fit inside folds the same MLP scores 4.1759.

## Challenges and how we solved them
| Challenge | What happened | Solution |
|---|---|---|
| **Dirty categories** | `Holstien` typo and stray spaces (` Brown Swiss`, `Brown Swiss `) split breeds into 7 groups instead of 4. | `BreedCleaner` transformer (strip + fix), applied identically to train and test. |
| **Missing values with no obvious donor** | Feed (10,481) and housing (6,279) are missing, but inputs are nearly uncorrelated, so no column predicts them. | Studied farm-level variance; compared median, KNN and RandomForest imputers; ended with farm/global median + missing flags because the fancy imputers gave no gain. |
| **Duplicate / derived columns** | kg vs lb feed (corr 0.999, same missing rows), `Rumination_Time_hrs` = `Previous_Week_Avg_Yield`, `Lactation_Stage` derived from days-in-milk. | `ColumnDropper` with the list in `config.py`. |
| **Impossible values** | More than half of the rumination hours were negative; 74 negative milk yields. | Column dropped as a duplicate; targets left alone (kept in training, treated as noise) and checked that trimming outliers did not help. |
| **Slow / impossible models** | Kernel ridge needs an n x n matrix (210k^2) and crashed our machines; KNN imputation costs 444 s per 5-fold run; RandomForest 128 s. | Switched to memory-friendly models; histogram gradient boosting trains in ~18 s per 5-fold run. |
| **Outlier detector warnings** | EllipticEnvelope warned about a non-full-rank covariance (one-hot columns) and LOF / OneClassSVM do not scale. | Replaced row removal by in-fold winsorizing and tested trimming with IsolationForest inside the folds. |
| **Data leakage in v1** | Outlier detectors (using the target), percentile caps, medians, imputers and scalers were fit on the whole training set before CV, so validation rows influenced preprocessing. | Every stateful step is an sklearn transformer fit inside each fold; `run_cv` clones the entire pipeline per fold. |
| **Train/test column mismatch** | `get_dummies` was run on train and test separately, and v1 had to hand-copy each cleaning step to the test set and re-align columns. | One `Pipeline` with `OneHotEncoder(handle_unknown='ignore')`; `predict(X_test)` applies all steps. |
| **Noisy target** | Many promising feature ideas changed nothing. | Quantified the noise floor (fold std ~0.013) and judged each idea against it instead of by single-run differences. |
| **Monolithic notebook** | Everything lived in one notebook with cells depending on hidden state. | Modular `src/`, each file < 250 lines; notebook now drives the code and records decisions. |

## Limitations and next steps
- The metric is assumed to be RMSE; confirm against the competition page.
- No hyper-parameter search beyond a hand-picked tuned configuration; no stacking/blending.
- LightGBM, XGBoost and CatBoost may squeeze out a little more, but only after approval to add them.
- Iteration on feature engineering is unlikely to help; more promising would be understanding the noise in the target (for example, any structure in the 74 negative yields).
- Housing and feed imputation could be revisited if new columns correlating with them become available.
