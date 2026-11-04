# Technical notes

The workflow below was reconstructed from ten Colab notebooks, the model experiment log, the final presentation, and the original data exports. File names identify the intermediate outputs used by the notebooks.

| Stage | Notebook | Input and output |
| --- | --- | --- |
| Build player intervals | `research.ipynb` | `archive_v450.zip` → `valuations_final_450.csv`. Pair consecutive valuations and aggregate appearances and match statistics. |
| Consolidate the table | `DFconfirmation.ipynb` | `valuations_final_450.csv` → `final_v450.csv`. |
| Explore the domain | `Gdescription.ipynb`, `domain research.ipynb` | Explore a branch from `final_v450.csv` that includes non-penalty goals. |
| Clean and explore features | `EDA1 notebook.ipynb`, `EDA2.ipynb` | Review data quality, distributions, and feature relationships across the September exports. |
| Engineer model features | `EDA3.ipynb` | Build interval-to-interval performance deltas and add league context. Write `processDF_nov3.csv` and `for_model_nov3.csv`. |
| Compare models | `model.ipynb` | Read `for_model_nov3.csv`. Run regression, outlier classification, and oversampling experiments. Export `df_attack_with_predictions_2.csv`. |
| Evaluate later samples | `NEW_Season/newData.ipynb` | Generate `Attackers_new_samples_for_model.csv` and evaluate models trained on historical attackers. |
| Investigate alternatives | `TestDF.ipynb`, `option.ipynb` | Check player-stat calculations and explore an alternative data source. |

## Data and prediction timing

The target is `100 × (ending valuation − starting valuation) / starting valuation`. Each row represents an interval for one player. The later-season notebook collects appearances from the previous valuation date through the current valuation date. The early historical builder uses the current and next dates for its appearance window and takes the previous valuation with a separate shift. The exact alignment of features and target in the final historical export needs validation against intermediate rows before interpreting the experiment as a forecast at a specified time.

The historical source archive contains player valuations, appearances, games, lineups, events, clubs, and competitions. The model feature export contains 122,615 rows across four position groups. The 33,665 attacker rows match the required columns in the historical prediction export after deriving the target, polynomial age features, and season order.

## Historical experiments

The capstone compared absolute and percentage targets, tuned CatBoost with Optuna and Random Forest with randomized search, and tested both clipping and excluding unusually large increases. Gaussian oversampling and an auxiliary classifier explored ways to improve predictions in the upper tail. The classifier missed all six positive cases in the later export and was dropped from the final feature set.

| Evaluation reported in the presentation | Model | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Historical split | Linear regression | 54.14 | 31.52 | 0.165 |
| Historical split | Random Forest | 47.30 | 27.91 | 0.363 |
| Historical split | CatBoost | 45.20 | 26.76 | 0.418 |
| Later export with mixed seasons | Random Forest | 40.65 | 26.31 | 0.097 |
| Later export with mixed seasons | CatBoost | 39.67 | 25.30 | 0.140 |

The mixed export has 819 rows. The notebook's cutoff removes one 2023/24 row, yielding the 818 rows described in the presentation. The README leads with the original test split and reports the 2024/25-only evaluation separately.

## Evaluation protocol

- The 2024/25 evaluation uses the 173 rows encoded as season 13. The original later export also contains 646 rows from 2023/24, a season present in the training export.
- The evaluator excludes historical training targets above 650% and keeps every 2024/25 test row. Its predictors exclude the auxiliary classifier and the target-derived season rank.
- The original notebooks compare clipping and excluding extreme targets, run experiments with Gaussian oversampling, and tune regressors. One CatBoost runner uses the same split for tuning and scoring. The presentation metrics are reported separately because the exact run configuration could not be verified.
- LassoCV emitted a convergence warning during the verified run. Its metrics are included for completeness.

## Reproduction boundary

The Python evaluator has been run successfully from the original model feature export and the later-season export. The path from the raw archive to the feature table remains documented at the notebook level. The CSVs and presentation decks are kept outside this repository bundle. Verify data snapshot provenance and redistribution terms before attaching source data.
