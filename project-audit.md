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

The target is `100 × (ending valuation − starting valuation) / starting valuation`. Each row represents an interval for one player. The source code collects matches through the interval's closing date. A deployment would need a defined scoring time and a check that each match and league statistic was available at that time.

The historical source archive contains player valuations, appearances, games, lineups, events, clubs, and competitions. The model feature export contains 122,615 rows across four position groups. The 33,665 attacker rows match the required columns in the historical prediction export after deriving the target, polynomial age features, and season order.

## Evaluation protocol

- The 2024/25 evaluation uses the 173 rows encoded as season 13. The original later export also contains 646 rows from 2023/24, a season present in the training export.
- The evaluator excludes historical training targets above 650% and keeps every 2024/25 test row. Its predictors exclude the auxiliary classifier and the target-derived season rank.
- The original notebooks compare clipping and excluding extreme targets, run experiments with Gaussian oversampling, and tune regressors. One CatBoost runner uses the same split for tuning and scoring. The presentation metrics are reported separately because the exact run configuration could not be verified.
- LassoCV emitted a convergence warning during the verified run. Its metrics are included for completeness.

## Reproduction boundary

The Python evaluator has been run successfully from the original model feature export and the later-season export. The path from the raw archive to the feature table remains documented at the notebook level. The CSVs and presentation decks are kept outside this repository bundle. Verify data snapshot provenance and redistribution terms before attaching source data.
