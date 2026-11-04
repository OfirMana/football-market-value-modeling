# Football Market Value Modeling

**How much does a footballer's market valuation change after a run of matches?**

This Magshimim AI capstone turned historical Transfermarkt valuations, match records, and player appearances into **122,615 player-interval samples**. We engineered features to capture how a player's performance changed between intervals, then compared linear and tree-based regressors on attackers.

![Pipeline from valuation snapshots and match records through interval features and a later-season evaluation](assets/pipeline.svg)

## From match records to a prediction

1. Pair consecutive valuation updates for each player and calculate the percentage change relative to the starting valuation.
2. Join player profiles, appearances, and match records. Aggregate performance between valuation updates.
3. Compare each interval with the player's previous interval. Features include changes in minutes per match, goals and assists per minute, and goals scored and conceded by the team per match.
4. Combine the performance changes with age, the starting valuation, and league and season context. Compare linear baselines with Random Forest and CatBoost.
5. Evaluate the model on a separate export of attacker samples covering 2023/24 and 2024/25.

The original data-building notebook includes matches through the end of the valuation interval. The current evaluation measures estimation after observing that interval's performances. A forecast issued at the start of the interval would require earlier features.

## Results reported in the capstone

| Evaluation | Model | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Historical split | Linear Regression | 54.14 | 31.52 | 0.165 |
| Historical split | Random Forest | 47.30 | 27.91 | 0.363 |
| Historical split | CatBoost | 45.20 | 26.76 | 0.418 |
| Later export | Random Forest | 40.65 | 26.31 | 0.097 |
| Later export | CatBoost | 39.67 | 25.30 | 0.140 |

RMSE and MAE are measured in percentage points. These figures come from the original capstone presentation. The supplied export has 646 rows from 2023/24 and 173 from 2024/25. The notebook's 650% target cutoff removes one 2023/24 row, yielding the reported total of 818. The historical training export also includes 2023/24.

## Experiments that shaped the final model

| Experiment | Decision |
| --- | --- |
| Absolute vs. percentage change | Use percentage change so the target accounts for the player's starting valuation. |
| Feature selection | Compare raw performance, changes from the previous interval, age transformations, and league and season context. |
| Model tuning | Use Optuna for CatBoost and randomized search for Random Forest in the original experiments. |
| Outlier handling | Compare clipping, excluding large increases, and keeping the full distribution. |
| Rare increases | Test Gaussian oversampling and an auxiliary classifier. The classifier missed all six positive cases in the later export and was dropped from the final feature set. |

## Evaluate the models on prepared data

```bash
python -m pip install -r requirements.txt
python evaluate.py data/for_model_nov3.csv data/Attackers_new_samples_for_model.csv --historical-feature-table --test-season-order 13
```

The script trains on historical attackers with valuation changes of at most 650% and reports RMSE, MAE, and R² on the 173 rows encoded as season 13 (2024/25). It adds a mean baseline, fits preprocessing on historical rows, and excludes the outlier classifier's predictions and the target-derived season rank. CatBoost uses the hyperparameters listed in the original presentation. The feature exports must use the same chronological season encoding.

The training input is the feature table produced by the EDA notebooks. The evaluator derives the percentage target, polynomial age features, and chronological season order. The resulting 33,665 attacker rows match the columns used for evaluation in the later historical export.

### Results on 2024/25 only

![2024/25 evaluation comparing CatBoost, Random Forest, linear regression, and the historical mean baseline on MAE and RMSE](assets/evaluation.svg)

The filtered training table contains 33,555 rows. On the 173 later-season rows, CatBoost had a slightly lower MAE than the historical-mean baseline (25.71 vs. 26.16 percentage points). The mean baseline had a lower RMSE (41.15 vs. 41.36). Both models had negative R² on this split. See [metrics.csv](metrics.csv) for the full comparison.

## Project notes

The [technical notes](project-audit.md) trace the Colab workflow from the source archive to evaluation and explain what the current Python entry point reproduces. The original presentation metrics are recorded separately because the exact run configuration has not been verified.

## Authors

**Ofir Mana** and **Noam Gamliel** completed this Magshimim AI capstone together as a two-person team. They jointly developed the dataset, designed the modeling experiments, and evaluated the results.
