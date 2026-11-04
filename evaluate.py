"""Evaluate the capstone's regression models from prepared feature tables."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "value_change_perc"
FEATURES = [
    "age_in_years", "age^2", "age^3", "price_before", "delta_mins_per_game",
    "league_rank", "delta_assists_p_min", "delta_goals_scored_p_min",
    "season_order", "delta_concede_per_game", "delta_team_goals_per_game",
]


def prepare_historical_features(data):
    """Build the model columns from the original EDA feature export."""
    required = {"position", "season", "age_in_years", "value_change", "price_before"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Historical feature table is missing columns: {', '.join(missing)}")
    data = data.loc[data["position"].eq("Attack")].copy()
    seasons = data["season"].astype("string")
    if not seasons.str.fullmatch(r"\d{2}/\d{2}").fillna(False).all():
        raise ValueError("Expected season labels such as 23/24")
    age = pd.to_numeric(data["age_in_years"], errors="raise")
    price = pd.to_numeric(data["price_before"], errors="raise")
    if price.le(0).any():
        raise ValueError("Historical feature table contains nonpositive starting valuations")
    data["age^2"] = age.pow(2)
    data["age^3"] = age.pow(3)
    data["season_order"] = seasons.str[:2].astype(int) - 11
    data[TARGET] = pd.to_numeric(data["value_change"], errors="raise") / price * 100
    return data


def load_samples(path, from_feature_table=False):
    data = pd.read_csv(path)
    if from_feature_table:
        data = prepare_historical_features(data)
    required = {"position", TARGET, *FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    attackers = data.loc[data["position"].eq("Attack"), FEATURES + [TARGET]]
    attackers = attackers.apply(pd.to_numeric, errors="raise")
    if len(attackers) < 2 or not np.isfinite(attackers.to_numpy()).all():
        raise ValueError(f"{path} needs at least two attacker rows with finite values")
    if attackers["price_before"].le(0).any():
        raise ValueError(f"{path} contains nonpositive starting valuations")
    return attackers


def evaluate(historical, new_season):
    from catboost import CatBoostRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = historical.loc[historical[TARGET].le(650)]
    if len(train) < 5:
        raise ValueError("Need at least five historical rows after target filtering")
    models = {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "RidgeCV": make_pipeline(StandardScaler(), RidgeCV()),
        "LassoCV": make_pipeline(
            StandardScaler(), LassoCV(alphas=np.logspace(-2, 2, 9), max_iter=10_000)
        ),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "CatBoost": CatBoostRegressor(
            depth=5, iterations=154, learning_rate=0.2597, random_seed=42,
            verbose=False, allow_writing_files=False,
        ),
    }
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(new_season[FEATURES])
        actual = new_season[TARGET]
        yield {
            "model": name,
            "train_rows": len(train),
            "test_rows": len(actual),
            "rmse": math.sqrt(mean_squared_error(actual, prediction)),
            "mae": mean_absolute_error(actual, prediction),
            "r2": r2_score(actual, prediction),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical", type=Path)
    parser.add_argument("new_season", type=Path)
    parser.add_argument(
        "--historical-feature-table", action="store_true",
        help="Read the original for_model_nov3.csv and derive the model columns",
    )
    parser.add_argument(
        "--test-season-order", type=int,
        help="Evaluate only this season when the test export covers multiple seasons",
    )
    parser.add_argument("--output", type=Path, default=Path("metrics.csv"))
    args = parser.parse_args()
    try:
        historical = load_samples(args.historical, args.historical_feature_table)
        new_season = load_samples(args.new_season)
        if args.test_season_order is not None:
            new_season = new_season.loc[
                new_season["season_order"].eq(args.test_season_order)
            ]
        if len(new_season) < 2:
            raise ValueError("Need at least two test rows in the selected season")
        if historical["season_order"].max() >= new_season["season_order"].min():
            raise ValueError(
                "All historical seasons must precede the test seasons. "
                "Use --test-season-order to select a later season."
            )
        metrics = list(evaluate(historical, new_season))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
