import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path("data/splits")
RESULTS_DIR = Path("experiments/results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(scope: str):
    if scope == "2017":
        train_path = DATA_DIR / "year2017_train.parquet"
        test_path = DATA_DIR / "year2017_test.parquet"
    elif scope == "all":
        train_path = DATA_DIR / "all_years_train.parquet"
        test_path = DATA_DIR / "all_years_test.parquet"
    else:
        raise ValueError("scope must be '2017' or 'all'")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def select_features(df: pd.DataFrame, mode: str):
    blacklist = {
        "target_log_price",
        "sale_price",
        "SALE PRICE",
        "SALE DATE",
    }

    ideology_cols = [c for c in df.columns if "ideology" in c.lower()]

    base_cols = [
        c for c in df.columns
        if c not in blacklist
        and not c.lower().startswith("target")
        and not c.lower().startswith("sale")
        and c not in ideology_cols
    ]

    if mode == "base":
        features = base_cols
    elif mode == "political":
        features = base_cols + ideology_cols
    else:
        raise ValueError("mode must be 'base' or 'political'")

    return features


def train_lgbm(train_df, test_df, features, target):
    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return model, mae, r2, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["base", "political"])
    parser.add_argument("--scope", required=True, choices=["2017", "all"])
    args = parser.parse_args()

    train_df, test_df = load_split(args.scope)

    assert "target_log_price" in train_df.columns
    target = "target_log_price"

    features = select_features(train_df, args.mode)

    # Guardrails
    assert target not in features
    if args.mode == "base":
        assert not any("ideology" in c.lower() for c in features)
    if args.mode == "political":
        assert any("ideology" in c.lower() for c in features)

    model, mae, r2, preds = train_lgbm(
        train_df,
        test_df,
        features,
        target
    )

    out = {
        "model": "LightGBM",
        "version": "O.R.B.I.T v2",
        "mode": args.mode,
        "scope": args.scope,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(features)),
        "features": features,
        "mae": mae,
        "r2": r2,
        "target": target,
        "random_state": 42,
        "params": model.get_params()
    }

    out_name = f"orbit_v2_{args.mode}_{args.scope}.json"
    out_path = RESULTS_DIR / out_name

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    model_path = MODELS_DIR / f"lgbm_orbit_v2_{args.mode}_{args.scope}.txt"
    model.booster_.save_model(str(model_path))

    print("====================================")
    print("O.R.B.I.T v2 — LightGBM")
    print("Mode:", args.mode)
    print("Scope:", args.scope)
    print("MAE:", round(mae, 4))
    print("R²:", round(r2, 4))
    print("Features:", len(features))
    print("Saved JSON:", out_path)
    print("Saved model:", model_path)
    print("====================================")


if __name__ == "__main__":
    main()
