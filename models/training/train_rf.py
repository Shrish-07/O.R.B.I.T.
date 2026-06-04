import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_loader import load_orbit_split, prepare_features


def main(mode, scope):
    df_train, df_test = load_orbit_split(scope)

    X_train, y_train, features = prepare_features(df_train, mode)
    X_test, y_test, _ = prepare_features(df_test, mode)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    results = {
        "model": "RandomForest",
        "mode": mode,
        "scope": scope,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": int(len(features)),
        "r2": float(r2),
        "mae": float(mae),
        "features": features,
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"rf_{mode}_{scope}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ ORBIT v2 — RandomForest ({mode.upper()} | {scope.upper()})")
    print("Train rows:   ", len(X_train))
    print("Test rows:    ", len(X_test))
    print("Features used:", len(features))
    print("R²:           ", round(r2, 4))
    print("MAE:          ", round(mae, 4))
    print("📁 Results saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "political"], required=True)
    parser.add_argument("--scope", choices=["2017", "all"], required=True)
    args = parser.parse_args()

    main(args.mode, args.scope)
