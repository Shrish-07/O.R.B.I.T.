import pandas as pd
from pathlib import Path

TARGET = "target_log_price"

LEAKAGE_FEATURES = [
    "SALE PRICE",
    "assessland",
    "assesstot",
    "SALE_PRICE_NUM",
]


def load_orbit_split(scope):
    split_dir = Path("data/splits")

    if scope == "2017":
        train_path = split_dir / "year2017_train.parquet"
        test_path = split_dir / "year2017_test.parquet"
    elif scope == "all":
        train_path = split_dir / "all_years_train.parquet"
        test_path = split_dir / "all_years_test.parquet"
    else:
        raise ValueError("scope must be '2017' or 'all'")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing split file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing split file: {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    return df_train, df_test


def prepare_features(df, mode):
    df = df.copy()

    dropped = []
    for col in LEAKAGE_FEATURES:
        if col in df.columns:
            dropped.append(col)
            df = df.drop(columns=[col])

    if dropped:
        print("🚫 Dropping leakage features:")
        for col in dropped:
            print(f"   {col}")

    drop_cols = {TARGET, "SALE DATE"}

    features = [
        c for c in df.columns
        if c not in drop_cols and df[c].dtype != "object"
    ]

    if mode == "base":
        features = [
            c for c in features
            if "ideology" not in c.lower()
            and "vote" not in c.lower()
            and "dem" not in c.lower()
            and "rep" not in c.lower()
        ]

    elif mode == "political":
        assert "district_ideology" in df.columns, \
            "district_ideology column missing from dataset"

        if "district_ideology" not in features:
            features.append("district_ideology")

        political_features = [
            c for c in features
            if any(k in c.lower() for k in ["ideology", "dem", "rep", "vote"])
        ]

        assert len(political_features) > 0, \
            "Political mode has no ideology features — merge failed?"

    else:
        raise ValueError("mode must be 'base' or 'political'")

    return df[features], df[TARGET], features
