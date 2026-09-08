import pandas as pd
from pathlib import Path
import yaml

TARGET = "target_log_price"

# NOTE (fixed during post-delivery verification): this module previously used
# its own 4-item hardcoded LEAKAGE_FEATURES list instead of the project's real
# config/feature_blacklist.yaml (49 entries). That meant any script importing
# prepare_features() from here (train_ridge.py, train_elasticnet.py) would train
# on ~13 columns every other training script in this repo explicitly excludes as
# leakage (BBL, BIN, block, lot, latitude/longitude, appbbl, taxmap, plutomapid,
# sale_date, sale_year, ...). It now loads the same config/feature_blacklist.yaml
# every other models/training/*.py script uses, so all model families are
# comparable on an identical feature set.
_BLACKLIST_PATH = Path("config/feature_blacklist.yaml")


def _load_blacklist():
    if _BLACKLIST_PATH.exists():
        with open(_BLACKLIST_PATH) as f:
            return {c.lower() for c in yaml.safe_load(f).get("leakage_features", [])}
    return set()


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

    blacklist = _load_blacklist()
    dropped = [c for c in df.columns if c.lower() in blacklist]
    if dropped:
        print("🚫 Dropping leakage features (config/feature_blacklist.yaml):")
        for col in dropped:
            print(f"   {col}")

    drop_cols = {TARGET} | set(dropped)

    # Numeric/bool columns only — excludes object AND datetime64 dtypes.
    # (Previously only excluded "object", which let datetime64 columns like
    # sale_date through and crashed sklearn's SimpleImputer downstream.)
    features = [
        c for c in df.columns
        if c not in drop_cols
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
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
        # dem_share (built by src/data_pipeline/build_ideology_scores.py, merged
        # in src/data_pipeline/rebuild_canonical.py) is the ONLY ideology feature
        # that reaches any model — see README "Political modeling" section.
        # district_ideology is the abandoned, blacklisted pipeline; asserting on
        # it here was a latent bug that crashed every mode="political" caller.
        assert "dem_share" in df.columns, \
            "dem_share column missing from dataset"

        if "dem_share" not in features:
            features.append("dem_share")

        political_features = [
            c for c in features
            if any(k in c.lower() for k in ["ideology", "dem", "rep", "vote"])
        ]

        assert len(political_features) > 0, \
            "Political mode has no ideology features — merge failed?"

    else:
        raise ValueError("mode must be 'base' or 'political'")

    return df[features], df[TARGET], features
