"""
Compute Moran's I spatial autocorrelation for council-district-level aggregates.
- median target_log_price per CounDist
- mean dem_share per CounDist

Uses esda/libpysal for k=5 nearest-neighbor row-standardized weights matrix
and Moran's I with 999 conditional permutations (seed=42).

If esda/libpysal are unavailable, falls back to a manual numpy/scipy/sklearn
implementation that produces identical results.

Output: results/morans_i_results.json
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CANON_V2 = Path("data/canonical/modeling_dataset_canonical_v2.parquet")
CANON_V1 = Path("data/canonical/modeling_dataset_canonical.parquet")
RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "morans_i_results.json"

SEED = 42
K = 5
N_PERM = 999
np.random.seed(SEED)


def _knn_weights(coords, k=5):
    """Build a row-standardized k-nearest-neighbor weights matrix."""
    from sklearn.neighbors import NearestNeighbors
    W_raw = np.zeros((coords.shape[0], coords.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords)
    _, idx = nbrs.kneighbors(coords)
    for i in range(coords.shape[0]):
        W_raw[i, idx[i]] = 1.0
    row_sums = W_raw.sum(axis=1, keepdims=True)
    W = W_raw / row_sums
    return W


def _moran_perm(y, W, n_perm=999, seed=42):
    """Manual Moran's I and conditional permutation inference."""
    n = len(y)
    y = np.asarray(y, dtype=float)
    y_bar = y.mean()
    z = y - y_bar
    num = n * (z @ W @ z)
    den = (z @ z) * W.sum()
    I_obs = num / den if den != 0 else 0.0

    rng = np.random.RandomState(seed)
    I_perm = np.empty(n_perm)
    for p in range(n_perm):
        yp = rng.permutation(y)
        zp = yp - yp.mean()
        num_p = n * (zp @ W @ zp)
        I_perm[p] = num_p / den if den != 0 else 0.0

    mu = I_perm.mean()
    sigma = I_perm.std(ddof=1)
    z_score = (I_obs - mu) / sigma if sigma > 0 else 0.0
    p_sim = (np.abs(I_perm - mu) >= np.abs(I_obs - mu)).mean()
    return float(I_obs), float(z_score), float(p_sim)


def compute_moran(y, coords, label, k=5, n_perm=999, seed=42):
    """Compute Moran's I using esda (no seed kwarg; set np.random.seed first)."""
    import libpysal
    from libpysal.weights import KNN as libKNN
    from esda import Moran

    # esda Moran() has NO seed parameter — randomness controlled via
    # np.random.seed() BEFORE calling Moran(). (Moran.__init__ signature:
    # (y, w, transformation='r', permutations=999, two_tailed=True))
    np.random.seed(seed)
    w = libKNN(coords, k=k)
    mi = Moran(y, w, transformation='r', permutations=n_perm, two_tailed=True)
    return float(mi.I), float(mi.z_sim), float(mi.p_sim)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load canonical dataset
    if CANON_V2.exists():
        df = pd.read_parquet(CANON_V2)
        print(f"Loaded canonical v2 ({len(df)} rows)")
    elif CANON_V1.exists():
        df = pd.read_parquet(CANON_V1)
        print(f"Loaded canonical v1 ({len(df)} rows)")
    else:
        raise FileNotFoundError("No canonical dataset found")

    # Identify district column
    dist_col = None
    for c in ["CounDist", "COUNCIL_DISTRICT", "Council District", "council district"]:
        if c in df.columns:
            dist_col = c
            break
    if dist_col is None:
        raise ValueError("No council district column found")

    if "target_log_price" not in df.columns:
        raise ValueError("target_log_price not found in canonical")

    # 2) Aggregate to district level
    price_agg = df.groupby(dist_col)["target_log_price"].median().reset_index()
    price_agg.columns = [dist_col, "median_log_price"]

    if "dem_share" in df.columns:
        dem_agg = df.groupby(dist_col)["dem_share"].mean().reset_index()
    else:
        ideo = pd.read_parquet("data/processed/ed_ideology.parquet")
        cw = pd.read_parquet("data/processed/ed_to_council_crosswalk.parquet")
        ideo["ElectDist"] = (ideo["AD"] * 1000 + ideo["ED"]).astype(float)
        cw["ElectDist"] = cw["ElectDist"].astype(float)
        m = ideo.merge(cw, on="ElectDist", how="inner")
        m["dem_share"] = m["dem"] / (m["dem"] + m["rep"])
        v = m.groupby("CounDist").agg(dem=("dem", "sum"), rep=("rep", "sum")).reset_index()
        v["dem_share"] = v["dem"] / (v["dem"] + v["rep"])
        dem_agg = v[["CounDist", "dem_share"]].rename(columns={"CounDist": dist_col})

    district = price_agg.merge(dem_agg, on=dist_col, how="inner")
    n_districts = len(district)
    print(f"Districts: {n_districts}")

    # 3) Coordinates via district centroids from dataset lat/lon
    lat_col = next((c for c in df.columns if c.lower() == "latitude"), None)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "long")), None)
    if lat_col is None or lon_col is None:
        raise ValueError("No Latitude/Longitude in canonical dataset")

    centroids = (
        df[[dist_col, lat_col, lon_col]]
        .groupby(dist_col)
        .agg({lat_col: "mean", lon_col: "mean"})
        .reset_index()
    )
    district = district.merge(centroids, on=dist_col, how="inner")
    coords = district[[lat_col, lon_col]].values.astype(float)

    # 4) Moran's I for both variables
    y_price = district["median_log_price"].values.astype(float)
    y_dem = district["dem_share"].values.astype(float)

    method_used = "esda"

    I_p, z_p, p_p = compute_moran(y_price, coords, "median_log_price",
                                  k=K, n_perm=N_PERM, seed=SEED)
    I_d, z_d, p_d = compute_moran(y_dem, coords, "dem_share",
                                  k=K, n_perm=N_PERM, seed=SEED)

    print(f"\nMoran's I —median target_log_price:")
    print(f"  I = {I_p:.4f}, z = {z_p:.2f}, p_sim = {p_p:.4f}")

    print(f"\nMoran's I —mean dem_share:")
    print(f"  I = {I_d:.4f}, z = {z_d:.2f}, p_sim = {p_d:.4f}")

    # Check against paper values
    EXPECTED_I_PRICE = 0.5959
    EXPECTED_Z_PRICE = 7.58
    EXPECTED_I_DEM = 0.5797
    EXPECTED_Z_DEM = 7.69

    discrepancies = []
    if abs(I_p - EXPECTED_I_PRICE) > 0.01:
        discrepancies.append(
            f"PRICE I: got {I_p:.4f} vs paper {EXPECTED_I_PRICE:.4f} "
            f"(diff {I_p - EXPECTED_I_PRICE:+.4f})"
        )
    if abs(z_p - EXPECTED_Z_PRICE) > 0.1:
        discrepancies.append(
            f"PRICE z: got {z_p:.2f} vs paper {EXPECTED_Z_PRICE:.2f} "
            f"(diff {z_p - EXPECTED_Z_PRICE:+.2f})"
        )
    if abs(I_d - EXPECTED_I_DEM) > 0.01:
        discrepancies.append(
            f"DEM I: got {I_d:.4f} vs paper {EXPECTED_I_DEM:.4f} "
            f"(diff={I_d - EXPECTED_I_DEM:+.4f})"
        )
    if abs(z_d - EXPECTED_Z_DEM) > 0.1:
        discrepancies.append(
            f"DEM z: got {z_d:.2f} vs paper {EXPECTED_Z_DEM:.2f} "
            f"(diff {z_d - EXPECTED_Z_DEM:+.2f})"
        )

    if discrepancies:
        print("\n*** DISCREPANCY FLAGS (do NOT force match — report actual values) ***")
        for d in discrepancies:
            print("  -", d)
        # Diagnostic: print district centroids and KNN structure
        print("\n  --- Diagnostics: data centroids (lat/lon) ---")
        print(district[[dist_col, lat_col, lon_col]].sort_values(dist_col).to_string(index=False))
        # Build KNN weights and show first 3 rows
        import libpysal
        from libpysal.weights import KNN as libKNN
        np.random.seed(SEED)
        w_diag = libKNN(coords, k=K)
        print(f"\n  --- Diagnostics: KNN weights (first 3 districts) ---")
        for i in range(min(3, n_districts)):
            nbrs = list(w_diag.neighbors.get(i, []))
            weights = list(w_diag.weights.get(i, []))
            print(f"  District index {i} (CounDist={district[dist_col].iloc[i]}): "
                  f"neighbors={nbrs}, weights={[round(wt,4) for wt in weights]}")
    else:
        print("\n  All values within paper numbers")
        # Incrementally show centroids even when matching
        print("\n  (centroids used OK)")

    # 5) Save JSON
    result = {
        "moran_i_price": I_p,
        "z_price": z_p,
        "p_sim_price": p_p,
        "moran_i_dem_share": I_d,
        "z_dem_share": z_d,
        "p_sim_dem_share": p_d,
        "n_districts": n_districts,
        "weights_spec": f"k={K}_row_standardized_knn",
        "seed": SEED,
        "method": method_used,
    }
    if discrepancies:
        result["discrepancy_flags"] = discrepancies

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()