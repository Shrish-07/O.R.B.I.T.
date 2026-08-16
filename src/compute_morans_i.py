"""
Compute Moran's I spatial autocorrelation for council-district-level aggregates,
using the OFFICIAL NYC City Council District boundary shapefile.

Variables (district-level aggregates merged from the canonical dataset):
- median target_log_price per CounDist
- mean dem_share per CounDist

Spatial weights:
- Coordinates come from the council-district POLYGON CENTROIDS
  (geometry.centroid), NOT from the mean lat/lon of individual sale
  transactions. (Earlier versions of this script grouped sale rows by
  CounDist and averaged their Latitude/Longitude, which is wrong.)
- k=5 nearest-neighbor weights, row-standardized.

Inference:
- esda.Moran, 999 conditional permutations, seed=42.

This script requires esda and libpysal (declared in requirements.txt).

Output: results/morans_i_results.json
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CANON_V2 = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
CANON_V1 = ROOT / "data" / "canonical" / "modeling_dataset_canonical.parquet"
# Official NYC City Council District boundary shapefile (the Council District
# file — NOT the geo_export_*.shp Election District file in the same folder).
COUNCIL_SHP = ROOT / "data" / "raw" / "election_districts" / "NYC_City_Council_Districts.shp"
# District-level ideology used to cross-check the dem_share aggregate.
IDEO_BY_COUNCIL = ROOT / "data" / "processed" / "ideology_by_council.parquet"
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "morans_i_results.json"

SEED = 42
K = 5
N_PERM = 999
np.random.seed(SEED)


def compute_moran(y, coords, k=5, n_perm=999, seed=42):
    """Compute Moran's I using esda (no seed kwarg; set np.random.seed first)."""
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

    # 2) Aggregate to district level (median target_log_price, mean dem_share)
    price_agg = df.groupby(dist_col)["target_log_price"].median().reset_index()
    price_agg.columns = [dist_col, "median_log_price"]

    if "dem_share" not in df.columns:
        # dem_share is a first-class column of the canonical dataset (built by
        # src/build_ideology_scores.py -> ideology_by_council.parquet, merged in
        # src/rebuild_canonical.py). If it is absent in production that is a real
        # bug worth surfacing, so we raise instead of silently routing around it
        # via the abandoned district_ideology / ed_ideology parser.
        raise ValueError(
            "dem_share missing from canonical dataset. This indicates an upstream "
            "regression in src/rebuild_canonical.py / src/build_ideology_scores.py "
            "(ideology_by_council.parquet). NOT restoring the ed_ideology.parquet "
            "fallback: that pipeline is the blacklisted district_ideology pipeline, "
            "not the dem_share production path."
        )
    dem_agg = df.groupby(dist_col)["dem_share"].mean().reset_index()

    district = price_agg.merge(dem_agg, on=dist_col, how="inner")
    n_districts = len(district)
    print(f"Districts in canonical aggregates: {n_districts}")

    # 3) Load the OFFICIAL NYC City Council District boundary shapefile and
    #    build spatial weights from POLYGON CENTROIDS (geometry.centroid),
    #    NOT sale-transaction lat/lon centroids.
    print(f"\nLoading Council District shapefile: {COUNCIL_SHP}")
    if not COUNCIL_SHP.exists():
        raise FileNotFoundError(f"Council District shapefile not found: {COUNCIL_SHP}")
    gdf = gpd.read_file(COUNCIL_SHP)
    print(f"  shapefile columns: {gdf.columns.tolist()}")
    print(f"  shapefile rows: {len(gdf)}")

    # Join key: the shapefile's attribute table uses 'CounDist', which matches the
    # canonical dataset's 'CounDist' column directly. Confirm explicitly rather
    # than guess which council-id field to merge on.
    join_key = None
    for cand in ["CounDist", "coun_dist", "COUNCIL_DISTRICT", "Council District"]:
        if cand in gdf.columns:
            join_key = cand
            break
    if join_key is None:
        raise ValueError(
            f"No council-district identifier found in shapefile columns "
            f"{gdf.columns.tolist()}"
        )
    print(f"  join key (shapefile -> canonical): {join_key} -> {dist_col}")

    # Normalize both keys to comparable numeric types.
    gdf[join_key] = pd.to_numeric(gdf[join_key], errors="coerce")
    district[dist_col] = pd.to_numeric(district[dist_col], errors="coerce")

    # Merge the district-level aggregates onto the polygon geometries.
    merged = gdf[[join_key, "geometry"]].merge(
        district, left_on=join_key, right_on=dist_col, how="inner"
    )
    print(f"  districts matched after merge: {len(merged)} (of {n_districts} aggregates, {len(gdf)} polygons)")

    if len(merged) != n_districts:
        missing = set(district[dist_col].dropna().unique()) - set(merged[dist_col].dropna().unique())
        print(f"  WARNING: {len(missing)} aggregated districts had no matching polygon: {sorted(missing)}")

    # Polygon centroids (in the shapefile's native projected CRS — distances here
    # are only used to pick the k nearest neighbours, so the CRS choice does not
    # bias the contiguity structure).
    merged = merged.sort_values(dist_col).reset_index(drop=True)
    centroid = merged.geometry.centroid
    centroid = centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)

    print("\n  --- Polygon centroids used for KNN (CounDist, x, y) ---")
    print(merged.assign(_x=coords[:, 0], _y=coords[:, 1])[
        [dist_col, "_x", "_y"]
    ].to_string(index=False))

    # 4) Moran's I for both variables
    y_price = merged["median_log_price"].values.astype(float)
    y_dem = merged["dem_share"].values.astype(float)

    method_used = "esda_KNN_polygon_centroids"

    I_p, z_p, p_p = compute_moran(y_price, coords, k=K, n_perm=N_PERM, seed=SEED)
    I_d, z_d, p_d = compute_moran(y_dem, coords, k=K, n_perm=N_PERM, seed=SEED)

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
    else:
        print("\n  All values within paper numbers")

    # Cross-check: mean dem_share per district in the canonical aggregate should
    # be comparable to the per-year dem_share in ideology_by_council.parquet.
    ideo_cross_ref = {}
    if IDEO_BY_COUNCIL.exists():
        ideo = pd.read_parquet(IDEO_BY_COUNCIL)
        for cd in sorted(ideo["CounDist"].dropna().unique()):
            sub = ideo[ideo["CounDist"] == cd]
            ideo_cross_ref[str(int(cd))] = {
                str(int(r["election_year"])): float(r["dem_share"])
                for _, r in sub.iterrows()
            }

    # 5) Save JSON, with a self-documenting methodology block.
    result = {
        "moran_i_price": I_p,
        "z_price": z_p,
        "p_sim_price": p_p,
        "moran_i_dem_share": I_d,
        "z_dem_share": z_d,
        "p_sim_dem_share": p_d,
        "n_districts": n_districts,
        "n_polygons_matched": int(len(merged)),
        "weights_spec": f"k={K}_row_standardized_knn",
        "seed": SEED,
        "method": method_used,
        "methodology": {
            "shapefile_path": str(COUNCIL_SHP.relative_to(ROOT)),
            "shapefile_crs": str(merged.crs),
            "coordinate_source": "council-district POLYGON CENTROIDS (geometry.centroid of the official NYC City Council District boundary shapefile)",
            "join_key_shapefile": join_key,
            "join_key_canonical": dist_col,
            "weights_construction": f"libpysal KNN k={K}, row-standardized (esda Moran transformation='r')",
            "permutations": N_PERM,
            "aggregates": {
                "median_log_price": "median of canonical target_log_price grouped by CounDist",
                "dem_share": "mean of canonical dem_share grouped by CounDist",
            },
            "ideology_by_council_cross_reference": ideo_cross_ref,
        },
    }
    if discrepancies:
        result["discrepancy_flags"] = discrepancies
        result["paper_targets"] = {
            "moran_i_price": EXPECTED_I_PRICE,
            "z_price": EXPECTED_Z_PRICE,
            "moran_i_dem_share": EXPECTED_I_DEM,
            "z_dem_share": EXPECTED_Z_DEM,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()