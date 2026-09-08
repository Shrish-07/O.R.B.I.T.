"""Independent verification of Section 4.5 / Table 5 (district-level spatial
lag and spatial error regression).

Pattern follows verify_regression_ols.py / verify_followup_resolution.py and
the internal cross-check idiom of generate_robustness_checks.py: the spatial
models are RE-FIT here from raw inputs (canonical parquet + official Council
District shapefile, k=5 row-standardized KNN on polygon centroids, seed 42)
WITHOUT importing anything from src/analysis/spatial_regression_model.py, and
the recomputed values are then compared against
  (a) the manuscript's reported numbers (Table 5 / Section 4.5 prose) and
  (b) the pipeline's own results/spatial_regression.json output.

Writes results/spatial_regression_verification.json (with matches_paper /
matches_pipeline_output booleans, the same convention used by the other
verify_*.py scripts).

Requires: spreg (pip install spreg), libpysal, geopandas — same stack as
src/analysis/compute_morans_i.py.

Manuscript claims verified (orbit_paper_CER.tex, Section 4.5 + Table 5):
  - OLS-anchor residual Moran's I = 0.389 (z = 6.05, p < 0.001)
  - Spatial lag rho = 0.709 (Table 5: 0.7086), p < 0.001
  - Spatial error lambda = 0.787 (Table 5: 0.7872), p < 0.001
  - dem_share p-values: 0.385 (OLS anchor), 0.170 (spatial lag), 0.374 (spatial error)
  - dem_share coefficients positive in all three: 0.201--0.229
  - pseudo-R^2: 0.708 (spatial lag), 0.406 (spatial error)
  - mean_commfar significant in all three (p < 0.01)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
import spreg

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "canonical" / "modeling_dataset_canonical_v2.parquet"
COUNCIL_SHP = ROOT / "data" / "raw" / "election_districts" / "NYC_City_Council_Districts.shp"
PIPELINE_JSON = ROOT / "results" / "spatial_regression.json"
OUT_JSON = ROOT / "results" / "spatial_regression_verification.json"

SEED = 42
K = 5

# Paper's reported values (rounded as printed in Table 5 / Section 4.5 prose).
PAPER = {
    "rho_slm": 0.7086,          # Table 5 cell; prose: 0.709
    "lam_sem": 0.7872,          # Table 5 cell; prose: 0.787
    "dem_share_p_ols": 0.385,
    "dem_share_p_slm": 0.170,
    "dem_share_p_sem": 0.374,
    "dem_share_coef_range": (0.201, 0.229),  # prose: coefficients "0.201 to 0.229"
    "ols_resid_morans_i": 0.389,
    "ols_resid_morans_z": 6.05,
    "pr2_slm": 0.708,
    "pr2_sem": 0.406,
    "n": 51,
}


def close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def main():
    # --- Independent re-derivation (no import of the analysis script) --------
    df = pd.read_parquet(CANON)
    agg = df.groupby("CounDist").agg(
        median_log_price=("target_log_price", "median"),
        dem_share=("dem_share", "mean"),
        mean_numfloors=("numfloors", "mean"),
        mean_yearbuilt=("yearbuilt", "mean"),
        mean_commfar=("commfar", "mean"),
    ).reset_index()

    gdf = gpd.read_file(COUNCIL_SHP)
    gdf["CounDist"] = pd.to_numeric(gdf["CounDist"], errors="coerce")
    agg["CounDist"] = pd.to_numeric(agg["CounDist"], errors="coerce")
    merged = gdf[["CounDist", "geometry"]].merge(agg, on="CounDist", how="inner")
    merged = merged.sort_values("CounDist").reset_index(drop=True)

    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)

    np.random.seed(SEED)
    w = KNN(coords, k=K)
    w.transform = "r"

    y = merged[["median_log_price"]].values
    X = merged[["dem_share", "mean_numfloors", "mean_yearbuilt", "mean_commfar"]].values
    x_names = ["dem_share", "mean_numfloors", "mean_yearbuilt", "mean_commfar"]

    ols = spreg.OLS(y, X, w=w, name_y="median_log_price", name_x=x_names,
                     name_w="knn5", spat_diag=True, moran=True)
    slm = spreg.ML_Lag(y, X, w=w, name_y="median_log_price", name_x=x_names, name_w="knn5")
    sem = spreg.ML_Error(y, X, w=w, name_y="median_log_price", name_x=x_names, name_w="knn5")

    # Fully independent OLS cross-check (numpy least squares, no spreg at all):
    # guards the coefficient<->label alignment that the original draft of this
    # analysis got wrong (see the extract() docstring in
    # src/analysis/spatial_regression_model.py).
    X_np = np.column_stack([np.ones(len(merged)), X])
    betas_np = np.linalg.lstsq(X_np, y.ravel(), rcond=None)[0]

    def beta_map(model):
        names = list(model.name_x)
        betas = model.betas.flatten().tolist()
        se = list(model.std_err)
        z_stat = getattr(model, "z_stat", None) or getattr(model, "t_stat", None)
        out = {}
        for i, lbl in enumerate(names):
            entry = {"coef": float(betas[i]), "se": float(se[i])}
            if z_stat is not None and i < len(z_stat):
                entry["z"] = float(z_stat[i][0])
                entry["p"] = float(z_stat[i][1])
            out[lbl] = entry
        return out

    ols_m, slm_m, sem_m = beta_map(ols), beta_map(slm), beta_map(sem)

    rec = {
        "n_districts": int(len(merged)),
        "weights_spec": f"k={K}_row_standardized_knn (independent re-fit; "
                        f"identical construction to compute_morans_i.py)",
        "seed": SEED,
        "ols_anchor": ols_m,
        "spatial_lag_model": {**slm_m, "pr2": float(slm.pr2)},
        "spatial_error_model": {**sem_m, "pr2": float(sem.pr2)},
        "ols_residual_morans_i": {
            "I": float(ols.moran_res[0]),
            "z": float(ols.moran_res[1]),
            "p": float(ols.moran_res[2]),
        },
        "numpy_ols_crosscheck": {
            "order": ["CONSTANT"] + x_names,
            "betas": [float(b) for b in betas_np],
        },
    }
    # The OLS anchor must match the numpy least-squares fit coefficient-for
    # coefficient (same design matrix, same order) — this is the regression
    # test for the mislabeling class of bug documented in the analysis script.
    rec["ols_matches_numpy_lstsq"] = bool(np.allclose(
        [ols_m[n]["coef"] for n in ["CONSTANT"] + x_names], betas_np, atol=1e-8))

    # --- Compare against the PIPELINE's own output ---------------------------
    pipeline = json.loads(PIPELINE_JSON.read_text()) if PIPELINE_JSON.exists() else None
    pipeline_agreement = None
    if pipeline is not None:
        pipeline_agreement = {}
        for spec, got in [("ols_anchor", ols_m), ("spatial_lag_model", slm_m),
                          ("spatial_error_model", sem_m)]:
            ok = []
            for name, cell in pipeline[spec]["coefficients"].items():
                mine = got.get(name)
                ok.append(mine is not None
                          and abs(mine["coef"] - cell["coef"]) < 1e-6
                          and abs(mine["se"] - cell["se"]) < 1e-6)
            pipeline_agreement[spec] = bool(all(ok))
        pipeline_agreement["ols_residual_morans_i"] = bool(np.allclose(
            [rec["ols_residual_morans_i"]["I"], rec["ols_residual_morans_i"]["z"]],
            [pipeline["ols_residual_morans_i"]["I"], pipeline["ols_residual_morans_i"]["z"]],
            atol=1e-6))
    rec["matches_pipeline_output"] = (
        pipeline_agreement if pipeline is not None else
        "results/spatial_regression.json not found — run generate_spatial_regression first")

    # --- Compare against the manuscript's reported values --------------------
    paper_checks = {
        "n_districts_eq_51": rec["n_districts"] == PAPER["n"],
        "rho_slm (paper 0.7086, prose 0.709)": close(slm_m["W_median_log_price"]["coef"], PAPER["rho_slm"], 5e-5),
        "slm_rho_p_below_0.001": slm_m["W_median_log_price"]["p"] < 0.001,
        "lambda_sem (paper 0.7872, prose 0.787)": close(sem_m["lambda"]["coef"], PAPER["lam_sem"], 5e-5),
        "sem_lambda_p_below_0.001": sem_m["lambda"]["p"] < 0.001,
        "dem_share_p_ols (paper 0.385)": close(ols_m["dem_share"]["p"], PAPER["dem_share_p_ols"], 5e-4),
        "dem_share_p_slm (paper 0.170)": close(slm_m["dem_share"]["p"], PAPER["dem_share_p_slm"], 5e-4),
        "dem_share_p_sem (paper 0.374)": close(sem_m["dem_share"]["p"], PAPER["dem_share_p_sem"], 5e-4),
        "dem_share_coefs_within_0.201_0.229": all(
            PAPER["dem_share_coef_range"][0] - 5e-4 <= m["dem_share"]["coef"] <= PAPER["dem_share_coef_range"][1] + 5e-4
            and m["dem_share"]["coef"] > 0 for m in (ols_m, slm_m, sem_m)),
        "ols_resid_morans_i (paper 0.389)": close(rec["ols_residual_morans_i"]["I"], PAPER["ols_resid_morans_i"], 5e-4),
        "ols_resid_morans_z (paper 6.05)": close(rec["ols_residual_morans_i"]["z"], PAPER["ols_resid_morans_z"], 5e-3),
        "ols_resid_morans_p_below_0.001": rec["ols_residual_morans_i"]["p"] < 0.001,
        "pr2_slm (paper 0.708)": close(float(slm.pr2), PAPER["pr2_slm"], 5e-4),
        "pr2_sem (paper 0.406)": close(float(sem.pr2), PAPER["pr2_sem"], 5e-4),
        "mean_commfar_significant_in_all_three": all(m["mean_commfar"]["p"] < 0.01 for m in (ols_m, slm_m, sem_m)),
    }
    rec["matches_paper"] = bool(all(paper_checks.values()))
    rec["paper_checks"] = paper_checks

    OUT_JSON.write_text(json.dumps({"section_4_5_spatial_regression": rec}, indent=2))
    print(f"WROTE {OUT_JSON.relative_to(ROOT)}")
    print(json.dumps(paper_checks, indent=2))
    print("agreement with results/spatial_regression.json:", rec["matches_pipeline_output"])
    print("ols_matches_numpy_lstsq:", rec["ols_matches_numpy_lstsq"])
    print("rho:", slm_m["W_median_log_price"])
    print("lambda:", sem_m["lambda"])
    print("dem_share p:", ols_m["dem_share"]["p"], slm_m["dem_share"]["p"], sem_m["dem_share"]["p"])
    print("residual Moran's I:", rec["ols_residual_morans_i"])


if __name__ == "__main__":
    main()


