"""
District-level spatial lag and spatial error regression, extending the
bivariate district-level analysis in Section 4.1 and the Moran's I evidence
in Section 4.2 with a formal spatial-econometric treatment -- exactly the
extension the paper's own Future Work section calls for.

Uses the IDENTICAL district-level aggregation, shapefile, join key, and KNN
weights construction as src/analysis/compute_morans_i.py (k=5,
row-standardized, official Council District polygon centroids), so results
are directly comparable to the Moran's I statistics already reported.

N=51 (one row per council district), median_log_price on dem_share, with a
small set of district-level structural controls aggregated from the
canonical dataset (mean numfloors, mean yearbuilt, mean commfar) so the
regression is not purely bivariate.

Two specifications:
  - Spatial Lag Model (SLM): y = rho * W*y + X*beta + epsilon
  - Spatial Error Model (SEM): y = X*beta + u,  u = lambda * W*u + epsilon
Both estimated by maximum likelihood via PySAL's spreg.

Output: results/spatial_regression.json, tables/table_spatial_regression.tex
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
OUT_JSON = ROOT / "results" / "spatial_regression.json"
OUT_TEX = ROOT / "tables" / "table_spatial_regression.tex"

SEED = 42
K = 5
np.random.seed(SEED)


def main():
    df = pd.read_parquet(CANON)
    dist_col = "CounDist"

    # Same aggregation as compute_morans_i.py, plus three district-level
    # structural controls so the model is not purely bivariate.
    agg = df.groupby(dist_col).agg(
        median_log_price=("target_log_price", "median"),
        dem_share=("dem_share", "mean"),
        mean_numfloors=("numfloors", "mean"),
        mean_yearbuilt=("yearbuilt", "mean"),
        mean_commfar=("commfar", "mean"),
    ).reset_index()
    n_districts = len(agg)

    gdf = gpd.read_file(COUNCIL_SHP)
    gdf["CounDist"] = pd.to_numeric(gdf["CounDist"], errors="coerce")
    agg["CounDist"] = pd.to_numeric(agg["CounDist"], errors="coerce")

    merged = gdf[["CounDist", "geometry"]].merge(agg, on="CounDist", how="inner")
    merged = merged.sort_values("CounDist").reset_index(drop=True)
    assert len(merged) == n_districts, f"district merge lost rows: {len(merged)} vs {n_districts}"

    centroid = merged.geometry.centroid.set_crs(merged.crs)
    coords = np.column_stack([centroid.x.values, centroid.y.values]).astype(float)

    np.random.seed(SEED)
    w = KNN(coords, k=K)
    w.transform = "r"

    y = merged[["median_log_price"]].values
    X = merged[["dem_share", "mean_numfloors", "mean_yearbuilt", "mean_commfar"]].values
    x_names = ["dem_share", "mean_numfloors", "mean_yearbuilt", "mean_commfar"]

    # --- OLS anchor (same X, no spatial term) for direct comparison ---
    ols = spreg.OLS(y, X, w=w, name_y="median_log_price", name_x=x_names,
                     name_w="knn5", spat_diag=True, moran=True)

    # --- Spatial Lag Model (ML estimation) ---
    slm = spreg.ML_Lag(y, X, w=w, name_y="median_log_price", name_x=x_names, name_w="knn5")

    # --- Spatial Error Model (ML estimation) ---
    sem = spreg.ML_Error(y, X, w=w, name_y="median_log_price", name_x=x_names, name_w="knn5")

    def extract(model):
        """model.name_x already includes 'CONSTANT' first and, for the
        spatial models, the spatial parameter last ('W_<depvar>' for
        ML_Lag's rho, 'lambda' for ML_Error) -- zip directly against
        betas/std_err/z_stat in that exact order. Do NOT prepend anything
        extra; the earlier version of this script did, which silently
        shifted every coefficient by one position (mean_commfar's real
        value ended up unlabeled and dem_share was mislabeled with the
        constant). Caught by cross-checking against spreg's own printed
        .summary text before trusting any of this for the paper -- see
        README.md ("Where the paper stands") / VERIFICATION_REPORT.md for the
        general pattern of catching this class of bug throughout the
        project.
        """
        names = model.name_x
        betas = model.betas.flatten().tolist()
        se = model.std_err.flatten().tolist() if hasattr(model, "std_err") else [None] * len(betas)
        z_stat = getattr(model, "z_stat", None) or getattr(model, "t_stat", None)
        out = {"pr2": float(model.pr2) if hasattr(model, "pr2") else None,
               "n": int(model.n), "k": int(model.k),
               "coefficients": {}}
        for i, lbl in enumerate(names):
            entry = {"coef": betas[i], "se": se[i]}
            if z_stat is not None and i < len(z_stat):
                entry["z"] = float(z_stat[i][0])
                entry["p"] = float(z_stat[i][1])
            out["coefficients"][lbl] = entry
        return out

    result = {
        "n_districts": n_districts,
        "weights_spec": f"k={K}_row_standardized_knn (identical construction to compute_morans_i.py)",
        "seed": SEED,
        "controls": x_names[1:],
        "ols_anchor": extract(ols),
        "spatial_lag_model": extract(slm),
        "spatial_error_model": extract(sem),
    }

    # Moran's I of OLS residuals (diagnostic already computed by spat_diag=True)
    try:
        result["ols_residual_morans_i"] = {
            "I": float(ols.moran_res[0]),
            "z": float(ols.moran_res[1]),
            "p": float(ols.moran_res[2]),
        }
    except Exception as e:
        result["ols_residual_morans_i_error"] = str(e)

    # Likelihood-ratio-style comparison via log-likelihood if available
    for label, model in [("ols", ols), ("slm", slm), ("sem", sem)]:
        if hasattr(model, "logll"):
            result.setdefault("log_likelihood", {})[label] = float(model.logll)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")

    # --- LaTeX table ---
    def coef_row(label, d, key):
        c = d["coefficients"].get(key)
        if c is None:
            return f"{label} & --- & --- \\\\"
        stars = ""
        return f"{label} & {c['coef']:.4f} & ({c['se']:.4f}) \\\\" if c['se'] is not None else f"{label} & {c['coef']:.4f} & --- \\\\"

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{District-level ($N=51$) spatial regression: median log price on dem\_share, with structural controls}",
        r"\label{tab:spatial}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & OLS (no spatial term) & Spatial Lag (ML) & Spatial Error (ML) \\",
        r"\midrule",
    ]
    for lbl, key in [("dem\\_share", "dem_share"), ("Mean numfloors", "mean_numfloors"),
                      ("Mean yearbuilt", "mean_yearbuilt"), ("Mean commfar", "mean_commfar")]:
        row = [lbl]
        for m in [result["ols_anchor"], result["spatial_lag_model"], result["spatial_error_model"]]:
            c = m["coefficients"].get(key)
            row.append(f"{c['coef']:.4f} ({c['se']:.4f})" if c and c['se'] is not None else "---")
        tex_lines.append(" & ".join(row) + r" \\")
    rho = result["spatial_lag_model"]["coefficients"].get("W_median_log_price")
    lam = result["spatial_error_model"]["coefficients"].get("lambda")
    tex_lines.append(r"\midrule")
    tex_lines.append(f"$\\rho$ (spatial lag) & --- & {rho['coef']:.4f} ({rho['se']:.4f}) & --- \\\\" if rho and rho['se'] else "")
    tex_lines.append(f"$\\lambda$ (spatial error) & --- & --- & {lam['coef']:.4f} ({lam['se']:.4f}) \\\\" if lam and lam['se'] else "")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\vspace{2pt}")
    tex_lines.append(r"\begin{minipage}{0.95\textwidth}")
    tex_lines.append(
        r"\footnotesize $N=51$ council districts. Standard errors in parentheses. "
        r"Weights: $k=5$ row-standardized KNN on official Council District polygon centroids, "
        r"identical construction to the Moran's I statistics in Section~\ref{sec:moran}. "
        r"See \texttt{results/spatial\_regression.json} for full output."
    )
    tex_lines.append(r"\end{minipage}")
    tex_lines.append(r"\end{table}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join([l for l in tex_lines if l]))
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in result.items() if k != "log_likelihood"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
