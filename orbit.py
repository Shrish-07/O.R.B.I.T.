# orbit.py
# NYC Property Price Predictor — Hardened & Full Version (with fixes)
# - Adds Streamlit exception hook
# - Removes deprecated set_option call
# - Fixes pandas to_numeric deprecation usage
# - Keeps safety/stability helpers from your hardened version
# - Expanded About/Docs & guidance text per user request (TODO placeholders for author/affiliation)

import os
import io
import time
import hashlib
import zipfile
import warnings
import traceback
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st
import cloudpickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Global exception hook for Streamlit
# ---------------------------
def _st_excepthook(exc_type, exc_value, exc_traceback):
    # Format trace
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    # Write to Streamlit app if possible
    try:
        if isinstance(exc_value, ConnectionError):
            st.warning("⚠️ Connection error occurred. If this persists, try a different browser or restart the app.")
        else:
            st.error("❌ Uncaught exception (see details below).")
            st.text(tb)
    except Exception:
        # Fallback to stderr
        print("Uncaught exception:", tb, flush=True)

# Register excepthook
import sys
sys.excepthook = _st_excepthook

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="NYC Property Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NOTE: `deprecation.showPyplotGlobalUse` is no longer supported in newer Streamlit.
# we purposely avoid calling st.set_option to silence that deprecation.

# ---------------------------
# SECURITY / STABILITY SETTINGS
# ---------------------------
SAFE_MODE_DEFAULT = True                 # Blocks untrusted pickles by default
MAX_UPLOAD_MB = 200                      # Max upload size for CSV/ZIP/PKL
MAX_ZIP_MEMBERS = 20                     # Limit files inside ZIP
MAX_ZIP_UNCOMPRESSED_MB = 1500           # Uncompressed size cap to avoid zip-bombs
MAX_EDA_SAMPLE = 50_000                  # Cap rows for heavy EDA charts
MAX_SHAP_BACKGROUND = 1_000              # SHAP background sample cap
MAX_BATCH_PREVIEW_ROWS = 50              # Preview rows for batch results
CURRENT_YEAR = 2025

DERIVED = {"AGE", "IS_NEW", "HAS_LAND"}
TARGET_COLS = {"SALE PRICE", "SALE_PRICE", "SALEPRICE"}
LEAKY_FEATURES = {"PRICE PER SQFT", "PRICE_PER_SQFT", "PRICE_PER_SQUARE_FOOT"}

# Make warnings visible but avoid flooding the UI with repeated messages
warnings.simplefilter("once")

# Suppress noisy scikit-learn cross-version messages (but still show once)
try:
    from sklearn.utils._estimator_html_repr import _write_label_html  # noqa: F401
except Exception:
    pass
warnings.filterwarnings(
    "once",
    message="Trying to unpickle estimator .* from version .* when using version .*",
    category=UserWarning,
    module="sklearn",
)

# ===========================
# Utilities
# ===========================
def usd(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def _bytes_to_mb(n: int) -> float:
    try:
        return float(n) / (1024 * 1024)
    except Exception:
        return 0.0

def _hash_filelike(f) -> str:
    """Hash a file-like without consuming its pointer for caching keys."""
    try:
        pos = None
        try:
            pos = f.tell()
        except Exception:
            pos = None
        try:
            f.seek(0)
        except Exception:
            pass
        h = hashlib.sha256()
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "ignore")
            h.update(chunk)
        digest = h.hexdigest()
        try:
            if pos is not None:
                f.seek(pos)
        except Exception:
            pass
        return digest
    except Exception:
        try:
            if hasattr(f, "seek") and pos is not None:
                f.seek(pos)
        except Exception:
            pass
        return str(time.time())

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace(" ", "_").upper() for c in df.columns]
    return df

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # YEAR BUILT -> AGE / IS_NEW
    ybuilt_candidates = [c for c in df.columns if c in {"YEAR_BUILT", "YEARBUILT", "YEAR_BUILT"}]
    land_candidates = [c for c in df.columns if c in {"LAND_SQUARE_FEET", "LANDSQUAREFEET", "LAND_SQUARE_FEET"}]
    if ybuilt_candidates:
        ycol = ybuilt_candidates[0]
        df["AGE"] = CURRENT_YEAR - pd.to_numeric(df[ycol], errors="coerce")
        df["IS_NEW"] = (df["AGE"] < 5).astype("Int64")
    if land_candidates:
        lcol = land_candidates[0]
        df["HAS_LAND"] = (pd.to_numeric(df[lcol], errors="coerce") > 0).astype("Int64")
    return df

def _coerce_numeric_inplace(df: pd.DataFrame) -> None:
    # Safe numeric conversion without deprecated errors='ignore'
    for c in df.columns:
        try:
            # Attempt conversion; if fails, keep original
            df[c] = pd.to_numeric(df[c])
        except Exception:
            # leave column unchanged (likely non-numeric)
            pass

def _find_target(df: pd.DataFrame) -> Optional[str]:
    for cand in df.columns:
        if cand.replace(" ", "_").upper() in {t.replace(" ", "_").upper() for t in TARGET_COLS}:
            return cand
    return None

def _extract_model_features(model) -> Optional[List[str]]:
    """Try several places to get feature names the model expects."""
    if model is None:
        return None
    # Direct
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # Pipelines / named_steps
    try:
        if hasattr(model, "named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                if hasattr(step, "feature_names_in_"):
                    try:
                        return list(step.feature_names_in_)
                    except Exception:
                        pass
    except Exception:
        pass
    # Underlying estimator
    try:
        if hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
            return list(model.estimator.feature_names_in_)
    except Exception:
        pass
    return None

def _norm_name(s: str) -> str:
    return str(s).replace(" ", "_").upper()

def _build_feature_bridge(model, df_cols: List[str]):
    """
    Map between UI-normalized feature names and model's original feature names.
    Returns:
      ui_feats, model_ord, ui_to_orig, missing
    """
    raw = _extract_model_features(model) or []
    pairs = [(_norm_name(f), f) for f in raw]
    df_cols_set = set(df_cols)
    ui_feats   = [ui for ui, orig in pairs if ui in df_cols_set]
    model_ord  = [orig for ui, orig in pairs if ui in df_cols_set]
    ui_to_orig = {ui: orig for ui, orig in pairs if ui in df_cols_set}
    missing    = [orig for ui, orig in pairs if ui not in df_cols_set]
    return ui_feats, model_ord, ui_to_orig, missing

def _align_X(df_like: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_list if c not in df_like.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df_like[feature_list]

def _compute_feature_stats(df_eng: pd.DataFrame, features: List[str]) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    med = df_eng[features].median(numeric_only=True)
    q = df_eng[features].quantile([0.01, 0.5, 0.99]).T
    pctl = {}
    for f in features:
        if f in q.index:
            pctl[f] = {"p1": float(q.loc[f, 0.01]), "p50": float(q.loc[f, 0.5]), "p99": float(q.loc[f, 0.99])}
        else:
            m = float(med.get(f, 0.0))
            pctl[f] = {"p1": m * 0.5 if m != 0 else 0.0, "p50": m, "p99": m * 1.5 if m != 0 else 1.0}
    return med, pctl

def _compute_impute_maps(df_eng: pd.DataFrame, features: List[str]) -> Tuple[pd.Series, Dict[str, object]]:
    """Return numeric medians and categorical modes for imputation."""
    med = df_eng[features].median(numeric_only=True)
    modes: Dict[str, object] = {}
    for f in features:
        s = df_eng[f] if f in df_eng.columns else pd.Series(dtype=float)
        if not pd.api.types.is_numeric_dtype(s):
            try:
                m = s.mode(dropna=True)
                modes[f] = (None if m.empty else m.iloc[0])
            except Exception:
                modes[f] = None
    return med, modes

def _get_top20_corr_features(df_eng: pd.DataFrame) -> List[str]:
    """Pick top20 numeric features by absolute correlation with target (drop leakage)."""
    tcol = _find_target(df_eng)
    num = df_eng.select_dtypes(include=[np.number]).copy()
    # remove target and leakage candidates if present
    if tcol and tcol in num.columns:
        y = num[tcol]
        X = num.drop(columns=[tcol], errors="ignore")
        for leak in LEAKY_FEATURES:
            X = X.drop(columns=[leak], errors="ignore")
        tmp = pd.concat([X, y], axis=1).dropna()
        if tmp.shape[0] >= 10:
            Xc = tmp.drop(columns=[tcol])
            yc = tmp[tcol]
            try:
                corr = Xc.corrwith(yc).abs().sort_values(ascending=False)
            except Exception:
                corr = X.corrwith(y.fillna(y.median())).abs().sort_values(ascending=False)
        else:
            corr = X.corrwith(y.fillna(y.median())).abs().sort_values(ascending=False)
        feats = corr.head(20).index.tolist()
    else:
        X = num.drop(columns=[c for c in num.columns if c in LEAKY_FEATURES], errors="ignore")
        feats = list(X.columns[:20])
    return feats

def _get_model_feature_importance(model, model_feature_order: List[str]) -> Optional[Dict[str, float]]:
    """Return importance scores per original model feature name if available."""
    if model is None:
        return None
    # Direct (tree-based)
    try:
        if hasattr(model, "feature_importances_"):
            vals = getattr(model, "feature_importances_", None)
            if vals is not None and len(vals) == len(model_feature_order):
                return {name: float(imp) for name, imp in zip(model_feature_order, vals)}
    except Exception:
        pass
    # Pipelines
    est = None
    try:
        if hasattr(model, "named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                    est = step
                    break
        elif hasattr(model, "estimator"):
            est = model.estimator
    except Exception:
        est = None
    if est is not None:
        try:
            if hasattr(est, "feature_importances_"):
                vals = getattr(est, "feature_importances_", None)
                if vals is not None and len(vals) == len(model_feature_order):
                    return {name: float(imp) for name, imp in zip(model_feature_order, vals)}
        except Exception:
            pass
        try:
            if hasattr(est, "coef_"):
                coef = getattr(est, "coef_", None)
                if coef is not None:
                    coef = np.ravel(coef)
                    if len(coef) == len(model_feature_order):
                        return {name: float(abs(c)) for name, c in zip(model_feature_order, coef)}
        except Exception:
            pass
    # Linear direct
    try:
        if hasattr(model, "coef_"):
            coef = getattr(model, "coef_", None)
            if coef is not None:
                coef = np.ravel(coef)
                if len(coef) == len(model_feature_order):
                    return {name: float(abs(c)) for name, c in zip(model_feature_order, coef)}
    except Exception:
        pass
    return None

def _predict_fn_factory(model, feature_order_ui: List[str], ui_to_model_map: Dict[str, str]) -> Callable[[pd.DataFrame], np.ndarray]:
    """Predict accepting UI-normalized columns; rename & order to model's expected columns on the fly."""
    feature_order_model = [ui_to_model_map.get(f, f) for f in feature_order_ui]
    def _predict_any(X_like) -> np.ndarray:
        if isinstance(X_like, pd.DataFrame):
            X_ui = X_like.copy()
        else:
            X_ui = pd.DataFrame(X_like, columns=feature_order_ui)
        X_ui = X_ui.reindex(columns=feature_order_ui, fill_value=np.nan)
        X_model = X_ui.rename(columns=ui_to_model_map)[feature_order_model]
        return model.predict(X_model)
    return _predict_any

# ===========================
# Safe file loading helpers
# ===========================
def _safe_zip_checks(z: zipfile.ZipFile) -> None:
    """Raise if suspicious zip (too many members / too large uncompressed / traversal)."""
    infos = z.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ValueError(f"ZIP has too many files ({len(infos)} > {MAX_ZIP_MEMBERS}).")
    total_uncompressed = sum(getattr(i, "file_size", 0) for i in infos)
    if _bytes_to_mb(total_uncompressed) > MAX_ZIP_UNCOMPRESSED_MB:
        raise ValueError(f"ZIP uncompressed size too large ({_bytes_to_mb(total_uncompressed):.1f} MB).")
    # path traversal check
    for i in infos:
        n = i.filename
        if ".." in n.replace("\\", "/"):
            raise ValueError("ZIP path traversal detected.")

@st.cache_data(show_spinner=False)
def _read_csv_path_cached(p: str) -> pd.DataFrame:
    """Cached CSV reader for filesystem paths."""
    return pd.read_csv(p, low_memory=False)

def _try_read_csv_or_zip(path_or_file) -> pd.DataFrame:
    """
    Prefer CSV; also supports the first CSV inside a ZIP.
    Accepts file-like object or filesystem path string. Includes safety caps.
    """
    # file-like
    if hasattr(path_or_file, "read"):
        f = path_or_file
        # Size guard for uploads (Streamlit UploadedFile exposes .size)
        size = getattr(f, "size", None)
        if size is not None and _bytes_to_mb(size) > MAX_UPLOAD_MB:
            raise ValueError(f"Uploaded file too large ({_bytes_to_mb(size):.1f} MB > {MAX_UPLOAD_MB} MB limit).")
        try:
            f.seek(0)
        except Exception:
            pass
        # Try CSV directly
        try:
            return pd.read_csv(f, low_memory=False, on_bad_lines="skip")
        except Exception:
            pass
        # Try ZIP safely
        try:
            try:
                f.seek(0)
            except Exception:
                pass
            with zipfile.ZipFile(f) as z:
                _safe_zip_checks(z)
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise ValueError("No CSV found inside ZIP.")
                # Read first CSV
                with z.open(csvs[0]) as zf:
                    return pd.read_csv(zf, low_memory=False, on_bad_lines="skip")
        except zipfile.BadZipFile:
            raise
    # path string
    p = str(path_or_file)
    if p.lower().endswith(".csv"):
        # If local file is massive, still allow but cache and warn in UI where called
        return _read_csv_path_cached(p)
    if p.lower().endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            _safe_zip_checks(z)
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("No CSV found inside ZIP.")
            with z.open(csvs[0]) as zf:
                return pd.read_csv(zf, low_memory=False, on_bad_lines="skip")
    # default: try as CSV
    return _read_csv_path_cached(p)

def _safe_load_pickle(model_source, allow_untrusted: bool) -> Optional[object]:
    """
    Secure-ish pickle loading:
      - Blocks by default for uploads (pickles can execute arbitrary code).
      - Allows local trusted files unless SAFE MODE forbids.
    """
    if model_source is None:
        return None

    # Uploaded file?
    is_uploaded = hasattr(model_source, "read") and hasattr(model_source, "name")
    if is_uploaded and not allow_untrusted:
        raise PermissionError(
            "Blocked loading uploaded pickle in Safe Mode. "
            "Enable 'I trust this model file' in sidebar to proceed."
        )

    # Size guard
    try:
        if is_uploaded and getattr(model_source, "size", 0) and _bytes_to_mb(model_source.size) > MAX_UPLOAD_MB:
            raise ValueError(f"Uploaded model too large ({_bytes_to_mb(model_source.size):.1f} MB > {MAX_UPLOAD_MB} MB limit).")
        if isinstance(model_source, str) and os.path.exists(model_source):
            sz = os.path.getsize(model_source)
            if _bytes_to_mb(sz) > MAX_UPLOAD_MB * 2:
                st.sidebar.warning(f"Model file is large ({_bytes_to_mb(sz):.1f} MB); loading may take time.")
    except Exception:
        pass

    # Load
    if isinstance(model_source, str):
        with open(model_source, "rb") as f:
            return cloudpickle.load(f)
    else:
        return cloudpickle.load(model_source)

# ===========================
# Auto-locate local files (CSV first)
# ===========================
MODEL_CANDIDATES = ["orbit.pkl", "model.pkl", "models/orbit.pkl"]
DATA_CANDIDATES = [
    "NYC Data Cleaned.csv", "data/NYC Data Cleaned.csv",
    "nyc.csv", "data/nyc.csv",
    "NYC Data.zip", "data/NYC Data.zip", "nyc_data.zip"
]

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return None

# ===========================
# Sidebar: status + optional overrides
# ===========================
st.sidebar.header("Artifacts & Settings")

# Safe Mode + Trust toggle
if "SAFE_MODE" not in st.session_state:
    st.session_state.SAFE_MODE = SAFE_MODE_DEFAULT

st.sidebar.checkbox("Safe Mode (recommended)", value=st.session_state.SAFE_MODE, key="SAFE_MODE")
allow_untrusted_pkl = st.sidebar.checkbox("I trust this model file (allow uploaded .pkl)", value=False)

auto_model_path = _first_existing(MODEL_CANDIDATES)
auto_data_path = _first_existing(DATA_CANDIDATES)

st.sidebar.write("**Auto-detected files**")
st.sidebar.write(f"Model: `{auto_model_path}`" if auto_model_path else "Model: _not found_")
st.sidebar.write(f"Training data: `{auto_data_path}`" if auto_data_path else "Training data: _not found_")

with st.sidebar.expander("Optional: override files"):
    model_file = st.file_uploader("Upload model (.pkl)", type=["pkl"], key="mdl_up", help="Uploading pickles is dangerous. Safe Mode blocks this unless you explicitly allow it.")
    data_file = st.file_uploader("Upload training data (CSV or ZIP)", type=["csv", "zip"], key="data_up")
st.sidebar.caption("Tip: This app prefers a cleaned CSV. ZIPs with a single CSV are also supported.")

# Resolve sources (override if uploads provided)
model_source = model_file if model_file is not None else auto_model_path
data_source = data_file if data_file is not None else auto_data_path

# ===========================
# Load model
# ===========================
model = None
if model_source is not None:
    try:
        model = _safe_load_pickle(model_source, allow_untrusted=(allow_untrusted_pkl and not st.session_state.SAFE_MODE))
        st.sidebar.success("Model loaded.")
    except PermissionError as e:
        st.sidebar.error(str(e))
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        # show stack in app for debugging
        try:
            st.text(traceback.format_exc())
        except Exception:
            pass
else:
    st.sidebar.warning("No model available. Place `orbit.pkl` in the app folder or upload one.")

# ===========================
# Load data (CSV-first)
# ===========================
df_train_raw = None
if data_source is not None:
    try:
        # Cache key for uploads uses content hash to avoid massive copies
        if hasattr(data_source, "read"):
            digest = _hash_filelike(data_source)
            with st.spinner("Reading uploaded data…"):
                df_train_raw = _try_read_csv_or_zip(data_source)
        else:
            with st.spinner("Reading data…"):
                df_train_raw = _try_read_csv_or_zip(data_source)

        # Warn if huge
        if df_train_raw is not None and len(df_train_raw) > MAX_EDA_SAMPLE:
            st.sidebar.info(f"Large dataset detected ({len(df_train_raw):,} rows). Charts will sample up to {MAX_EDA_SAMPLE:,} rows to stay responsive.")
        df_train_raw = _normalize_columns(df_train_raw)
        st.sidebar.success("Training data loaded.")
    except Exception as e:
        st.sidebar.error(f"Data load failed: {e}")
        try:
            st.text(traceback.format_exc())
        except Exception:
            pass
else:
    st.sidebar.warning("No training data available. Put `NYC Data Cleaned.csv` (or a ZIP with a CSV) next to the app or upload one.")

# Precompute engineered + numeric view for internal use
df_train_eng = None
if df_train_raw is not None:
    try:
        df_train_eng = _engineer_features(df_train_raw.copy())
        _coerce_numeric_inplace(df_train_eng)
    except Exception as e:
        st.sidebar.error(f"Feature engineering failed: {e}")
        try:
            st.text(traceback.format_exc())
        except Exception:
            pass
        df_train_eng = None

# ===========================
# Choose feature set (model-driven + keep only predictive)
# ===========================
features: List[str] = []                 # UI-normalized feature names
medians: Optional[pd.Series] = None      # numeric impute map
modes: Dict[str, object] = {}            # categorical impute map
pctl: Dict[str, Dict[str, float]] = {}
ui_to_model: Dict[str, str] = {}         # UI -> original model name
model_feature_order: List[str] = []      # original names aligned to features

if df_train_eng is not None:
    if model is not None:
        try:
            ui_feats, model_ord, ui_to_orig, missing = _build_feature_bridge(model, list(df_train_eng.columns))

            # Try to keep ONLY predictive features from the model (importance/coef > 0)
            imp_map = _get_model_feature_importance(model, model_ord)
            if imp_map:
                # Map importances (original) -> UI names
                imp_ui = { _norm_name(k): v for k, v in imp_map.items() }
                # Filter ui_feats to those with positive importance
                predictive_ui = [f for f in ui_feats if imp_ui.get(f, 0.0) > 0.0]
                if len(predictive_ui) == 0:
                    st.sidebar.warning("Model importances are zero/empty; falling back to correlation-based features.")
                    feats_corr = _get_top20_corr_features(df_train_eng)
                    predictive_ui = [f for f in feats_corr if f in ui_feats]
                features = predictive_ui
                model_feature_order = [ui_to_orig[f] for f in features]
                ui_to_model = {f: ui_to_orig[f] for f in features}
                if missing:
                    st.sidebar.info(f"Using {len(features)} predictive model features. {len(missing)} model features not found after normalization.")
            else:
                # No importances available -> correlation-based selection, but keep only those the model accepts
                feats_corr = _get_top20_corr_features(df_train_eng)
                features = [f for f in feats_corr if f in ui_feats]
                if not features:
                    features = ui_feats[:20]  # cap for UI sanity
                model_feature_order = [ui_to_orig[f] for f in features]
                ui_to_model = {f: ui_to_orig[f] for f in features}

            # Stats + imputers
            medians, pctl = _compute_feature_stats(df_train_eng, features) if features else (None, {})
            _, modes = _compute_impute_maps(df_train_eng, features)

            if not features:
                st.sidebar.error("Model exposes feature names, but none matched & scored after normalization.")
        except Exception as e:
            st.sidebar.error(f"Feature selection failed: {e}")
            try:
                st.text(traceback.format_exc())
            except Exception:
                pass
            features = []
            medians, modes, pctl = None, {}, {}
            ui_to_model, model_feature_order = {}, []
    else:
        try:
            # No model: use correlation-only selection from data
            features = _get_top20_corr_features(df_train_eng)
            medians, pctl = _compute_feature_stats(df_train_eng, features)
            _, modes = _compute_impute_maps(df_train_eng, features)
            ui_to_model = {f: f for f in features}
            model_feature_order = features
        except Exception as e:
            st.sidebar.error(f"Automatic feature selection failed: {e}")
            try:
                st.text(traceback.format_exc())
            except Exception:
                pass
            features = []
            medians, modes, pctl = None, {}, {}
else:
    st.sidebar.info("Feature statistics will appear after data loads.")

# Prediction function aligned to model (with UI->model renaming)
predict_any = None
if model is not None and features:
    try:
        predict_any = _predict_fn_factory(model, features, ui_to_model)
    except Exception as e:
        st.sidebar.error(f"Prediction wrapper failed: {e}")
        try:
            st.text(traceback.format_exc())
        except Exception:
            pass
        predict_any = None

# ===========================
# Tabs
# ===========================
tabs = st.tabs(["Research Project Timeline", "Predictions", "Explainability & SHAP", "Data Explorer & Model Info"])

# --- Tab 0: Project Timeline (expanded About)
with tabs[0]:
    st.subheader("About This Project")
    st.markdown(
        """
**Project Title:** NYC Property Price Prediction Using Ensemble Learning  

**Author:** Shrish Mudumby Venugopal, Sourish Mudumby Venugopal & Raghav Ganesh

**Institution:** Inspirit AI

**Date:** August 2025  

This research uses a stacked ensemble model (XGBoost + CatBoost + LightGBM → Bayesian Ridge).
We engineer features (`AGE`, `IS_NEW`, `HAS_LAND`), select a model-aligned feature space (or top-20 by correlation),
standardize/prepare inputs, and present a full research-facing Streamlit app with explainability, EDA, PDP and confidence intervals.

### Overview
- **Models**: Heterogeneous ensemble: gradient-boosted trees + linear meta-learner (Bayesian Ridge).  
- **Training Data**: NYC property transactions (cleaned), with typical columns such as:  
  `GROSS_SQUARE_FEET`, `LAND_SQUARE_FEET`, `YEAR_BUILT`, `RESIDENTIAL_UNITS`, `COMMERCIAL_UNITS`, (and categorical fields like `BOROUGH`, `NEIGHBORHOOD`, `BUILDING_CLASS_CATEGORY`).  
- **Engineered Features**:  
  - `AGE = CURRENT_YEAR - YEAR_BUILT`  
  - `IS_NEW = 1{AGE < 5}`  
  - `HAS_LAND = 1{LAND_SQUARE_FEET > 0}`  
- **Leakage Controls**: Filters and heuristics to exclude columns like `"PRICE_PER_SQFT"` and anything trivially derived from the target.  
- **Explainability**: Local + global SHAP, quick PDPs for top features.  
- **Confidence Intervals**: Empirical (residual-quantile) or Gaussian around point predictions.

> **Note:** The active feature set used by the model in this app is **auto-detected** from your loaded model and/or the training dataset.  
> See **Tab 3 — Data Explorer & Model Info** for the exact list currently in effect.

### Reproducibility & Environment
- **Serialization**: Pickles are allowed for local files; uploaded pickles are blocked unless you explicitly trust them (Safe Mode).  
- **Environment**: Use pinned versions from your `requirements.txt` to avoid cross-version pickle issues (e.g., scikit-learn 1.6.1).  
- **Deployment**: Streamlit app with guard rails for large files, zip-bombs, and long-running charts.

### Credits & Meta
- **Authors**: **Shrish Mudumby Venugopal, Sourish Mudumby Venugopal & Raghav Ganesh**  
- **Affiliation**: **Inspirit AI**  
- **Contact / Repo**: **Shrishvenugopal11@gmail.com**
"""
    )
    st.info("Edit the Author/Affiliation/Contact items above to personalize the project page.")

# --- Tab 3: Data Explorer & Model Info
with tabs[3]:
    st.subheader("Data Explorer & Model Info")
    if df_train_raw is None:
        st.info("Place your training data file next to the app or upload it in the sidebar to explore it.")
        st.markdown(
            """
**Expected columns (examples):**

- **Core numeric**: `GROSS_SQUARE_FEET`, `LAND_SQUARE_FEET`, `YEAR_BUILT`, `RESIDENTIAL_UNITS`, `COMMERCIAL_UNITS`.  
- **Categorical / text**: `BOROUGH`, `NEIGHBORHOOD`, `BUILDING_CLASS_CATEGORY`.  
- **Target (if present)**: one of `SALE PRICE`, `SALE_PRICE`, `SALEPRICE`.

> Columns are normalized by replacing spaces with underscores and uppercasing (e.g., `Year Built` → `YEAR_BUILT`).
> Engineered features include `AGE`, `IS_NEW`, `HAS_LAND`.
            """
        )
    else:
        try:
            st.markdown("#### Data snapshot (raw)")
            st.dataframe(df_train_raw.head(30), use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to show data snapshot: {e}")

        try:
            st.markdown("#### Dataset summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows", f"{len(df_train_raw):,}")
            with c2:
                st.metric("Columns", f"{len(df_train_raw.columns):,}")
            with c3:
                tcol = _find_target(df_train_raw)
                st.metric("Detected target", tcol if tcol else "None")
        except Exception as e:
            st.warning(f"Basic dataset summary unavailable: {e}")

        # Missingness (raw)
        try:
            st.markdown("#### Missingness (top columns)")
            miss = (df_train_raw.isna().mean() * 100).sort_values(ascending=False)
            if not miss[miss > 0].empty:
                df_missing = (
                    miss[miss > 0]
                    .head(30)
                    .reset_index()
                    .rename(columns={"index": "column", 0: "missing_pct"})
                )
                fig = px.bar(df_missing, x="column", y="missing_pct", title="Top Missing Columns")
                fig.update_layout(xaxis_tickangle=-45, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No significant missing values detected.")
        except Exception as e:
            st.warning(f"Missingness plot failed: {e}")

        # Correlations with target (engineered numeric)
        try:
            if df_train_eng is not None:
                tcol_eng = _find_target(df_train_eng)
                num = df_train_eng.select_dtypes(include=[np.number])
                if tcol_eng and tcol_eng in num.columns:
                    corrs = num.corr(numeric_only=True)[tcol_eng].sort_values(ascending=False)
                    st.markdown("#### Top correlations with target")
                    st.dataframe(corrs.head(20).to_frame("corr"), use_container_width=True)
                else:
                    st.info("Target column not numeric or not found; skipping correlation display.")
        except Exception as e:
            st.warning(f"Correlation section failed: {e}")

        # Distributions for active features (engineered)
        try:
            if df_train_eng is not None and features:
                st.markdown("#### Distributions: Active features")
                view = df_train_eng
                if len(view) > MAX_EDA_SAMPLE:
                    view = view.sample(MAX_EDA_SAMPLE, random_state=42)
                    st.caption(f"Showing histograms from a sample of {len(view):,} rows for responsiveness.")
                cols = st.columns(3)
                for i, feat in enumerate(features[:9]):
                    with cols[i % 3]:
                        try:
                            fig = px.histogram(view, x=feat, nbins=50, title=feat)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.write(f"(Skipped histogram for {feat})")
        except Exception as e:
            st.warning(f"Distribution plots failed: {e}")

        # Model info
        st.markdown("---")
        st.markdown("### Model info")
        if model is None:
            st.info("No model loaded.")
        else:
            try:
                st.write("Model type:", type(model).__name__)
            except Exception:
                st.write("Model type: (unavailable)")
            try:
                embedded_feats = _extract_model_features(model)
                st.write("Model feature list available:", "Yes" if embedded_feats else "No")
            except Exception:
                st.write("Model feature list available: Unknown")
            if features:
                st.write(f"Active predictive feature set used here ({len(features)}):")
                st.code(", ".join(features))
            st.caption("Tip: For batch scoring, include as many of these active features as possible in your CSV for best results.")

# --- Tab 1: Predictions
with tabs[1]:
    st.subheader("Predictions (Single & Batch)")
    st.markdown(
        """
**How to use this tab**

- **Single Property**: Enter feature values manually (defaults use training medians/modes).  
- **Batch (CSV upload)**: Upload a CSV; missing active features will be imputed (numeric → median, categorical → mode).  
- **Active features** are shown in **Tab 3 → Model info**; include as many as possible for best accuracy.  
- **Target column** (if present) should be one of: `SALE PRICE`, `SALE_PRICE`, `SALEPRICE` (case/space-insensitive).  
- **Confidence intervals** are computed from training residuals (empirical quantiles) or via Gaussian approximation.  

_Examples of commonly useful columns in uploads:_  
`GROSS_SQUARE_FEET`, `LAND_SQUARE_FEET`, `YEAR_BUILT`, `RESIDENTIAL_UNITS`, `COMMERCIAL_UNITS`, `BOROUGH`, `NEIGHBORHOOD`, `BUILDING_CLASS_CATEGORY`
        """
    )
    if model is None or df_train_eng is None or not features or predict_any is None or medians is None:
        st.warning("Load model & training data to enable predictions (auto-detected from disk).")
    else:
        mode = st.radio("Mode", ["Single Property", "Batch (CSV upload)"], horizontal=True)

        if mode == "Single Property":
            st.markdown("Fill property attributes. Defaults are medians/modes from the engineered training data.")
            cols = st.columns(3)
            inputs = {}
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    series = df_train_eng[feat] if feat in df_train_eng.columns else pd.Series(dtype=float)
                    if pd.api.types.is_numeric_dtype(series):
                        # Use precomputed percentiles when available
                        p = pctl.get(
                            feat,
                            {
                                "p1": float(series.quantile(0.01)) if not series.empty else 0.0,
                                "p50": float(series.median()) if not series.empty else 0.0,
                                "p99": float(series.quantile(0.99)) if not series.empty else 1.0,
                            },
                        )
                        # Step safeguards
                        span = max(0.0, p["p99"] - p["p1"])
                        step = max(1.0, span / 100) if span > 0 else 1.0
                        # Boundaries sanity
                        lo = p["p1"] if np.isfinite(p["p1"]) else 0.0
                        hi = p["p99"] if np.isfinite(p["p99"]) else (p["p50"] + 1.0)
                        if hi <= lo:
                            hi = lo + 1.0
                        val = p["p50"]
                        val = min(max(val, lo), hi)
                        inputs[feat] = st.number_input(
                            feat,
                            value=float(val),
                            min_value=float(lo),
                            max_value=float(hi),
                            step=step,
                            format="%.6f" if step < 1 else "%.4f",
                            help=f"approx p1={p['p1']:.2f}, p50={p['p50']:.2f}, p99={p['p99']:.2f}",
                        )
                    else:
                        # categorical/text: default to mode if available
                        default_text = ""
                        if not series.empty:
                            try:
                                m = series.mode(dropna=True)
                                if not m.empty:
                                    default_text = str(m.iloc[0])
                            except Exception:
                                default_text = ""
                        inputs[feat] = st.text_input(feat, value=default_text, max_chars=200)

            st.markdown("")  # spacing
            ci_method = st.selectbox("Confidence interval method", ["Empirical residual quantiles", "Gaussian (mean ± z·σ)"])
            conf_level = st.slider("Confidence level (%)", 50, 99, 90, 1)
            show_resid_hist = st.checkbox("Show residual distribution used for CI", value=True)

            if st.button("Predict & Show CI", type="primary"):
                try:
                    # Build UI-normalized row -> normalize/engineer -> impute -> predict
                    df_one_ui = pd.DataFrame([inputs])
                    df_one_norm = _normalize_columns(df_one_ui)
                    df_one_eng = _engineer_features(df_one_norm)
                    X_ui = df_one_eng.reindex(columns=features, fill_value=np.nan)
                    X_ui = X_ui.fillna(medians.to_dict()).fillna(modes)
                    pred = float(predict_any(X_ui)[0])

                    # Residuals on training data
                    residuals = None
                    target_col = _find_target(df_train_eng)
                    if target_col:
                        df_for_pred = df_train_eng[features].copy()
                        df_for_pred = df_for_pred.fillna(medians.to_dict()).fillna(modes)
                        preds_train_np = predict_any(df_for_pred)
                        preds_train = pd.Series(preds_train_np, index=df_for_pred.index, name="pred")
                        y_true = pd.to_numeric(df_train_eng[target_col], errors="coerce")
                        mask = y_true.notna() & preds_train.notna()
                        if mask.sum() > 0:
                            residuals = (y_true[mask] - preds_train[mask])

                    if residuals is None or residuals.shape[0] < 20:
                        # Fallback ±20%
                        fallback_width = 0.20 * pred if pred != 0 else 10000.0
                        lower, upper = pred - fallback_width, pred + fallback_width
                        st.warning("No reliable residual distribution available from training data. Falling back to a rough ±20% interval.")
                        st.success(f"Prediction: **{usd(pred)}**")
                        st.info(f"{conf_level}% interval (fallback): {usd(lower)} — {usd(upper)}")
                    else:
                        alpha = (100 - conf_level) / 100.0
                        if ci_method == "Empirical residual quantiles":
                            lo_q = residuals.quantile(alpha / 2.0)
                            hi_q = residuals.quantile(1 - alpha / 2.0)
                            lower = pred + lo_q
                            upper = pred + hi_q
                        else:
                            mu = residuals.mean()
                            sigma = residuals.std(ddof=1)
                            z = norm.ppf(1 - alpha / 2.0)
                            lower = pred + mu - z * sigma
                            upper = pred + mu + z * sigma

                        st.success(f"Prediction: **{usd(pred)}**")
                        st.info(f"{conf_level}% confidence interval: **{usd(lower)} — {usd(upper)}**")

                        if show_resid_hist:
                            try:
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(x=residuals, nbinsx=80, name="Residuals"))
                                fig.update_layout(title="Training residual distribution (Actual - Predicted)", xaxis_title="Residual", yaxis_title="Count", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                st.caption("(Residual histogram unavailable)")

                    # Percentile rank vs. training sales
                    try:
                        target_col = _find_target(df_train_eng)
                        if target_col:
                            train_target = pd.to_numeric(df_train_eng[target_col], errors="coerce").dropna()
                            if not train_target.empty:
                                pct_rank = float((train_target < pred).mean() * 100.0)
                                st.metric("Predicted price percentile vs training sales", f"{pct_rank:.1f}th percentile")
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Prediction/CIs failed: {e}")
                    try:
                        st.text(traceback.format_exc())
                    except Exception:
                        pass

        else:
            st.markdown(
                """
**Batch Scoring Instructions**

1. Upload a CSV file with columns named similarly to your training data (spaces → underscores, uppercased automatically).  
2. Include as many **active features** (Tab 3) as possible.  
3. If a recognized **target** exists (`SALE PRICE`, `SALE_PRICE`, `SALEPRICE`), we’ll report MAE/RMSE/R² on those rows.  
4. We’ll impute missing values: **numeric → median**, **categorical → mode** (based on training data).  
                """
            )
            uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch_csv")

            # ---- Data Consistency Report helper
            def build_consistency_report(df_in_eng: pd.DataFrame, active_feats: List[str]) -> pd.DataFrame:
                present = []
                missing = []
                dtype = []
                strategy = []

                for f in active_feats:
                    if f in df_in_eng.columns:
                        present.append(f)
                        missing.append("")
                        s = df_in_eng[f]
                        dtype.append(str(s.dtype))
                        strategy.append("as-is")
                    else:
                        present.append("")
                        missing.append(f)
                        dtype.append("n/a")
                        strategy.append("filled: numeric→median, categorical→mode")

                # Extra columns (not used by model features)
                extra_cols = [c for c in df_in_eng.columns if c not in active_feats]
                rows = []
                for f, m, d, strat in zip(present, missing, dtype, strategy):
                    if f:
                        rows.append({"status": "present", "feature": f, "dtype": d, "handling": strat})
                    else:
                        rows.append({"status": "missing", "feature": m, "dtype": d, "handling": strat})
                for c in extra_cols:
                    rows.append({"status": "extra", "feature": c, "dtype": str(df_in_eng[c].dtype), "handling": "ignored"})
                rep = pd.DataFrame(rows)
                return rep.sort_values(by=["status", "feature"]).reset_index(drop=True)

            if uploaded is not None:
                # Size guard
                if hasattr(uploaded, "size") and _bytes_to_mb(uploaded.size) > MAX_UPLOAD_MB:
                    st.error(f"Uploaded CSV too large ({_bytes_to_mb(uploaded.size):.1f} MB > {MAX_UPLOAD_MB} MB).")
                else:
                    try:
                        df_in = pd.read_csv(uploaded, on_bad_lines="skip", low_memory=False, engine="python")
                        df_in_norm = _normalize_columns(df_in)
                        df_in_eng = _engineer_features(df_in_norm)

                        # ---- Show Consistency Report BEFORE scoring
                        report = build_consistency_report(df_in_eng, features)
                        st.markdown("### Data Consistency Report")
                        st.dataframe(report, use_container_width=True, height=300)
                        csv_buf = io.StringIO()
                        report.to_csv(csv_buf, index=False)
                        st.download_button("Download consistency_report.csv", csv_buf.getvalue(), file_name="consistency_report.csv", mime="text/csv")

                        # ---- Build model input with imputation
                        X_ui = df_in_eng.reindex(columns=features, fill_value=np.nan)
                        X_ui = X_ui.fillna(medians.to_dict()).fillna(modes)
                        preds_np = predict_any(X_ui)
                        preds = pd.Series(preds_np, index=X_ui.index, name="PREDICTED_SALE_PRICE")

                        out = df_in.copy()
                        out["PREDICTED_SALE_PRICE"] = preds.astype(float)
                        st.success(f"Scored {len(out):,} rows.")
                        st.dataframe(out.head(MAX_BATCH_PREVIEW_ROWS), use_container_width=True)

                        # Optional evaluation if target present in the upload
                        target_col = _find_target(df_in_eng)
                        if target_col:
                            try:
                                y_true = pd.to_numeric(df_in_eng[target_col], errors="coerce")
                                mask = y_true.notna() & preds.notna()
                                if mask.sum() > 0:
                                    mae = mean_absolute_error(y_true[mask], preds[mask])
                                    rmse = mean_squared_error(y_true[mask], preds[mask], squared=False)
                                    r2 = r2_score(y_true[mask], preds[mask])
                                    st.info(f"Eval on uploaded rows with ground truth — MAE: {usd(mae)}, RMSE: {usd(rmse)}, R²: {r2:.3f}")
                            except Exception:
                                st.caption("(Evaluation skipped due to errors)")

                        # Add empirical CI columns based on training residuals
                        try:
                            target_col_train = _find_target(df_train_eng)
                            if target_col_train:
                                df_for_pred = df_train_eng[features].copy()
                                df_for_pred = df_for_pred.fillna(medians.to_dict()).fillna(modes)
                                preds_train_np = predict_any(df_for_pred)
                                preds_train = pd.Series(preds_train_np, index=df_for_pred.index, name="pred")
                                ytrain = pd.to_numeric(df_train_eng[target_col_train], errors="coerce")
                                mask_t = ytrain.notna() & preds_train.notna()
                                if mask_t.sum() > 0:
                                    resid_train = (ytrain[mask_t] - preds_train[mask_t])
                                    conf_choice = st.slider("Confidence (%) for batch intervals", 50, 99, 90, 1)
                                    alpha = (100 - conf_choice) / 100.0
                                    lo_q = resid_train.quantile(alpha / 2.0)
                                    hi_q = resid_train.quantile(1 - alpha / 2.0)
                                    out["CI_LO"] = out["PREDICTED_SALE_PRICE"] + lo_q
                                    out["CI_HI"] = out["PREDICTED_SALE_PRICE"] + hi_q
                                    st.write(f"Added empirical {conf_choice}% interval columns CI_LO / CI_HI based on training residuals.")
                                    st.dataframe(out.head(10), use_container_width=True)
                        except Exception:
                            st.caption("(Could not compute batch confidence intervals)")

                        buf = io.StringIO()
                        out.to_csv(buf, index=False)
                        st.download_button("Download predictions CSV", buf.getvalue(), file_name="nyc_predictions.csv", mime="text/csv")

                    except Exception as e:
                        st.error(f"Batch scoring failed: {e}")
                        try:
                            st.text(traceback.format_exc())
                        except Exception:
                            pass

# --- Tab 2: Explainability & SHAP
with tabs[2]:
    st.subheader("Explainability & SHAP Dashboard")
    st.caption("Generate SHAP explanations for single inputs or training-data samples. Choose the visual you prefer.")
    if model is None or df_train_eng is None or not features or predict_any is None or medians is None:
        st.warning("Load model & training data first (auto-detected from disk).")
    else:
        # Build background for SHAP
        ok_shap = False
        try:
            bg_df = df_train_eng[features].copy()
            bg_df = bg_df.fillna(medians.to_dict()).fillna(modes)
            # Keep to a reasonable size for performance
            if len(bg_df) > MAX_SHAP_BACKGROUND:
                bg_df = bg_df.sample(MAX_SHAP_BACKGROUND, random_state=42)

            # SHAP needs a function that accepts numpy -> wrap to rebuild DataFrame with UI columns
            shap_predict = _predict_fn_factory(model, features, ui_to_model)
            with st.spinner("Preparing SHAP explainer..."):
                # Provide background as array for model-agnostic Explainer
                # (works across SHAP versions)
                explainer = shap.Explainer(shap_predict, bg_df.values, feature_names=features)
            ok_shap = True
        except Exception as e:
            st.error(f"SHAP explainer initialization failed: {e}")
            try:
                st.text(traceback.format_exc())
            except Exception:
                pass
            ok_shap = False

        if ok_shap:
            explain_mode = st.radio("Explain mode", ["Single Input (manual)", "Sample from training data"], horizontal=True)
            if explain_mode == "Single Input (manual)":
                st.markdown("Create a single input (defaults use training medians/modes).")
                cols = st.columns(3)
                values = {}
                for i, feat in enumerate(features):
                    with cols[i % 3]:
                        s = df_train_eng[feat] if feat in df_train_eng.columns else pd.Series(dtype=float)
                        if pd.api.types.is_numeric_dtype(s):
                            p = pctl.get(
                                feat,
                                {
                                    "p1": float(s.quantile(0.01)) if not s.empty else 0.0,
                                    "p50": float(s.median()) if not s.empty else 0.0,
                                    "p99": float(s.quantile(0.99)) if not s.empty else 1.0,
                                },
                            )
                            span = max(0.0, p["p99"] - p["p1"])
                            step = max(1.0, span / 100) if span > 0 else 1.0
                            val = p["p50"]
                            values[feat] = st.number_input(feat, value=float(val), step=step)
                        else:
                            default_text = ""
                            if not s.empty:
                                try:
                                    m = s.mode(dropna=True)
                                    if not m.empty:
                                        default_text = str(m.iloc[0])
                                except Exception:
                                    default_text = ""
                            values[feat] = st.text_input(feat, value=default_text, max_chars=200)

                if st.button("Explain input"):
                    try:
                        X_df = pd.DataFrame([values], columns=features)
                        X_df = X_df.fillna(medians.to_dict()).fillna(modes)
                        pred = float(predict_any(X_df)[0])
                        st.success(f"Predicted: {usd(pred)}")
                        sv = explainer(X_df.values)
                        plot_type = st.selectbox("SHAP plot type", ["Waterfall", "Bar", "Beeswarm", "Scatter"])
                        if plot_type == "Waterfall":
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.waterfall(sv[0], show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Waterfall plot unavailable)")
                            finally:
                                plt.close("all")
                        elif plot_type == "Bar":
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.bar(sv[0], show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Bar plot unavailable)")
                            finally:
                                plt.close("all")
                        elif plot_type == "Beeswarm":
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.beeswarm(sv, show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Beeswarm plot unavailable)")
                            finally:
                                plt.close("all")
                        else:
                            feat_choice = st.selectbox("Feature for scatter", features)
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.scatter(sv[:, feat_choice], show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Scatter plot unavailable)")
                            finally:
                                plt.close("all")
                    except Exception as e:
                        st.error(f"SHAP explanation failed: {e}")
                        try:
                            st.text(traceback.format_exc())
                        except Exception:
                            pass
            else:
                # Make slider bounds robust to small datasets to avoid Streamlit value-range errors
                max_n = int(len(df_train_eng)) if df_train_eng is not None else 0
                if max_n <= 0:
                    st.warning("No rows available for SHAP sampling.")
                else:
                    default_n = min(1000, max_n)
                    min_n = 1
                    step_val = max(1, default_n // 10)
                    sample_n = st.slider("Sample size", min_value=min_n, max_value=max_n, value=default_n, step=step_val)
                    try:
                        df_sample = df_train_eng[features].copy()
                        df_sample = df_sample.fillna(medians.to_dict()).fillna(modes)
                        df_sample = df_sample.sample(n=min(sample_n, len(df_sample)), random_state=42)
                        sv = explainer(df_sample.values)
                        plot_type = st.selectbox("SHAP plot type (global)", ["Bar (global)", "Beeswarm", "Scatter (single feature)"])
                        if plot_type == "Bar (global)":
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.bar(sv, show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Global bar plot unavailable)")
                            finally:
                                plt.close("all")
                        elif plot_type == "Beeswarm":
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.beeswarm(sv, show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Beeswarm plot unavailable)")
                            finally:
                                plt.close("all")
                        else:
                            feat_choice = st.selectbox("Feature for scatter", features)
                            try:
                                fig = plt.figure(figsize=(8, 4))
                                shap.plots.scatter(sv[:, feat_choice], show=False)
                                st.pyplot(fig, clear_figure=True)
                            except Exception:
                                st.caption("(Scatter plot unavailable)")
                            finally:
                                plt.close("all")
                    except Exception as e:
                        st.error(f"Global SHAP failed: {e}")
                        try:
                            st.text(traceback.format_exc())
                        except Exception:
                            pass

        # Partial Dependence (fast approximate)
        st.markdown("---")
        st.markdown("### Partial Dependence (fast approximate)")
        st.caption("Quick PDPs over the most important features (as detected for your current model/data).")
        if features:
            if st.button("Compute a quick PDP for top 1-2 features"):
                f1 = features[0]
                f2 = features[1] if len(features) > 1 else None
                try:
                    base = df_train_eng[features].copy().fillna(medians.to_dict()).fillna(modes).median(numeric_only=True).to_dict()
                    grid_pts = 40
                    if f2:
                        v1 = np.linspace(pctl[f1]["p1"], pctl[f1]["p99"], grid_pts)
                        v2 = np.linspace(pctl[f2]["p1"], pctl[f2]["p99"], grid_pts)
                        rows = []
                        for a in v1:
                            for b in v2:
                                r = base.copy()
                                r[f1] = a
                                r[f2] = b
                                rows.append(r)
                        grid_df = pd.DataFrame(rows, columns=features).fillna(medians.to_dict()).fillna(modes)
                        Xg = grid_df.reindex(columns=features, fill_value=np.nan)
                        preds = predict_any(Xg)
                        fig = px.density_heatmap(grid_df, x=f1, y=f2, z=preds, histfunc="avg", nbinsx=40, nbinsy=40, title=f"PDP heatmap: {f1} × {f2}")
                        fig.update_layout(coloraxis_colorbar=dict(title="Pred Price"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        v1 = np.linspace(pctl[f1]["p1"], pctl[f1]["p99"], grid_pts)
                        rows = []
                        for a in v1:
                            r = base.copy()
                            r[f1] = a
                            rows.append(r)
                        grid_df = pd.DataFrame(rows, columns=features).fillna(medians.to_dict()).fillna(modes)
                        Xg = grid_df.reindex(columns=features, fill_value=np.nan)
                        preds = predict_any(Xg)
                        fig = px.line(x=grid_df[f1], y=preds, labels={"x": f1, "y": "Pred Price"}, title=f"PDP: {f1}")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"PDP computation failed: {e}")
                    try:
                        st.text(traceback.format_exc())
                    except Exception:
                        pass

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr><div style='text-align:center'>NYC Property Price Predictor • Research Showcase • © 2025 • Author: <b>Shrish Mudumby Venugopal, Sourish Mudumby Venugopal & Raghav Ganesh </b></div>", unsafe_allow_html=True)
