import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path('docs')
FIGS = DOCS / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

# The paper's headline SHAP result (Section 4.4.2 / Table 3 / Figure 4) is the
# POLITICAL model (lgbm_all_years_political, 17 trained features, dem_share at
# rank 12). Earlier versions of this script resolved whichever model
# experiments/champion.json currently selected and wrote the figure to a generic
# filename, silently swapping models when champion.json changed. We now
# explicitly produce BOTH the paper-model figure and a clearly-named champion
# figure so neither overwrites the other.
PAPER_SHAP_MODEL = 'lgbm_all_years_political'

def _resolve_champion_name():
    """Return the experiment name champion.json currently points at, or None."""
    champ_path = Path('experiments/champion.json')
    reg_path = Path('experiments/registry.json')
    if not (champ_path.exists() and reg_path.exists()):
        return None
    champ_id = json.loads(champ_path.read_text()).get('selected_experiment')
    reg = json.loads(reg_path.read_text())
    for e in reg:
        if e.get('id') == champ_id:
            return e.get('name')
    return None

champ_name = _resolve_champion_name()
if champ_name is None:
    champ_name = 'lgbm_all_years_base'  # fallback

def _plot_shap_topN(shap_path, n, out_png, title_suffix):
    """Read a *_shap_summary.json and render a horizontal bar chart of the top N."""
    if not shap_path.exists():
        print(f'  SHAP summary not found: {shap_path} — skipping {out_png.name}')
        return None
    s = json.loads(shap_path.read_text())
    summary = s.get('summary', [])[:n]
    df_shap = pd.DataFrame(summary).sort_values('mean_abs_shap', ascending=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_shap['feature'], df_shap['mean_abs_shap'], color='#1f77b4')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(f'Top {n} features by mean |SHAP| ({title_suffix})')
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f'  saved {out_png}')
    return df_shap

# 1) SHAP figures — paper/political model is primary, champion model is secondary.
print('SHAP figures:')
print('  paper model (political):', PAPER_SHAP_MODEL)
print('  current champion       :', champ_name)

# Primary (paper) figures: shap_top10.png / shap_top15.png = POLITICAL model.
paper_top15 = _plot_shap_topN(
    Path('models/artifacts') / f"{PAPER_SHAP_MODEL}_shap_summary.json",
    15,
    FIGS / 'shap_top15.png',
    'Political model (paper)',
)
paper_top10 = _plot_shap_topN(
    Path('models/artifacts') / f"{PAPER_SHAP_MODEL}_shap_summary.json",
    10,
    FIGS / 'shap_top10.png',
    'Political model (paper)',
)

# Secondary (champion) figures: kept under distinct, clearly-labeled names so
# they never overwrite the paper figures and never silently swap models.
if champ_name != PAPER_SHAP_MODEL:
    print('  (champion differs from paper model -> also emitting champion figures)')
    _plot_shap_topN(
        Path('models/artifacts') / f"{champ_name}_shap_summary.json",
        15,
        FIGS / 'shap_top15_champion.png',
        f'Champion ({champ_name})',
    )
    _plot_shap_topN(
        Path('models/artifacts') / f"{champ_name}_shap_summary.json",
        10,
        FIGS / 'shap_top10_champion.png',
        f'Champion ({champ_name})',
    )
else:
    print('  (champion == paper model; no separate champion figure needed)')

# 2) Model comparison (non-tainted experiments)
reg_path = Path('experiments/registry.json')
if reg_path.exists():
    reg = json.loads(reg_path.read_text())
    rows = []
    for e in reg:
        if e.get('tainted'):
            continue
        m = e.get('metrics', {}) or {}
        mae = m.get('mae') or m.get('mae_log_price') or None
        rows.append({'name': e.get('name'), 'mae': mae})
    df_comp = pd.DataFrame(rows).dropna().sort_values('mae')
    # champion
    champ = None
    champ_path = Path('experiments/champion.json')
    if champ_path.exists():
        try:
            champ = json.loads(champ_path.read_text()).get('selected_experiment')
            # Map id->name
            reg_map = {e.get('id'): e.get('name') for e in reg}
            champ_name = reg_map.get(champ)
        except Exception:
            champ_name = None
    else:
        champ_name = None
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ca02c' if (champ_name and name == champ_name) else '#1f77b4' for name in df_comp['name']]
    ax.barh(df_comp['name'], df_comp['mae'], color=colors)
    ax.set_xlabel('MAE')
    ax.set_title('Model comparison (MAE)')
    plt.tight_layout()
    fig.savefig(FIGS / 'model_comparison.png', dpi=300)
    plt.close(fig)

# 3) Scenario impact by district
scen_dir = Path('results/political_scenarios')
scens = ['liberal_policy', 'conservative_policy', 'mixed_governance']
dfs = []
for s in scens:
    f = scen_dir / f"{s}_by_council.csv"
    if f.exists():
        df = pd.read_csv(f)
        # accept common name for district
        cid_col = next((c for c in df.columns if 'coun' in c.lower() and 'dist' in c.lower()), None)
        if cid_col is None:
            cid_col = df.columns[0]
        # prefer mean_delta or mean_pct_change or mean_pct_change
        if 'mean_delta' in df.columns:
            val_col = 'mean_delta'
        elif 'mean_pct_change' in df.columns:
            val_col = 'mean_pct_change'
        elif 'mean_delta' in df.columns:
            val_col = 'mean_delta'
        elif 'scen_mean' in df.columns and 'base_mean' in df.columns:
            df['mean_delta'] = df['scen_mean'] - df['base_mean']
            val_col = 'mean_delta'
        elif 'pct_change' in df.columns:
            val_col = 'pct_change'
        else:
            # pick first numeric column besides count
            val_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != cid_col), None)
        df2 = df[[cid_col, val_col]].copy()
        df2.columns = ['CounDist', s]
        df2['CounDist'] = df2['CounDist'].astype(float).astype(int)
        dfs.append(df2.set_index('CounDist'))

if dfs:
    merged = pd.concat(dfs, axis=1).fillna(0)
    merged = merged.sort_index()
    # grouped bar chart
    ind = np.arange(len(merged.index))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, s in enumerate(merged.columns):
        ax.bar(ind + i*width, merged[s].values, width, label=s)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(merged.index.astype(str), rotation=90)
    ax.set_xlabel('Council District')
    ax.set_ylabel('Mean delta / pct')
    ax.set_title('Scenario impact by council district')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS / 'scenario_impact_by_district.png', dpi=300)
    plt.close(fig)

print('Figures saved to', FIGS)
