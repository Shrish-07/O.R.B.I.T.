import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path('docs')
FIGS = DOCS / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

# 1) SHAP top-15 — use champion dynamically, not hardcoded
champ_json = json.loads(Path('experiments/champion.json').read_text())
reg_json = json.loads(Path('experiments/registry.json').read_text())
champ_id = champ_json.get('selected_experiment')
champ_name = None
for e in reg_json:
    if e.get('id') == champ_id:
        champ_name = e.get('name')
        break
if champ_name is None:
    champ_name = 'lgbm_all_years_base'  # fallback
shap_path = Path('models/artifacts') / f"{champ_name}_shap_summary.json"
if shap_path.exists():
    s = json.loads(shap_path.read_text())
    summary = s.get('summary', [])[:15]
    df_shap = pd.DataFrame(summary)
    df_shap = df_shap.sort_values('mean_abs_shap', ascending=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_shap['feature'], df_shap['mean_abs_shap'], color='#1f77b4')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('Top 15 features by mean |SHAP| (Champion)')
    plt.tight_layout()
    fig.savefig(FIGS / 'shap_top15.png', dpi=300)
    plt.close(fig)

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
