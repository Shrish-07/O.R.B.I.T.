import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DOCS = Path('docs')
FIGS = DOCS / 'figures'
DOCS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


registry = json.loads(Path('experiments/registry.json').read_text())
champ = json.loads(Path('experiments/champion.json').read_text())

# model comparison table (exclude tainted and stale)
rows = []
for e in registry:
    if e.get('tainted') or e.get('stale'):
        continue
    m = e.get('metrics', {})
    mae = m.get('mae_log_price') or m.get('mae')
    rows.append({'name': e.get('name'), 'mae': mae, 'r2': m.get('r2_temporal') or m.get('r2')})
df_comp = pd.DataFrame(rows).sort_values('mae')
df_comp.to_csv(DOCS / 'model_comparison_table.csv', index=False)

# bar chart of MAE
plt.figure(figsize=(8,4))
plt.bar(df_comp['name'], df_comp['mae'])
plt.xticks(rotation=90)
plt.title('Model comparison (MAE)')
plt.tight_layout()
plt.savefig(FIGS / 'model_comparison_mae.png', dpi=300)
plt.close()

# SHAP top-10 if available
shap_path = Path('models/artifacts/lgbm_all_years_political_shap_summary.json')
shap_list = []
if shap_path.exists():
    shap = json.loads(shap_path.read_text())
    shap_list = shap.get('summary', [])[:10]
    df_shap = pd.DataFrame(shap_list)
    df_shap.plot.bar(x='feature', y='mean_abs_shap', legend=False, figsize=(8,4))
    plt.title('SHAP top-10 (champion)')
    plt.tight_layout()
    plt.savefig(FIGS / 'shap_top10.png', dpi=300)
    plt.close()

# political scenario summary
scen_dir = Path('results/political_scenarios')
scen_files = list(scen_dir.glob('*_by_council.csv'))
scen_summaries = {}
for f in scen_files:
    df = pd.read_csv(f)
    scen_summaries[f.stem] = df['mean_delta'].abs().mean() if 'mean_delta' in df.columns else None

# write markdown summary
with open(DOCS / 'research_summary.md', 'w') as f:
    f.write('# Research Summary\n\n')
    f.write('## Champion\n')
    f.write(f"- Selected champion: {champ.get('selected_experiment')} (mae={champ.get('metric')})\n\n")
    f.write('## Model comparison\n')
    f.write('\n')
    f.write(df_comp.to_csv(index=False))
    f.write('\n\n')
    if shap_list:
        f.write('## SHAP top-10\n')
        f.write('\n')
        f.write(df_shap.to_string(index=False))
        f.write('\n\n')
    if scen_summaries:
        f.write('## Political scenario summaries\n')
        for k,v in scen_summaries.items():
            f.write(f'- {k}: mean abs delta by council = {v}\n')

print('Research summary generated at', DOCS / 'research_summary.md')
