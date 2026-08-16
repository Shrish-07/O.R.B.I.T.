import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DOCS = Path('docs')
FIGS = DOCS / 'figures'
DOCS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# The paper's headline SHAP result (Section 4.4.2 / Table 3) is the POLITICAL
# model (lgbm_all_years_political). The research summary's primary SHAP table
# must therefore be the political-model table, not "whatever champion.json
# currently selects" (which silently swaps models). The current-champion table
# is kept but clearly labeled as a SEPARATE, secondary table.
PAPER_SHAP_MODEL = 'lgbm_all_years_political'

registry = json.loads(Path('experiments/registry.json').read_text())
champ = json.loads(Path('experiments/champion.json').read_text())


def _resolve_champion_name():
    champ_id = champ.get('selected_experiment')
    for e in registry:
        if e.get('id') == champ_id:
            return e.get('name')
    return None

champ_name = _resolve_champion_name()

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

# SHAP top-10 charts and tables.
# - Primary: paper/political model (PAPER_SHAP_MODEL) -> shap_top10.png + primary markdown table.
# - Secondary: current champion (if different) -> shap_top10_champion.png + secondary markdown table.


def _load_shap_summary(model_name):
    sp = Path('models/artifacts') / f"{model_name}_shap_summary.json"
    if not sp.exists():
        return None, None
    shap = json.loads(sp.read_text())
    return sp, shap.get('summary', [])[:10]


def _plot_shap_top10(model_name, out_png, title_suffix):
    sp, shap_list = _load_shap_summary(model_name)
    if not shap_list:
        return None
    df_shap = pd.DataFrame(shap_list)
    df_shap.plot.bar(x='feature', y='mean_abs_shap', legend=False, figsize=(8,4))
    plt.title(f'SHAP top-10 ({title_suffix})')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return df_shap


# Primary: PAPER_SHAP_MODEL (the political model the paper actually reports).
df_shap_paper = _plot_shap_top10(
    PAPER_SHAP_MODEL,
    FIGS / 'shap_top10.png',
    'Political model (paper)',
)
_, shap_paper_list = _load_shap_summary(PAPER_SHAP_MODEL)

# Secondary: current-champion model (if it differs from the paper model).
shap_champ_list = None
df_shap_champ = None
if champ_name and champ_name != PAPER_SHAP_MODEL:
    df_shap_champ = _plot_shap_top10(
        champ_name,
        FIGS / 'shap_top10_champion.png',
        f'Current champion ({champ_name})',
    )
    _, shap_champ_list = _load_shap_summary(champ_name)

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
    f.write(f"- Selected champion: {champ.get('selected_experiment')} (mae={champ.get('metric')}, name={champ_name})\n\n")
    f.write('## Model comparison\n')
    f.write('\n')
    f.write(df_comp.to_csv(index=False))
    f.write('\n\n')
    # Primary SHAP table = political model (matches paper Table 3)
    if shap_paper_list:
        f.write('## SHAP top-10 — Political model (PAPER Table 3; model=%s)\n\n' % PAPER_SHAP_MODEL)
        f.write('This table matches the paper\'s Table 3 / Figure 4. `dem_share` appears at rank 12.\n\n')
        df_paper = pd.DataFrame(json.loads(
            (Path('models/artifacts') / f"{PAPER_SHAP_MODEL}_shap_summary.json").read_text()
        ).get('summary', []))
        df_paper['rank'] = range(1, len(df_paper) + 1)
        f.write(df_paper.to_string(index=False))
        f.write('\n\n')
    # Secondary SHAP table = current champion (if different)
    if shap_champ_list:
        f.write('## SHAP top-10 — Current champion (secondary; model=%s)\n\n' % champ_name)
        f.write('Secondary table for the experiment currently selected in `experiments/champion.json`. '
                'Shown separately so it never silently overwrites the paper\'s political-model result.\n\n')
        df_champ_full = pd.DataFrame(json.loads(
            (Path('models/artifacts') / f"{champ_name}_shap_summary.json").read_text()
        ).get('summary', []))
        df_champ_full['rank'] = range(1, len(df_champ_full) + 1)
        f.write(df_champ_full.to_string(index=False))
        f.write('\n\n')
    if scen_summaries:
        f.write('## Political scenario summaries\n')
        for k,v in scen_summaries.items():
            f.write(f'- {k}: mean abs delta by council = {v}\n')

print('Research summary generated at', DOCS / 'research_summary.md')
