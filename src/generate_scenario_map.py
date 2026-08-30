from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

scenarios = {}
for s in ['liberal_policy', 'conservative_policy', 'mixed_governance']:
    df_s = pd.read_csv(ROOT / 'results' / 'political_scenarios' / f'{s}_by_council.csv')
    scenarios[s] = df_s

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors_map = ['liberal_policy', 'conservative_policy', 'mixed_governance']
titles = ['Liberal Policy Scenario\n(+0.15 dem_share)', 'Conservative Policy Scenario\n(-0.15 dem_share)', 'Mixed Governance Scenario\n(balanced shift)']
cmaps = ['RdYlGn', 'RdYlGn_r', 'PuOr']

for ax, scen, title, cmap in zip(axes, colors_map, titles, cmaps):
    df_s = scenarios[scen]
    vcol = 'mean_pct_change'
    ccol = next(c for c in df_s.columns if 'coun' in c.lower())
    sorted_df = df_s.sort_values(vcol)
    vals = sorted_df[vcol]
    norm = mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=vals.max())
    colors = plt.get_cmap(cmap)(norm(vals))
    ax.barh(sorted_df[ccol].astype(int).astype(str), sorted_df[vcol], color=colors, alpha=0.9)
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Mean % Price Change', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white', labelsize=7)
    ax.spines[['bottom','left','top','right']].set_color('#30363d')
    max_d = sorted_df.loc[sorted_df[vcol].abs().idxmax(), ccol]
    max_v = sorted_df[vcol].abs().max()
    ax.set_xlabel(f'% Price Change | Max: District {int(max_d)}: {max_v:.1f}%', color='white', fontsize=9)

fig.patch.set_facecolor('#0d1117')
fig.suptitle('Political Scenario Impact on Property Prices by NYC Council District', color='white', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'scenario_impact_all_three_FIXED.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Saved scenario_impact_all_three_FIXED.png')
print()
for scen in colors_map:
    df_s = scenarios[scen]
    print(f'{scen}: max={df_s["mean_pct_change"].abs().max():.2f}%  mean={df_s["mean_pct_change"].mean():.2f}%  districts_positive={(df_s["mean_pct_change"]>0).sum()}  districts_negative={(df_s["mean_pct_change"]<0).sum()}')
