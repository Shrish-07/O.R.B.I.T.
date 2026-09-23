"""Figure 5 (Section 4.7): three-panel mean scenario impact by council district,
written to docs/figures/scenario_impact_all_three.png — the exact filename the
manuscript's \\includegraphics{../figures/scenario_impact_all_three.png} expects.

Why this script exists (2026-09-06, Round-4 follow-up):
previously NO script in this repo wrote that exact file (the manuscript's copy
lived only in the paper's own figures/ folder; the repo's producers wrote
scenario_impact_all_three_FIXED.png and scenario_impact_by_district.png).
Panel styling mirrors generate_scenario_map.py (both scripts share the same zero-straddle-safe color norm);
the data source is political_scenarios.py's per-district CSVs.

Confirms at write time: 51 districts per scenario panel, three panels,
manuscript-exact output filename.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ['liberal_policy', 'conservative_policy', 'mixed_governance']
TITLES = ['Liberal Policy Scenario\n(+0.15 dem_share)',
          'Conservative Policy Scenario\n(-0.15 dem_share)',
          'Mixed Governance Scenario\n(balanced shift)']
CMAPS = ['RdYlGn', 'RdYlGn_r', 'PuOr']

frames = {}
for s in SCENARIOS:
    df_s = pd.read_csv(ROOT / 'results' / 'political_scenarios' / f'{s}_by_council.csv')
    ccol = next(c for c in df_s.columns if 'coun' in c.lower())
    frames[s] = df_s[[ccol, 'mean_pct_change']].rename(columns={ccol: 'CounDist'})
    assert len(frames[s]) == 51, f'{s}: expected 51 districts, got {len(frames[s])}'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, scen, title, cmap in zip(axes, SCENARIOS, TITLES, CMAPS):
    df_s = frames[scen].sort_values('mean_pct_change')
    vals = df_s['mean_pct_change']
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        # One-sided data (all values share a sign): TwoSlopeNorm requires
        # vmin < vcenter < vmax, so fall back to a plain linear norm.
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(vals))
    ax.barh(df_s['CounDist'].astype(int).astype(str), vals, color=colors, alpha=0.9)
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Mean % Price Change', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white', labelsize=7)
    ax.spines[['bottom', 'left', 'top', 'right']].set_color('#30363d')
    max_d = df_s.loc[vals.abs().idxmax(), 'CounDist']
    max_v = vals.abs().max()
    ax.set_xlabel(f'% Price Change | Max: District {int(max_d)}: {max_v:.1f}%',
                  color='white', fontsize=9)

fig.patch.set_facecolor('#0d1117')
fig.suptitle('Political Scenario Impact on Property Prices by NYC Council District',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()
out = OUT / 'scenario_impact_all_three.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Saved {out.relative_to(ROOT)}')
for scen in SCENARIOS:
    df_s = frames[scen]
    print(f'{scen}: mean={df_s["mean_pct_change"].mean():.2f}%  '
          f'positive={(df_s["mean_pct_change"] > 0).sum()}  '
          f'negative={(df_s["mean_pct_change"] < 0).sum()}')
