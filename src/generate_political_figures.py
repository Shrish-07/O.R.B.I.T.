from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Load canonical dataset
Df = pd.read_parquet(ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet')

# Figure 1: dem_share vs median log price by council district
if 'CounDist' not in Df.columns:
    raise KeyError('CounDist column not found in canonical dataset')
for col in ['target_log_price', 'dem_share']:
    if col not in Df.columns:
        raise KeyError(f'{col} column not found in canonical dataset')

district_stats = Df.groupby('CounDist').agg(
    median_log_price=('target_log_price', 'median'),
    mean_dem_share=('dem_share', 'mean'),
    count=('target_log_price', 'count')
).reset_index()
district_stats = district_stats[district_stats['count'] >= 50]

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    district_stats['mean_dem_share'],
    district_stats['median_log_price'],
    c=district_stats['median_log_price'],
    cmap='RdYlGn', s=district_stats['count'] / 20, alpha=0.8, edgecolors='white', linewidths=0.5
)
z = np.polyfit(district_stats['mean_dem_share'], district_stats['median_log_price'], 1)
p = np.poly1d(z)
xline = np.linspace(district_stats['mean_dem_share'].min(), district_stats['mean_dem_share'].max(), 100)
ax.plot(xline, p(xline), 'w--', alpha=0.7, linewidth=1.5, label='Linear trend')
for _, row in district_stats.iterrows():
    ax.annotate(int(row['CounDist']), (row['mean_dem_share'], row['median_log_price']),
                fontsize=6, color='#cccccc', alpha=0.7)
corr = district_stats['mean_dem_share'].corr(district_stats['median_log_price'])
ax.set_xlabel('Mean Democratic Vote Share (dem_share)', color='white', fontsize=12)
ax.set_ylabel('Median Log Sale Price', color='white', fontsize=12)
ax.set_title(f'Political Ideology vs Property Prices by NYC Council District\n(r = {corr:.3f}, n={len(district_stats)} districts)', color='white', fontsize=13)
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#0d1117')
ax.tick_params(colors='white')
ax.spines[['bottom','left','top','right']].set_color('#30363d')
plt.colorbar(scatter, ax=ax, label='Log Price').ax.yaxis.label.set_color('white')
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
plt.tight_layout()
plt.savefig(OUT / 'political_ideology_vs_price.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Saved political_ideology_vs_price.png | correlation: {corr:.4f}')

# Figure 2: Temporal ideology shift 2017 to 2025
ideo = pd.read_parquet(ROOT / 'data' / 'processed' / 'ideology_by_council.parquet')
if 'CounDist' not in ideo.columns or 'election_year' not in ideo.columns or 'dem_share' not in ideo.columns:
    raise KeyError('ideology_by_council parquet missing required columns')
pivot = ideo.pivot(index='CounDist', columns='election_year', values='dem_share').reset_index()
if 2017 not in pivot.columns or 2025 not in pivot.columns:
    raise KeyError('Required election years not present in ideology_by_council.parquet')
pivot['shift_2017_2025'] = pivot[2025] - pivot[2017]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(pivot[2017], pivot[2025], c=pivot['shift_2017_2025'], cmap='RdYlGn', s=80, alpha=0.9, edgecolors='white', linewidths=0.5)
axes[0].plot([0.1, 1.0], [0.1, 1.0], 'w--', alpha=0.5, linewidth=1)
for _, row in pivot.iterrows():
    axes[0].annotate(int(row['CounDist']), (row[2017], row[2025]), fontsize=6, color='#cccccc', alpha=0.7)
axes[0].set_xlabel('dem_share 2017', color='white')
axes[0].set_ylabel('dem_share 2025', color='white')
axes[0].set_title('Ideology Shift: 2017 vs 2025\n(above diagonal = more liberal in 2025)', color='white')
axes[0].set_facecolor('#1a1a2e')
axes[0].tick_params(colors='white')
axes[0].spines[['bottom','left','top','right']].set_color('#30363d')

# Right: top shifting districts bar chart
try:
    # The paper's Figure 2 caption is fixed: "the eight largest positive and
    # four largest negative district-level shifts" (12 districts total:
    # 5, 49, 43, 50, 20, 29, 44, 47 on top; 15, 14, 16, 17 on bottom).
    # Earlier versions used nlargest(10)/nsmallest(5) (15 districts), which
    # added districts 4, 26 and 42 not described by the caption. Changing the
    # code here is the single-place fix (the caption text is the source of
    # truth and stays untouched).
    top_shift = pivot.nlargest(8, 'shift_2017_2025')
    bottom_shift = pivot.nsmallest(4, 'shift_2017_2025')
    combined = pd.concat([top_shift, bottom_shift]).sort_values('shift_2017_2025')
    colors = ['#f85149' if x < 0 else '#56d364' for x in combined['shift_2017_2025']]
    axes[1].barh(combined['CounDist'].astype(str), combined['shift_2017_2025'], color=colors, alpha=0.9)
    axes[1].set_xlabel('Shift in dem_share 2017-2025', color='white')
    axes[1].set_title('Top and Bottom Council District Ideology Shifts', color='white')
    axes[1].set_facecolor('#1a1a2e')
    axes[1].tick_params(colors='white', labelsize=7)
    axes[1].spines[['bottom','left','top','right']].set_color('#30363d')
except Exception:
    axes[1].text(0.5, 0.5, 'Not enough data for shift chart', color='white', ha='center', va='center')

fig.patch.set_facecolor('#0d1117')
fig.suptitle('Temporal Council District Ideology Shifts: 2017 to 2025', color='white', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'temporal_ideology_shift.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print('Saved temporal_ideology_shift.png')
