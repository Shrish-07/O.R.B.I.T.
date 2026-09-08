"""Round-3: diff the regenerated 17-feature political SHAP summary against
the previously recorded (pre-blacklist-era, 18-row) artifact."""
import json

old = json.load(open('archive/lgbm_all_years_political_shap_summary.pre20260906.json'))['summary']
new = json.load(open('models/artifacts/lgbm_all_years_political_shap_summary.json'))['summary']
oldm = {r['feature']: r['mean_abs_shap'] for r in old}
print('new feature count:', len(new))
print('EASE-MENT in new artifact:', 'EASE-MENT' in {r['feature'] for r in new})
print(f"{'feature':28s} {'new':>12s} {'old':>12s} {'abs diff':>10s}")
for i, r in enumerate(new, 1):
    o = oldm.get(r['feature'])
    print(f"{i:2d}. {r['feature']:24s} {r['mean_abs_shap']:12.8f} {o:12.8f} {abs(r['mean_abs_shap'] - o):.2e}")
