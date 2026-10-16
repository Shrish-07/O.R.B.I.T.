import json
from pathlib import Path
import pandas as pd

# Round 9: the interpreted-model set is EVERY eligible registry model, ranked
# via src/ops/champion_ranking.py (rank 1 = champion.json selection; ranks
# 2..N = ascending test MAE, recomputed fresh each run; audit in
# results/rank1_rank2_selection.json). ALL families -- LightGBM included -- go
# through the ONE implementation in src/ops/shap_tiers.py: for the LightGBM
# paper models tier (a) is the same exact-TreeSHAP computation on the same
# low-NA seed-42 sample the pre-Round-9 classic writer used, so their Table 3
# numbers are unchanged; only audit bookkeeping (shap_fallback_tier etc.) is
# added. No family gets a bespoke code path anymore.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ops.champion_ranking import get_all_models
from src.ops.shap_tiers import compute_shap_payload
from src.ops.predictors import load_entry_model

CHAMP = Path('experiments/champion.json')
REG = Path('experiments/registry.json')
SPLIT_DIR = Path('data/splits')
ARTIFACT_DIR = Path('models/artifacts')


def main():
    if not CHAMP.exists() or not REG.exists():
        print('Champion or registry missing')
        raise SystemExit(1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(SPLIT_DIR / 'all_years_test.parquet')

    for entry in get_all_models():
        out_path = ARTIFACT_DIR / f"{entry['name']}_shap_summary.json"
        model, _preproc = load_entry_model(entry)
        payload = compute_shap_payload(entry, model, test)
        out_path.write_text(json.dumps(payload, indent=2))
        print('Saved SHAP summary (method=' + str(payload['method'])
              + ', tier=' + str(payload.get('shap_fallback_tier'))
              + ') for ' + str(entry['name']) + ' (rank ' + str(entry['rank'])
              + ', ' + str(entry['family']) + ') to', out_path)

    print('SHAP summaries written for every eligible model via one code path',
          '(--> models/artifacts/{name}_shap_summary.json).')


if __name__ == '__main__':
    # __main__ guard is REQUIRED: src/ops/shap_tiers.py enforces its per-tier
    # timeout via a spawned child process, and Windows spawn re-imports this
    # module in the child. Without the guard the child would re-execute the
    # whole script (and would recursively spawn its own children).
    main()
