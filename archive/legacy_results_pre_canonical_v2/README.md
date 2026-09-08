# Legacy pre-canonical-v2 results

`base/` and `political/` (each with `2017/` and `all/` subfolders, each
containing `metrics.json` + `model.txt`) were found sitting in `results/`
during a later audit pass, uncleaned by either reorganization round.

They are NOT part of the current, verified pipeline:
- `political/*/metrics.json` trains on `district_ideology` — the abandoned,
  blacklisted ideology feature (see `config/feature_blacklist.yaml` and
  `PAPER_TO_CODE_MAP.md`'s "Section 3.2" row). No current model trains on it.
- Both use a feature set (`log_land_square_feet`, `log_gross_square_feet`,
  `building_age`, `far`, `SALE_YEAR`, `Latitude`, `Longitude`) that does not
  match the current canonical schema (`models/artifacts/canonical_schema.json`)
  at all — different column names, no `dem_share`, no `BBL_pluto`/`CounDist`.

These are output from an early, since-superseded modeling iteration
(consistent with the `logs/actions.log` timestamps from mid-May 2026, well
before the canonical-v2 rebuild and the `dem_share` production pipeline
existed). Moved here rather than deleted, in case the history is useful;
they should not be confused with anything in `experiments/results/`, which
is the current, verified location for model metrics.
