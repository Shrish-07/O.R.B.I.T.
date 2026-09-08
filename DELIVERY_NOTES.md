# Delivery notes for this zip

Two things were deliberately left out of this download to keep it a
reasonable size. Neither is a loss — both are regenerable in one command.

1. **`data/`** — excluded entirely. It's already in your `.gitignore` and
   you already have the raw sources (this is what `Data_1.zip`,
   `Data_2.zip`, and `Data_3.zip` were). Recreate it as:
   ```
   data/raw/          <- from Data_1.zip's "raw" folder
   data/splits/        <- from Data_1.zip's "splits" folder
   data/processed/      <- from Data_2.zip's "processed" folder
   data/canonical/      <- from Data_3.zip's "canonical" folder
   data/intermediate/   <- empty (from Data_3.zip; used by the unfinished src/causal/causal_analysis.py)
   data/output/         <- empty (from Data_3.zip)
   ```
   `python main.py --check-data` will confirm it's laid out correctly once
   you've copied it in.

2. **`models/rf_all_years_political_n200.joblib`** — excluded. It's the
   trained Random Forest comparison model and, unusually for a joblib file,
   is ~1.1GB uncompressed (scikit-learn Random Forests serialize every leaf
   of every one of the 200 trees). Every other trained model in this repo
   (LightGBM's two `.txt` dumps, etc.) is small and *is* included. Regenerate
   this one with:
   ```
   python models/training/train_random_forest.py --variant all_years --mode political --n_estimators 200
   ```
   or simply `python main.py --stage train_random_forest`.

Everything else in this zip — every script, every config, every already-run
result, log, table, and figure produced during this reorganization — is
included as-is.

This reorganization was validated against your **real data** and **real
trained models** in the working environment before this zip was built, not
just read for syntax: `python main.py --check-data` passed, and 15 of the
pipeline's stages (all data-pipeline validation, Moran's I, the OLS
regression table, the new robustness-checks script, and 11 of the 15
verification scripts) were actually executed end to end and reproduced the
paper's published numbers, including two real cross-platform path bugs found
and fixed in the process (see `ACTION_PLAN_FOR_SUBMISSION.md`, Part 3).
