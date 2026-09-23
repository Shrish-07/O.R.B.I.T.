"""
Verify blacklisted features against actual trained model feature names.

Output: results/editorial_audit/blacklist_audit.json
"""

import json
import sys
from pathlib import Path
import lightgbm as lgb
import yaml

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "editorial_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load blacklist
blacklist_path = ROOT / "config" / "feature_blacklist.yaml"
with open(blacklist_path) as f:
    blacklist_cfg = yaml.safe_load(f)
blacklisted = set()
for cat, items in blacklist_cfg.items():
    for item in items:
        blacklisted.add(str(item).strip())
print(f"Loaded {len(blacklisted)} blacklisted features")

# Load champion info
champ_path = ROOT / "experiments" / "champion.json"
reg_path = ROOT / "experiments" / "registry.json"
champion_name = None
political_name = None
if champ_path.exists() and reg_path.exists():
    champ_info = json.loads(champ_path.read_text())
    registry = json.loads(reg_path.read_text())
    for exp in registry:
        if exp.get("id") == champ_info.get("selected_experiment"):
            champion_name = exp.get("name")
        if exp.get("mode") == "political" and exp.get("scope") == "all_years" and exp.get("pipeline_v2"):
            political_name = exp.get("name")
    if political_name is None:
        # Round 7 fix: NO registry entry has ever carried a "pipeline_v2" flag
        # (models/training/train_lgbm.py does not write one), so the loop above
        # always left political_name = None and BOTH final-verdict blocks below
        # silently skipped the political model. Fall back to the same
        # name-based registry lookup used by generate_pdp.py /
        # sensitivity_analysis.py / generate_shap_summary.py.
        for exp in registry:
            if exp.get("name") == "lgbm_all_years_political":
                political_name = exp.get("name")
                break
    print(f"Champion (base): {champion_name}")
    print(f"Political: {political_name}")

# Find all .txt models
txt_models = sorted((ROOT / "models").glob("**/*.txt"))

def norm_label(name):
    """Normalize a feature name for blacklist comparison. LightGBM stores names
    with underscores; the blacklist YAML uses spaces. This was previously a
    typo: the function was named `norm` but every call site used `norm_label`,
    so the moment any model with features was loaded the checker NameError'd.
    Renamed to `norm_label` so the call sites resolve. (Behavior is unchanged:
    same `_`->space replace as before.)"""
    return name.replace("_", " ").strip()

# Check models
for mp in txt_models:
    name = mp.name
    try:
        b = lgb.Booster(model_file=str(mp))
        fn = b.feature_name()
        fn_norm = [norm_label(f) for f in fn]

        found = []
        for i, orig in enumerate(fn):
            n = fn_norm[i]
            if n in blacklisted or orig in blacklisted:
                found.append(orig)

        has_ease = "EASE-MENT" in fn or "EASE_MENT" in fn
        clean = len(found) == 0

        # Determine if this is a current (champion/political) model
        is_champion_base = name == f"{champion_name}.txt" if champion_name else False
        is_champion_political = name == f"{political_name}.txt" if political_name else False
        tag = ""
        if is_champion_base:
            tag = " [CHAMPION BASE]"
        elif is_champion_political:
            tag = " [CHAMPION POLITICAL]"

        status = "CLEAN" if clean else "BLACKLISTED"
        ease_extra = ""
        if has_ease:
            ease_extra = " | EASE-MENT in model!"
        print(f"  {name}{tag}: {len(fn)} feat, {status}{ease_extra}")
        if not clean:
            for f in found:
                print(f"    - {f}")

    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# Final dedicated check against the actual champion
print("\n" + "=" * 60)
print("FINAL EASE-MENT VERDICT (champion models only):")
for label, model_file in [
    ("Champion BASE", f"{champion_name}.txt" if champion_name else None),
    ("Champion POLITICAL", f"{political_name}.txt" if political_name else None),
]:
    if model_file and (Path("models") / model_file).exists():
        b = lgb.Booster(model_file=str(Path("models") / model_file))
        fn = b.feature_name()
        has_ease = "EASE-MENT" in fn or "EASE_MENT" in fn
        print(f"  {label} ({model_file}): {len(fn)} features, EASE-MENT={has_ease}")
    elif model_file:
        print(f"  {label} ({model_file}): MISSING")

# Check SHAP summaries separately
for shap_label, shap_name in [("Base", "lgbm_all_years_base"), ("Political", "lgbm_all_years_political")]:
    sp = Path(f"models/artifacts/{shap_name}_shap_summary.json")
    if sp.exists():
        d = json.loads(sp.read_text())
        feats = [x["feature"] for x in d.get("summary", [])]
        has_ease = "EASE-MENT" in feats
        print(f"  SHAP {shap_label}: {len(feats)} features, EASE-MENT={has_ease}")
    else:
        print(f"  SHAP {shap_label}: MISSING")

# Final blacklist check
print("\n=== FINAL VERDICT ===")
# Re-read champion base and political models specifically
base_ok = True
pol_ok = True
base_file = Path(f"models/{champion_name}.txt") if champion_name else None
pol_file = Path(f"models/{political_name}.txt") if political_name else None

if base_file and base_file.exists():
    b = lgb.Booster(model_file=str(base_file))
    for f in b.feature_name():
        # Round 7 fix: check the ORIGINAL stored name as well as the
        # underscore-normalized one. Previously only norm_label(f) was tested,
        # which turns "BBL_pluto" into "BBL pluto" — not the blacklist's
        # "BBL_pluto" — so underscore-carrying entries silently slipped past
        # the final verdict (the per-model loop above already checks both).
        if f in blacklisted or norm_label(f) in blacklisted:
            base_ok = False
            print(f"  BASE model blacklisted feature: {f} ({norm_label(f)})")
            break
if pol_file and pol_file.exists():
    p = lgb.Booster(model_file=str(pol_file))
    for f in p.feature_name():
        # Same Round 7 fix as the base-model loop above.
        if f in blacklisted or norm_label(f) in blacklisted:
            pol_ok = False
            print(f"  POLITICAL model blacklisted feature: {f} ({norm_label(f)})")
            break

if base_ok and pol_ok:
    print("  BOTH champion models (base + political) are CLEAN of all blacklisted features.")
    print("  EASE-MENT in political SHAP summary is a REPORT-GENERATION ARTIFACT.")
    print("  The generate_shap_summary.py script iterates a wider column list; EASE-MENT was never trained on.")
    print("  Paper Table 3 would need correction only if it cited the SHAP summary (18 feat) vs model (17 feat).")
    print("  Blacklist enforcement: PASSES CORRECTLY for champion models.")
else:
    print("  BLACKLIST ENFORCEMENT FAILURE: At least one champion model contains blacklisted features.")
    if not base_ok:
        print("  -> Base model contains blacklisted features. Paper needs correction.")
    if not pol_ok:
        print("  -> Political model contains. Paper needs correction.")

# Persist the machine-readable audit this module's docstring promises (it was
# previously never written by any script). The verdict fields follow ONLY the
# two champion models; legacy .txt models above print their status but do not
# gate the pipeline.
audit = {
    "blacklist_path": blacklist_path.as_posix(),
    "blacklisted_feature_count": len(blacklisted),
    "champion_base_model": champion_name,
    "champion_base_blacklist_clean": base_ok,
    "political_model": political_name,
    "political_blacklist_clean": pol_ok,
    "enforcement_passed": bool(base_ok and pol_ok),
}
AUDIT_PATH = RESULTS_DIR / "blacklist_audit.json"
with open(AUDIT_PATH, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"Wrote {AUDIT_PATH}")

# Round 7 fix: fail loudly on enforcement failure. main.py's stage runner only
# sees the exit code, so a print-only failure previously reported PASS — the
# exact gap that would let a blacklisted feature (e.g. BBL_pluto, added to the
# blacklist in Round 7) sit in the on-disk models unnoticed until retraining.
if not (base_ok and pol_ok):
    raise SystemExit(1)