"""Fresh cross-cutting probes for TRUTH.md section K (this session)."""
import sys, glob, json
from pathlib import Path
import importlib

ROOT = Path(__file__).resolve().parent.parent
OUT = {}

# 1) PIL load of EVERY figure in docs/figures/*.png (+ scenario_impact_all_three_FIXED.png)
from PIL import Image
fig_dir = ROOT / "docs" / "figures"
pngs = sorted(fig_dir.glob("*.png"))
fig_results = {}
for p in pngs:
    try:
        im = Image.open(p)
        im.verify()
        im2 = Image.open(p)
        fig_results[p.name] = {"opens": True, "size": list(im2.size), "bytes": p.stat().st_size}
    except Exception as e:
        fig_results[p.name] = {"opens": False, "error": str(e), "bytes": p.stat().st_size}
OUT["pil_all_figures"] = fig_results

# 2) requirements.txt declares esda / libpysal / geopandas (with versions)
req = (ROOT / "requirements.txt").read_text()
hits = {}
for pkg in ["esda", "libpysal", "geopandas", "shap", "lightgbm", "xgboost", "catboost", "scikit-learn", "statsmodels", "pytz", "pillow"]:
    line = next((ln for ln in req.splitlines() if ln.lower().startswith(pkg.lower() + "==")), None)
    hits[pkg] = line
OUT["requirements_declared_packages"] = hits

# 3) no .tex / .docx manuscript anywhere in repo (exclude venv/.venv)
tex = [str(p) for p in Path(ROOT).rglob("*.tex") if ".venv" not in str(p) and "venv" not in str(p)]
docx = [str(p) for p in Path(ROOT).rglob("*.docx") if ".venv" not in str(p) and "venv" not in str(p)]
pdf = [str(p) for p in Path(ROOT).rglob("*.pdf") if ".venv" not in str(p) and "venv" not in str(p)]
OUT["no_tex"] = (len(tex) == 0)
OUT["no_docx"] = (len(docx) == 0)
OUT["no_pdf_in_repo"] = (len(pdf) == 0)
OUT["tex_files"] = tex
OUT["docx_files"] = docx
OUT["pdf_files_outside_venv"] = pdf

# 4) fresh import presence of esda/libpysal/geopandas (proves requirements installable + importable)
for m in ["esda", "libpysal", "geopandas"]:
    try:
        mod = importlib.import_module(m)
        OUT.setdefault("fresh_imports", {})[m] = getattr(mod, "__version__", "ok")
    except Exception as e:
        OUT.setdefault("fresh_imports", {})[m] = "FAIL: " + repr(e)

OUT["print_done"] = True
(ROOT / "results" / "_truth_capture" / "crosscut.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print("WROTE results/_truth_capture/crosscut.json")
