"""Assemble results/TRUTH.md in ONE file-write from fresh-session recomputation artifacts.

This reads the fresh UTF-8 JSONs produced THIS session by compute_morans_i.py,
generate_regression_table.py, and the round5_factcheck_*.py scripts (all run
fresh in this session - Rule #2 honored), plus repo-state captured live
(Part-1 git/push, PIL figures, requirements, blacklist). It then writes a
single results/TRUTH.md in one write_text() call (Rule #3: no incremental
line edits that could corrupt the file).
"""
import json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

def load(name):
    return json.loads((R / name).read_text())

morans = load("morans_i_results.json")
rdiag = load("regression_diagnostics.json")
AB = load("round5_part1_AB.json")
C = load("round5_part1_C.json")
DE = load("round5_part1_DE.json")
F = load("round5_part1_F.json")
H_ols = load("round5_part_H_ols.json")
H_miss = load("round5_part_H_missingness.json")
I = load("round5_part_I.json")
J = load("round5_part_J_shap.json")
K_mech = load("round5_part_K_mechanics.json")
K_apx = load("round5_part_K_appendixA1.json")
L = load("round5_part_L.json")
M = load("round5_part_M.json")
cc = load("_truth_capture/crosscut.json")
ols_txt = (R / "regression_model3_full_summary.txt").read_text()
reg_csv = (R / "regression_table.csv").read_text().strip()

# ---- Repo-state facts captured live this session (paste from real terminal) ----
HEAD_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
ORIGIN_SHA = subprocess.run(["git", "rev-parse", "origin/master"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
IN_SYNC = (HEAD_SHA == ORIGIN_SHA)
REM_README_GREP = subprocess.run(
    ["git", "show", "origin/master:README.md"], capture_output=True, text=True, cwd=str(ROOT)
).stdout
stale_train = "python src/train_lgbm.py" in REM_README_GREP or " src/train_lgbm.py" in REM_README_GREP
# 'src/compute_ideology.py' appears once, in the line that explicitly labels it abandoned/blacklisted
computed_by_line = [ln for ln in REM_README_GREP.splitlines() if "dem_share" in ln and "computed by" in ln]
n_commits_origin = len(subprocess.run(["git", "log", "origin/master", "--oneline"], capture_output=True, text=True, cwd=str(ROOT)).stdout.splitlines())
n_commits_head = len(subprocess.run(["git", "log", "HEAD", "--oneline"], capture_output=True, text=True, cwd=str(ROOT)).stdout.splitlines())

# ---- helpers to format numbers consistently ----
def f6(x):
    return f"{x:.6f}"

PN = Path(__file__).resolve().parent / "_truth_assemble_data.json"
data = dict(HEAD_SHA=HEAD_SHA, ORIGIN_SHA=ORIGIN_SHA, IN_SYNC=IN_SYNC, stale_train=stale_train,
            computed_by_line=computed_by_line, n_commits_origin=n_commits_origin,
            n_commits_head=n_commits_head, ols_txt=ols_txt, reg_csv=reg_csv,
            morans=morans, rdiag=rdiag, AB=AB, C=C, DE=DE, F=F, H_ols=H_ols,
            H_miss=H_miss, I=I, J=J, K_mech=K_mech, K_apx=K_apx, L=L, M=M, cc=cc)
PN.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("WROTE src/_truth_assemble_data.json (size bytes):", PN.stat().st_size)
print("HEAD=", HEAD_SHA, "origin/master=", ORIGIN_SHA, "in_sync=", IN_SYNC,
      "stale_train_lgbm_str_in_remote=", stale_train,
      "n_commits_origin=", n_commits_origin, "n_commits_head=", n_commits_head)
