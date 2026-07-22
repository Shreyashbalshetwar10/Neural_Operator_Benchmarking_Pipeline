r"""
diagnose_results.py
-------------------
Run on Windows PowerShell after run_all_cases.py completes.
Identifies three categories of problem cases:

1. MISSING     — no .dat.h5 file saved (Fluent crashed or mesh missing)
2. UNCONVERGED — .dat.h5 exists but residuals never dropped below threshold
3. BAD_MESH    — mesh file missing entirely

Usage:
    cd C:\Users\Shreyash\cfd_project
    python diagnose_results.py
"""

import os
import re
import glob
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_DIR  = Path(__file__).resolve().parent.parent
MESH_DIR     = PROJECT_DIR / "simulation" / "meshes" / "fluent"
RESULTS_DIR  = PROJECT_DIR / "simulation" / "results"
CSV_PATH     = PROJECT_DIR / "simulation" / "cfd_dataset_cases.csv"
DONE_FILE    = RESULTS_DIR / "completed_cases.txt"
DIAG_CSV     = RESULTS_DIR / "diagnosis.csv"
# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(CSV_PATH)
print(f"Total cases in dataset : {len(df)}")

# Load completed set
completed = set()
if DONE_FILE.exists():
    completed = set(DONE_FILE.read_text().splitlines())
print(f"Marked as completed    : {len(completed)}")

# ============================================================
# Check each case
# ============================================================

results = []

for row in df.itertuples():
    case_id = int(row.case_id)
    family  = row.geometry_family
    re      = float(row.Re)
    aoa     = float(row.AoA)
    param   = float(row.geometry_parameter)
    key     = f"{case_id}_{family}"

    mesh_file = MESH_DIR  / f"mesh_{case_id}_{family}.msh"
    cas_file  = RESULTS_DIR / f"case_{case_id}_{family}.cas.h5"
    dat_file  = RESULTS_DIR / f"case_{case_id}_{family}.dat.h5"

    mesh_ok = mesh_file.exists()

    cas_ok = cas_file.exists()
    dat_ok = dat_file.exists()

    converged  = None
    final_cont = None

    if dat_ok:
        try:
            with h5py.File(dat_file, "r") as f:
                # Residuals are stored under /results/residuals/
                # Try to find continuity residual
                if "results" in f and "residuals" in f["results"]:
                    res_group = f["results"]["residuals"]
                    if "continuity" in res_group:
                        cont_vals = res_group["continuity"][:]
                        final_cont = float(cont_vals[-1]) if len(cont_vals) > 0 else None
                        converged  = final_cont < 1e-2 if final_cont else None
        except Exception as e:
            pass   # h5py can't read it — mark as suspicious

    if not mesh_ok:
        status = "NO_MESH"
    elif not dat_ok:
        status = "NO_RESULT"
    elif converged is False:
        status = "UNCONVERGED"
    elif converged is None:
        status = "UNKNOWN"
    else:
        status = "OK"

    results.append({
        "case_id":            case_id,
        "geometry_family":    family,
        "geometry_parameter": param,
        "Re":                 re,
        "AoA":                aoa,
        "mesh_exists":        mesh_ok,
        "cas_exists":         cas_ok,
        "dat_exists":         dat_ok,
        "final_continuity":   final_cont,
        "converged":          converged,
        "status":             status,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(DIAG_CSV, index=False)

print("\n=== Diagnosis Summary ===")
for status, group in results_df.groupby("status"):
    print(f"  {status:15s} : {len(group):4d} cases")

print(f"\nDetailed diagnosis saved to: {DIAG_CSV}")

print("\n=== Status by Family ===")
pivot = results_df.groupby(["geometry_family", "status"]).size().unstack(fill_value=0)
print(pivot.to_string())

no_mesh      = results_df[results_df["status"] == "NO_MESH"]
no_result    = results_df[results_df["status"] == "NO_RESULT"]
unconverged  = results_df[results_df["status"] == "UNCONVERGED"]
unknown      = results_df[results_df["status"] == "UNKNOWN"]

print(f"\n=== Cases needing action ===")
print(f"  No mesh file   : {len(no_mesh)}")
print(f"  No result file : {len(no_result)}")
print(f"  Unconverged    : {len(unconverged)}")
print(f"  Unknown        : {len(unknown)}")

# Save fix lists
no_mesh.to_csv(RESULTS_DIR / "fix_no_mesh.csv", index=False)
no_result.to_csv(RESULTS_DIR / "fix_no_result.csv", index=False)
unconverged.to_csv(RESULTS_DIR / "fix_unconverged.csv", index=False)

print(f"\nFix lists saved to {RESULTS_DIR}")