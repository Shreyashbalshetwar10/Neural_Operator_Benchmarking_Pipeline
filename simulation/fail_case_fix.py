r"""
fix_failed_cases.py
-------------------
Run on Windows PowerShell after diagnose_results.py.
Reruns cases that failed, with geometry-aware solver settings.

Reads fix lists produced by diagnose_results.py and reruns
each case with more conservative settings appropriate for
the failure mode and geometry family.

Usage:
    cd C:\Users\Shreyash\cfd_project
    python fix_failed_cases.py --mode no_result
    python fix_failed_cases.py --mode unconverged
    python fix_failed_cases.py --mode no_mesh
"""

import ansys.fluent.core as pyfluent
import argparse
import math
import os
import sys
import glob
import time
import signal
import atexit
import logging
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_DIR  = Path(__file__).resolve().parent.parent
MESH_DIR     = PROJECT_DIR / "simulation" / "meshes" / "fluent"
RESULTS_DIR  = PROJECT_DIR / "simulation" / "results"
GMSH_DIR     = PROJECT_DIR / "simulation" / "meshes" / "gmsh"
CSV_PATH     = PROJECT_DIR / "simulation" / "cfd_dataset_cases.csv"
DONE_FILE    = RESULTS_DIR / "completed_cases.txt"
LOG_FILE     = RESULTS_DIR / "fix_failed_cases.log"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["no_result", "unconverged", "no_mesh", "all"],
    default="all",
    help="Which fix list to process"
)
args = parser.parse_args()

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        ),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# Fluent cleanup
# ============================================================

solver_handle = [None]

def kill_stale_fluent():
    for proc in ["fluent.exe", "cortex.exe", "fl_mpi.exe"]:
        subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)
    temp_dir = os.environ.get("LOCALAPPDATA", "") + "\\Temp"
    for f in glob.glob(os.path.join(temp_dir, "serverinfo-*.txt")):
        try: os.remove(f)
        except: pass
    for f in glob.glob(str(PROJECT_DIR / "cleanup-fluent-*.bat")):
        try: os.remove(f)
        except: pass

def safe_exit(signum=None, frame=None):
    log.info("Interrupt - shutting down...")
    if solver_handle[0]:
        try: solver_handle[0].exit()
        except: kill_stale_fluent()
    sys.exit(0)

atexit.register(safe_exit)
signal.signal(signal.SIGINT,  safe_exit)
signal.signal(signal.SIGTERM, safe_exit)

# ============================================================
# Solver settings per geometry family
# ============================================================

FAMILY_SETTINGS = {
    "circle": {
        "mom": 0.3, "pressure": 0.2,
        "n_iter_1st": 0,      # no first-order stage
        "n_iter_2nd": 300,
    },
    "ellipse": {
        "mom": 0.3, "pressure": 0.2,
        "n_iter_1st": 0,
        "n_iter_2nd": 300,
    },
    "airfoil": {
        "mom": 0.3, "pressure": 0.2,
        "n_iter_1st": 0,
        "n_iter_2nd": 400,
    },
    "square": {
        "mom": 0.25, "pressure": 0.15,
        "n_iter_1st": 150,    # first order for stability
        "n_iter_2nd": 300,
    },
    "triangle": {
        "mom": 0.2, "pressure": 0.15,
        "n_iter_1st": 200,
        "n_iter_2nd": 400,
    },
    "diamond": {
        "mom": 0.2, "pressure": 0.15,
        "n_iter_1st": 200,
        "n_iter_2nd": 500,    # more iterations for sharp tips
    },
}

# ============================================================
# Launch Fluent with retry
# ============================================================

def launch_fluent_with_retry(max_attempts=3, wait_between=25):
    for attempt in range(1, max_attempts + 1):
        try:
            kill_stale_fluent()
            time.sleep(wait_between)
            solver = pyfluent.launch_fluent(
                product_version="25.2.0",
                mode=pyfluent.FluentMode.SOLVER,
                ui_mode=pyfluent.UIMode.NO_GUI,
                dimension=pyfluent.Dimension.THREE,
                precision=pyfluent.Precision.DOUBLE,
                processor_count=1,
                start_timeout=300,
            )
            return solver
        except Exception as e:
            log.warning(f"  Launch attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            log.info(f"  Retrying in 30 seconds...")
            kill_stale_fluent()
            time.sleep(30)

# ============================================================
# Run a single case
# ============================================================

def run_case(solver, row, mesh_file):

    case_id = int(row.case_id)
    family  = row.geometry_family
    re      = float(row.Re)
    aoa_deg = float(row.AoA)
    param   = float(row.geometry_parameter)

    vx = math.cos(math.radians(aoa_deg))
    vy = math.sin(math.radians(aoa_deg))
    mu = 1.0 / re

    settings = FAMILY_SETTINGS.get(family, FAMILY_SETTINGS["circle"])

    # Read mesh
    solver.settings.file.read_mesh(file_name=str(mesh_file))

    # Reference values
    rv = solver.settings.setup.reference_values
    rv.area.set_state(0.1)
    rv.density.set_state(1.0)
    rv.velocity.set_state(1.0)
    rv.length.set_state(1.0)
    rv.pressure.set_state(0.0)

    # Zone types
    bc = solver.settings.setup.boundary_conditions
    bc.set_zone_type(zone_list=["inlet"],       new_type="velocity-inlet")
    bc.set_zone_type(zone_list=["outlet"],      new_type="pressure-outlet")
    bc.set_zone_type(zone_list=["symmetry"],    new_type="symmetry")
    bc.set_zone_type(zone_list=["wall_object"], new_type="wall")
    bc.set_zone_type(zone_list=["front"],       new_type="symmetry")
    bc.set_zone_type(zone_list=["back"],        new_type="symmetry")

    # Solver
    solver.settings.setup.general.solver.type = "pressure-based"
    solver.settings.setup.general.solver.time = "steady"
    solver.settings.setup.models.viscous.model = "laminar"

    # Material
    air = solver.settings.setup.materials.fluid["air"]
    air.density.option   = "constant"
    air.density.value    = 1.0
    air.viscosity.option = "constant"
    air.viscosity.value  = mu

    # BCs
    bc.velocity_inlet["inlet"].momentum.set_state({
        "velocity_specification_method": "Magnitude and Direction",
        "velocity_magnitude": {"value": 1.0},
        "flow_direction": [vx, vy, 0.0],
        "coordinate_system": "Cartesian (X, Y, Z)",
    })
    bc.pressure_outlet["outlet"].momentum.gauge_pressure.value = 0.0

    # Solution methods
    solver.settings.solution.methods.p_v_coupling.flow_scheme = "SIMPLE"

    urf = solver.settings.solution.controls.under_relaxation
    urf["mom"]      = settings["mom"]
    urf["pressure"] = settings["pressure"]

    # Convergence monitors
    res = solver.settings.solution.monitor.residual.equations
    for eq in ["continuity", "x-velocity", "y-velocity", "z-velocity"]:
        res[eq].check_convergence = True
    res["continuity"].absolute_criteria = 1e-3
    res["x-velocity"].absolute_criteria = 1e-4
    res["y-velocity"].absolute_criteria = 1e-4
    res["z-velocity"].absolute_criteria = 1e-4

    # Initialize
    solver.settings.solution.initialization.hybrid_initialize()

    # Stage 1: first order (only for sharp geometries)
    if settings["n_iter_1st"] > 0:
        solver.tui.solve.set.discretization_scheme("mom", 0)
        solver.settings.solution.run_calculation.iterate(
            iter_count=settings["n_iter_1st"]
        )
        # Switch to second order
        solver.tui.solve.set.discretization_scheme("mom", 1)

    # Stage 2: main solve
    solver.settings.solution.run_calculation.iterate(
        iter_count=settings["n_iter_2nd"]
    )

    # Save
    case_file = RESULTS_DIR / f"case_{case_id}_{family}.cas.h5"
    solver.settings.file.write_case_data(file_name=str(case_file))

    log.info(f"  -> Saved: {case_file.name}")

# ============================================================
# Load fix lists
# ============================================================

fix_dfs = []

if args.mode in ["no_result", "all"]:
    p = RESULTS_DIR / "fix_no_result.csv"
    if p.exists():
        fix_dfs.append(pd.read_csv(p))
        log.info(f"Loaded no_result list: {len(fix_dfs[-1])} cases")

if args.mode in ["unconverged", "all"]:
    p = RESULTS_DIR / "fix_unconverged.csv"
    if p.exists():
        fix_dfs.append(pd.read_csv(p))
        log.info(f"Loaded unconverged list: {len(fix_dfs[-1])} cases")

if args.mode in ["no_mesh", "all"]:
    p = RESULTS_DIR / "fix_no_mesh.csv"
    if p.exists():
        log.warning(
            "no_mesh cases require mesh regeneration in WSL first. "
            "Run generate_all_meshes.py in WSL, then rerun this script."
        )

if not fix_dfs:
    log.info("No fix lists found. Run diagnose_results.py first.")
    sys.exit(0)

fix_df = pd.concat(fix_dfs, ignore_index=True).drop_duplicates(subset="case_id")
log.info(f"Total cases to fix: {len(fix_df)}")

completed = set()
if DONE_FILE.exists():
    completed = set(DONE_FILE.read_text().splitlines())

fix_key = lambda row: f"{int(row.case_id)}_{row.geometry_family}_fixed"

# ============================================================
# Main fix loop
# ============================================================

kill_stale_fluent()
failed = []

for i, row in enumerate(fix_df.itertuples()):
    case_id = int(row.case_id)
    family  = row.geometry_family
    key     = fix_key(row)

    if key in completed:
        log.info(f"[{i+1:3d}/{len(fix_df)}] case {case_id:4d} {family:10s} -- already fixed, skipping")
        continue

    mesh_file = MESH_DIR / f"mesh_{case_id}_{family}.msh"
    if not mesh_file.exists():
        log.warning(f"[{i+1:3d}/{len(fix_df)}] case {case_id:4d} {family:10s} -- mesh missing, skipping")
        failed.append(key)
        continue

    log.info(f"[{i+1:3d}/{len(fix_df)}] case {case_id:4d} {family:10s} Re={row.Re:.0f} AoA={row.AoA:.1f}")

    solver = None
    try:
        solver = launch_fluent_with_retry()
        solver_handle[0] = solver

        run_case(solver, row, mesh_file)

        # Mark as fixed
        with open(DONE_FILE, "a") as f:
            f.write(key + "\n")
        completed.add(key)

    except Exception as e:
        log.error(f"  -> FAILED: {e}")
        failed.append(key)

    finally:
        if solver is not None:
            try: solver.exit()
            except: kill_stale_fluent()
        solver_handle[0] = None
        time.sleep(25)

# ============================================================
# Summary
# ============================================================

log.info("=" * 60)
log.info(f"Fix run complete")
log.info(f"  Fixed   : {len(fix_df) - len(failed)}")
log.info(f"  Failed  : {len(failed)}")
if failed:
    log.info(f"  Still failing: {failed}")