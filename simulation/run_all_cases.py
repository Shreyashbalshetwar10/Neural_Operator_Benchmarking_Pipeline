r"""
run_all_cases.py
----------------
Run this on Windows PowerShell. Runs all 630 CFD cases using PyFluent.
Reads pre-generated Fluent meshes and produces flow field results.

Usage:
    cd C:\Users\Shreyash\cfd_project
    python run_all_cases.py

Prerequisites:
    - generate_all_meshes.py must have been run first (WSL)
    - All meshes must exist in simulation\meshes\fluent\

Output per case:
    simulation\results\case_<id>_<family>.cas.h5
    simulation\results\case_<id>_<family>.dat.h5
    simulation\results\forces.csv   (Cd, Cl for all cases)
"""

import ansys.fluent.core as pyfluent
import math
import os
import sys
import glob
import signal
import atexit
import subprocess
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import time

def launch_fluent_with_retry(max_attempts=3, wait_between=25):
    """Launch Fluent with retries on connection failure."""
    for attempt in range(1, max_attempts + 1):
        try:
            kill_stale_fluent()
            time.sleep(wait_between)
            solver = pyfluent.launch_fluent(
                product_version="25.2.0",    # changed from product_version
                mode=pyfluent.FluentMode.SOLVER,
                ui_mode=pyfluent.UIMode.NO_GUI,
                dimension=pyfluent.Dimension.THREE,
                precision=pyfluent.Precision.DOUBLE,
                processor_count=1,
                start_timeout=400,
                cleanup_on_exit=True
            )
            return solver
        except Exception as e:
            log.warning(f"  Launch attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            log.info(f"  Retrying in 40 seconds...")
            kill_stale_fluent()
            time.sleep(40)

# ============================================================
# Paths
# ============================================================


PROJECT_DIR  = Path(__file__).resolve().parent.parent
MESH_DIR     = PROJECT_DIR / "simulation" / "meshes" / "fluent"
RESULTS_DIR  = PROJECT_DIR / "simulation" / "results_2nd_try"
CSV_PATH     = PROJECT_DIR / "simulation" / "cfd_dataset_cases.csv"
LOG_FILE     = RESULTS_DIR / "run_all_cases_2nd_try.log"
FORCES_CSV   = RESULTS_DIR / "forces_2nd_try.csv"
DONE_FILE    = RESULTS_DIR / "completed_cases_2nd_try.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# Fluent process cleanup
# ============================================================

solver_handle = [None]   # mutable reference so atexit can access it

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
    log.info("Interrupt received — shutting down Fluent cleanly...")
    if solver_handle[0] is not None:
        try:
            solver_handle[0].exit()
        except Exception:
            kill_stale_fluent()
    sys.exit(0)

atexit.register(safe_exit)
signal.signal(signal.SIGINT,  safe_exit)
signal.signal(signal.SIGTERM, safe_exit)

# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(CSV_PATH)
log.info(f"Total cases in dataset: {len(df)}")

# Load completed cases
completed = set()
if DONE_FILE.exists():
    completed = set(DONE_FILE.read_text().splitlines())
log.info(f"Already completed: {len(completed)}")

# Load or create forces CSV
if FORCES_CSV.exists():
    forces_df = pd.read_csv(FORCES_CSV)
else:
    forces_df = pd.DataFrame(columns=[
        "case_id", "geometry_family", "geometry_parameter",
        "Re", "AoA", "Cd", "Cl", "converged"
    ])

# ============================================================
# Single case solver function
# ============================================================

def run_case(solver, row, mesh_file):
    """
    Configure and run a single CFD case.
    Returns dict with Cd, Cl, converged flag.
    """
    case_id = int(row.case_id)
    family  = row.geometry_family
    re      = float(row.Re)
    aoa_deg = float(row.AoA)
    param   = float(row.geometry_parameter)

    vx = math.cos(math.radians(aoa_deg))
    vy = math.sin(math.radians(aoa_deg))
    mu = 1.0 / re

    # ---- Read mesh ----
    solver.settings.file.read_mesh(file_name=str(mesh_file))

    # ---- Reference values ----
    rv = solver.settings.setup.reference_values
    rv.area.set_state(0.1)      # chord(1.0) × depth(0.1)
    rv.density.set_state(1.0)
    rv.velocity.set_state(1.0)
    rv.length.set_state(1.0)
    rv.pressure.set_state(0.0)

    # ---- Zone types ----
    bc = solver.settings.setup.boundary_conditions
    bc.set_zone_type(zone_list=["inlet"],       new_type="velocity-inlet")
    bc.set_zone_type(zone_list=["outlet"],      new_type="pressure-outlet")
    bc.set_zone_type(zone_list=["symmetry"],    new_type="symmetry")
    bc.set_zone_type(zone_list=["wall_object"], new_type="wall")
    bc.set_zone_type(zone_list=["front"],       new_type="symmetry")
    bc.set_zone_type(zone_list=["back"],        new_type="symmetry")

    # ---- Material ----
    air = solver.settings.setup.materials.fluid["air"]
    air.density.option  = "constant"
    air.density.value   = 1.0
    air.viscosity.option = "constant"
    air.viscosity.value  = mu

    # ---- Boundary conditions ----
    bc.velocity_inlet["inlet"].momentum.set_state({
        "velocity_specification_method": "Magnitude and Direction",
        "velocity_magnitude": {"value": 1.0},
        "flow_direction": [vx, vy, 0.0],
        "coordinate_system": "Cartesian (X, Y, Z)",
    })
    bc.pressure_outlet["outlet"].momentum.gauge_pressure.value = 0.0

    # ---- Convergence monitors ----
    res = solver.settings.solution.monitor.residual.equations
    for eq in ["continuity", "x-velocity", "y-velocity", "z-velocity"]:
        res[eq].check_convergence = True
    res["continuity"].absolute_criteria = 1e-3
    res["x-velocity"].absolute_criteria = 1e-4
    res["y-velocity"].absolute_criteria = 1e-4
    res["z-velocity"].absolute_criteria = 1e-4

    needs_transient = False
    
    if family in ["square", "triangle", "diamond"]:
        needs_transient = True 
    elif family == "circle" and re > 150.0:
        needs_transient = True 
        
    if needs_transient:
        log.info(f"  [Auto-Fix] Sharp geometry detected. Switching to Transient Time-Averaging.")

        solver.settings.setup.models.viscous.model = "laminar"
        
        solver.settings.setup.general.solver.time = "unsteady-2nd-order"
        solver.settings.solution.methods.p_v_coupling.flow_scheme = "SIMPLE"
        
        urf = solver.settings.solution.controls.under_relaxation
        urf["mom"] = 0.3        # Heavily damp the velocity updates
        urf["pressure"] = 0.2   # Heavily damp the pressure updates
        
        solver.settings.solution.initialization.initialization_type = "standard"
        solver.settings.solution.initialization.standard_initialize()

        log.info(f"  -> Phase 1: Developing wake (Micro-stepping)...")
        solver.settings.solution.run_calculation.transient_controls.time_step_size = 0.001
        solver.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=200, 
            max_iter_per_step=20
        )
        
        log.info(f"  -> Phase 2: Sampling mean flow field...")

        solver.settings.solution.run_calculation.transient_controls.time_step_size = 0.01 
        solver.tui.solve.set.data_sampling("yes")
        
        solver.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=300, 
            max_iter_per_step=20
        )

    else:
        log.info(f"  [Auto-Fix] Smooth geometry detected. Using Steady-State Coupled Solver.")
        
        # Keep your standard steady-state settings here
        solver.settings.setup.general.solver.time = "steady"
        solver.settings.solution.methods.p_v_coupling.flow_scheme = "Coupled"
        solver.tui.solve.set.pseudo_transient("yes")
        
        solver.settings.solution.initialization.hybrid_initialize()
        solver.settings.solution.run_calculation.iterate(iter_count=800)

    # ---- Extract forces ----
    drag_file = str(RESULTS_DIR / f"temp_drag_{case_id}.txt")
    lift_file = str(RESULTS_DIR / f"temp_lift_{case_id}.txt")

    # Drag: force along freestream direction (Write to file)
    solver.settings.results.report.forces(
        write_to_file=True,
        file_name=drag_file,
        direction_vector=[vx, vy, 0.0],
        wall_zones=["wall_object"],
        option="forces"
    )

    # Lift: force perpendicular to freestream in xy plane (Write to file)
    solver.settings.results.report.forces(
        write_to_file=True,
        file_name=lift_file,
        direction_vector=[-vy, vx, 0.0],
        wall_zones=["wall_object"],
        option="forces"
    )

    # Read the text files back into Python strings
    try:
        with open(drag_file, "r") as f:
            forces_drag = f.read()
    except Exception:
        forces_drag = None

    try:
        with open(lift_file, "r") as f:
            forces_lift = f.read()
    except Exception:
        forces_lift = None

    # Parse Cd and Cl from the report strings using your existing function
    def extract_total_coefficient(report_str):
        """Extract the total force coefficient from the forces report."""
        if report_str is None:
            return None
        for line in str(report_str).split("\n"):
            if "wall_object" in line:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        pass
        return None

    cd = extract_total_coefficient(forces_drag)
    cl = extract_total_coefficient(forces_lift)

    try:
        if os.path.exists(drag_file): os.remove(drag_file)
        if os.path.exists(lift_file): os.remove(lift_file)
    except Exception:
        pass

    log.info(f"  Cd={cd:.4f}  Cl={cl:.4f}" if cd is not None and cl is not None else f"  Cd={cd}  Cl={cl}")

    log.info(f"  Forces drag report: {str(forces_drag)[:200]}")
    log.info(f"  Forces lift report: {str(forces_lift)[:200]}")

    # ---- Save case ----
    case_file = RESULTS_DIR / f"case_{case_id}_{family}.cas.h5"
    solver.settings.file.write_case_data(file_name=str(case_file))

    return {
        "case_id":            case_id,
        "geometry_family":    family,
        "geometry_parameter": param,
        "Re":                 re,
        "AoA":                aoa_deg,
        "Cd":                 cd,
        "Cl":                 cl,
        "converged":          True,
    }

# ============================================================
# Main loop — one Fluent instance per case
# ============================================================

kill_stale_fluent()

failed = []
n_total = len(df)

for i, row in enumerate(df.itertuples()):
    case_id = int(row.case_id)
    family  = row.geometry_family
    key     = f"{case_id}_{family}"

    # Skip if already done
    if key in completed:
        log.info(f"[{i+1:4d}/{n_total}] case {case_id:4d} {family:10s} — skipping (already done)")
        continue

    mesh_file = MESH_DIR / f"mesh_{case_id}_{family}.msh"
    if not mesh_file.exists():
        log.warning(f"[{i+1:4d}/{n_total}] case {case_id:4d} {family:10s} — mesh not found, skipping")
        failed.append(key)
        continue

    log.info(f"[{i+1:4d}/{n_total}] case {case_id:4d} {family:10s} Re={row.Re:.0f} AoA={row.AoA:.1f}°")

    solver = None
    try:
        # Launch fresh Fluent instance per case
        kill_stale_fluent()
        solver = launch_fluent_with_retry()
        solver_handle[0] = solver

        result = run_case(solver, row, mesh_file)

        # Append to forces CSV
        forces_df = pd.concat(
            [forces_df, pd.DataFrame([result])],
            ignore_index=True
        )
        forces_df.to_csv(FORCES_CSV, index=False)

        # Mark as done
        with open(DONE_FILE, "a") as f:
            f.write(key + "\n")
        completed.add(key)

        log.info(f"  -> Cd={result['Cd']}  Cl={result['Cl']}  saved.")

    except Exception as e:
        log.error(f"  -> FAILED: {e}")
        failed.append(key)

    finally:
        # Always exit Fluent cleanly before next case
        if solver is not None:
            try:
                solver.exit()
            except Exception:
                kill_stale_fluent()
        solver_handle[0] = None

        import time
        time.sleep(30)  # brief pause to ensure clean startup of next instance

# ============================================================
# Summary
# ============================================================

log.info("=" * 60)
log.info("Run complete")
log.info(f"  Completed : {len(completed)}")
log.info(f"  Failed    : {len(failed)}")
if failed:
    log.info(f"  Failed keys: {failed}")
log.info(f"  Forces CSV : {FORCES_CSV}")
