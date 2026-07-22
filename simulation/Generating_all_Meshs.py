"""
generate_all_meshes.py
----------------------
Run this on WSL. Generates Gmsh 3D meshes for all 630 cases
and converts each to Fluent format via OpenFOAM.

Usage:
    source /opt/openfoam12/etc/bashrc
    python simulation/generate_all_meshes.py

Output per case:
    simulation/meshes/fluent/mesh_<id>_<family>.msh   ← Fluent-ready mesh
"""

import os
import sys
import subprocess
import shutil
import numpy as np
import pandas as pd
import gmsh
import logging
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_DIR  = Path(__file__).resolve().parent.parent
GEOM_DIR     = PROJECT_DIR / "simulation" / "generated_geometries"
MESH_DIR     = PROJECT_DIR / "simulation" / "meshes"
GMSH_DIR     = MESH_DIR / "gmsh"
FOAM_DIR     = MESH_DIR / "openfoam"
FLUENT_DIR   = MESH_DIR / "fluent"
LOG_FILE     = MESH_DIR / "mesh_generation.log"
CSV_PATH     = PROJECT_DIR / "simulation" / "cfd_dataset_cases.csv"

for d in [GMSH_DIR, FOAM_DIR, FLUENT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
# Physical domain
# ============================================================

X_MIN, X_MAX = -1.0, 5.0
Y_MIN, Y_MAX = -2.0, 2.0
Z_DEPTH      = 0.1
LC_FAR       = 0.15
LC_OBJ       = 0.02

# ============================================================
# OpenFOAM controlDict template
# ============================================================

CONTROL_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

# ============================================================
# Gmsh mesh generation
# ============================================================

def generate_gmsh_mesh(dat_path, gmsh_path, re, family):

    gmsh.initialize()
    gmsh.model.add("cfd_case")
    gmsh.option.setNumber("General.Verbosity", 0)

    # Squares need coarser object mesh — dense corner points cause
    # edge recovery failures in the constrained Delaunay triangulation
    lc_obj = LC_OBJ * 3.0 if family == "square" else LC_OBJ

    # Algorithm 5 = Delaunay (more robust for complex boundaries)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    coords = np.loadtxt(dat_path, delimiter=",")
    xy = coords[:, :2]
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]

    if family == "square":
        xy = xy[::4]   # every 4th point — ~70 points instead of 280

    obj_pts   = [gmsh.model.geo.addPoint(x, y, 0, lc_obj) for x, y in xy]
    n_obj     = len(obj_pts)
    obj_lines = [gmsh.model.geo.addLine(obj_pts[i], obj_pts[(i+1) % n_obj])
                 for i in range(n_obj)]
    obj_loop  = gmsh.model.geo.addCurveLoop(obj_lines)

    p1 = gmsh.model.geo.addPoint(X_MIN, Y_MIN, 0, LC_FAR)
    p2 = gmsh.model.geo.addPoint(X_MAX, Y_MIN, 0, LC_FAR)
    p3 = gmsh.model.geo.addPoint(X_MAX, Y_MAX, 0, LC_FAR)
    p4 = gmsh.model.geo.addPoint(X_MIN, Y_MAX, 0, LC_FAR)
    l_bot = gmsh.model.geo.addLine(p1, p2)
    l_rgt = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_lft = gmsh.model.geo.addLine(p4, p1)
    dom_loop   = gmsh.model.geo.addCurveLoop([l_bot, l_rgt, l_top, l_lft])
    fluid_surf = gmsh.model.geo.addPlaneSurface([dom_loop, obj_loop])

    gmsh.model.geo.synchronize()

    # Boundary layer — skip for squares to avoid edge conflicts
    if family != "square":
        bl = gmsh.model.mesh.field.add("BoundaryLayer")
        gmsh.model.mesh.field.setNumbers(bl, "CurvesList", obj_lines)
        gmsh.model.mesh.field.setNumber(bl,  "Size",       1.0 / re)
        gmsh.model.mesh.field.setNumber(bl,  "Ratio",      1.15)
        gmsh.model.mesh.field.setNumber(bl,  "Quads",      1)
        gmsh.model.mesh.field.setNumber(bl,  "NbLayers",   8)
        gmsh.model.mesh.field.setAsBoundaryLayer(bl)
    else:
        # For squares use a distance-based size field instead
        # This refines near the object without structured BL layers
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", obj_lines)
        thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh, "InField",    dist)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMin",    lc_obj)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMax",    LC_FAR)
        gmsh.model.mesh.field.setNumber(thresh, "DistMin",    0.05)
        gmsh.model.mesh.field.setNumber(thresh, "DistMax",    0.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh)

    # Extrude to 3D
    gmsh.model.geo.extrude(
        [(2, fluid_surf)],
        0, 0, Z_DEPTH,
        [1], [1.0], True
    )

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # ... surface classification and physical groups unchanged ...

    # Classify surfaces
    all_surfs = [tag for dim, tag in gmsh.model.getEntities(2)]
    vol_tags  = [tag for dim, tag in gmsh.model.getEntities(3)]

    tol = 0.01
    inlet_s = []; outlet_s = []; sym_s = []; wall_s = []; front_s = []; back_s = []

    for stag in all_surfs:
        xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, stag)
        cx = (xmn + xmx) / 2
        cy = (ymn + ymx) / 2
        if zmx - zmn < tol:
            (back_s if abs(zmn) < tol else front_s).append(stag)
        elif xmx - xmn < tol and abs(cx - X_MIN) < tol:
            inlet_s.append(stag)
        elif xmx - xmn < tol and abs(cx - X_MAX) < tol:
            outlet_s.append(stag)
        elif ymx - ymn < tol and (abs(cy - Y_MIN) < tol or abs(cy - Y_MAX) < tol):
            sym_s.append(stag)
        else:
            wall_s.append(stag)

    gmsh.model.addPhysicalGroup(3, vol_tags,  name="fluid")
    gmsh.model.addPhysicalGroup(2, inlet_s,   name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet_s,  name="outlet")
    gmsh.model.addPhysicalGroup(2, sym_s,     name="symmetry")
    gmsh.model.addPhysicalGroup(2, wall_s,    name="wall_object")
    gmsh.model.addPhysicalGroup(2, front_s,   name="front")
    gmsh.model.addPhysicalGroup(2, back_s,    name="back")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(gmsh_path))
    gmsh.finalize()


# ============================================================
# OpenFOAM conversion
# ============================================================

def convert_to_fluent(gmsh_path, fluent_path, case_id):
    """Convert Gmsh mesh to Fluent format via OpenFOAM."""

    foam_case = FOAM_DIR / f"case_{case_id}"

    # Clean and create case dir
    if foam_case.exists():
        shutil.rmtree(foam_case)
    (foam_case / "constant" / "polyMesh").mkdir(parents=True)
    (foam_case / "system").mkdir(parents=True)
    (foam_case / "0").mkdir(parents=True)

    # Write controlDict
    (foam_case / "system" / "controlDict").write_text(CONTROL_DICT)

    # Run gmshToFoam
    result = subprocess.run(
        ["gmshToFoam", str(gmsh_path)],
        cwd=str(foam_case),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"gmshToFoam failed:\n{result.stderr}")

    # Run foamMeshToFluent
    result = subprocess.run(
        ["foamMeshToFluent"],
        cwd=str(foam_case),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"foamMeshToFluent failed:\n{result.stderr}")

    # Copy result to fluent dir
    foam_msh = foam_case / "fluentInterface" / f"{foam_case.name}.msh"
    if not foam_msh.exists():
        # Try alternate name
        candidates = list((foam_case / "fluentInterface").glob("*.msh"))
        if not candidates:
            raise RuntimeError(f"No .msh file in fluentInterface for case {case_id}")
        foam_msh = candidates[0]

    shutil.copy(foam_msh, fluent_path)

    # Clean up OpenFOAM case to save disk space
    shutil.rmtree(foam_case)


# ============================================================
# Main loop
# ============================================================

df = pd.read_csv(CSV_PATH)

# Track progress
done_file = MESH_DIR / "completed_meshes.txt"
completed = set()
if done_file.exists():
    completed = set(done_file.read_text().splitlines())

log.info(f"Total cases: {len(df)}")
log.info(f"Already completed: {len(completed)}")

failed = []

for row in df.itertuples():
    case_id = int(row.case_id)
    family  = row.geometry_family
    re      = float(row.Re)

    key = f"{case_id}_{family}"

    if key in completed:
        continue

    dat_path    = GEOM_DIR / f"ansys_curve_{case_id}_{family}.dat"
    gmsh_path   = GMSH_DIR / f"mesh_{case_id}_{family}.msh"
    fluent_path = FLUENT_DIR / f"mesh_{case_id}_{family}.msh"

    # Skip if Fluent mesh already exists
    if fluent_path.exists():
        log.info(f"[{case_id:4d}/{len(df)}] {family:10s} — already exists, skipping")
        with open(done_file, "a") as f:
            f.write(key + "\n")
        completed.add(key)
        continue

    if not dat_path.exists():
        log.warning(f"[{case_id:4d}] {family} — .dat file not found, skipping")
        failed.append(key)
        continue

    try:
        log.info(f"[{case_id:4d}/{len(df)}] {family:10s} Re={re:.0f} — generating mesh...")

        generate_gmsh_mesh(dat_path, gmsh_path, re, family)
        convert_to_fluent(gmsh_path, fluent_path, case_id)

        # Clean up Gmsh file to save space
        gmsh_path.unlink(missing_ok=True)

        log.info(f"[{case_id:4d}/{len(df)}] {family:10s} — done → {fluent_path.name}")

        # Mark as completed
        with open(done_file, "a") as f:
            f.write(key + "\n")
        completed.add(key)

    except Exception as e:
        log.error(f"[{case_id:4d}] {family} — FAILED: {e}")
        failed.append(key)

# ============================================================
# Summary
# ============================================================

log.info("=" * 60)
log.info(f"Mesh generation complete")
log.info(f"  Completed : {len(completed)}")
log.info(f"  Failed    : {len(failed)}")
if failed:
    log.info(f"  Failed cases: {failed}")