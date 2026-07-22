import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from skimage import measure
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
from scipy.ndimage import shift as ndimage_shift
 
np.random.seed(42)
random.seed(42)
 
# ============================================================
# PHYSICAL DOMAIN DEFINITION
# All geometry is generated and exported in physical units.
# The object has characteristic length CHORD = 1.0.
# The CFD domain follows the standard external-aero convention:
#   1L upstream | object | 5L downstream | ±2L cross-stream
#
# Physical coordinate origin is at the object centroid.
# ============================================================
 
CHORD        = 1.0
INLET_DIST   = 1.0   # upstream  clearance (×CHORD)
OUTLET_DIST  = 5.0   # downstream clearance (×CHORD)
SIDE_DIST    = 2.0   # half-height clearance (×CHORD)
 
DOMAIN_X_MIN = -INLET_DIST  * CHORD   # -1.0
DOMAIN_X_MAX =  OUTLET_DIST * CHORD   #  5.0
DOMAIN_Y_MIN = -SIDE_DIST   * CHORD   # -2.0
DOMAIN_Y_MAX =  SIDE_DIST   * CHORD   #  2.0
 
DOMAIN_W = DOMAIN_X_MAX - DOMAIN_X_MIN   # 6.0
DOMAIN_H = DOMAIN_Y_MAX - DOMAIN_Y_MIN   # 4.0
 
# ============================================================
# PIXEL GRID — used ONLY for neural-network input encoding.
# The grid spans the full physical domain, so the 1L/5L layout
# is implicitly encoded in the pixel positions.
# 128×128 is fine here because this is a sampling grid, not
# the simulation domain (same approach as DeepCFD).
# ============================================================
 
N = 128
 
 
# ============================================================
# UTILITY
# ============================================================
 
def _scale_to_chord(x, y, chord=CHORD):
    """
    Scale boundary coordinates so the characteristic length
    (max bounding-box side) equals `chord`. Centre is set to the
    bounding-box midpoint (not the point mean, which is biased by
    the repeated closing vertex in closed polygons).
    """
    cx = (x.max() + x.min()) / 2
    cy = (y.max() + y.min()) / 2
    x  = x - cx
    y  = y - cy
    L  = max(x.max() - x.min(), y.max() - y.min())
    scale = chord / (L + 1e-12)
    return x * scale, y * scale
 
 
# ============================================================
# PHYSICAL GEOMETRY GENERATORS
# Each returns (x, y, meta) where x/y are closed boundary
# coordinate arrays in physical units centred at the origin
# with characteristic length = CHORD.
#
# `param` is the geometry_parameter from LHS sampling, range [0, 1].
# It is the sole deterministic shape descriptor for each family.
#
# NO geometry rotation is applied here.  Angle of attack is a
# CFD boundary condition (inlet velocity direction) defined by
# the `AoA` column in the dataset CSV and applied at the solver
# level in ANSYS.  Rotating geometry coordinates independently
# would corrupt the AoA label and make it uninterpretable.
# All geometries are exported in their canonical orientation:
#   - airfoil  : chord along +x axis, TE at right
#   - ellipse  : major axis along +x axis
#   - square   : sides axis-aligned
#   - triangle : apex pointing up (+y)
#   - diamond  : tips on ±x axis
# ============================================================
 
def circle_coords(param):
    # Circle has no shape DOF — param unused but accepted for consistent API.
    n    = 300
    t    = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x    = np.cos(t)
    y    = np.sin(t)
    x, y = _scale_to_chord(x, y)
    x    = np.append(x, x[0]);  y = np.append(y, y[0])
    meta = {"smooth": True, "shape": "generic_smooth"}
    return x, y, meta
 
 
def ellipse_coords(param):
    # param → aspect ratio b/a in [0.10, 0.95]
    # Low = very elongated (high drag),  high = nearly circular.
    # Major axis always along +x so AoA is well-defined.
    b_over_a = 0.10 + param * 0.85
    n        = 300
    t        = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x        = np.cos(t)           # a = 1 (absorbed by _scale_to_chord)
    y        = b_over_a * np.sin(t)
    x, y     = _scale_to_chord(x, y)
    x        = np.append(x, x[0]);  y = np.append(y, y[0])
    meta     = {"smooth": True, "shape": "generic_smooth"}
    return x, y, meta
 
 
def square_coords(param):
    # param → corner radius ratio cr/size in [0, 0.45]
    # 0 = sharp square,  1 = maximally rounded (approaches circle).
    # Sides axis-aligned.
    size     = 1.0
    cr_ratio = param * 0.45
    cr       = cr_ratio * size
 
    if cr_ratio < 0.01:
        corners = np.array([
            [ size,  size], [-size,  size],
            [-size, -size], [ size, -size],
            [ size,  size],
        ])
        x, y = corners[:, 0], corners[:, 1]
        x, y = _scale_to_chord(x, y)
        meta = {"smooth": False}
 
    else:
        pieces_x, pieces_y = [], []
        cc = [
            ( size - cr,  size - cr),
            (-size + cr,  size - cr),
            (-size + cr, -size + cr),
            ( size - cr, -size + cr),
        ]
        arc_starts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
 
        for k in range(4):
            arc_t = np.linspace(arc_starts[k], arc_starts[k] + np.pi / 2, 20)
            pieces_x.append(cc[k][0] + cr * np.cos(arc_t))
            pieces_y.append(cc[k][1] + cr * np.sin(arc_t))
 
            next_k    = (k + 1) % 4
            end_angle = arc_starts[k] + np.pi / 2
            ex = cc[k][0]      + cr * np.cos(end_angle)
            ey = cc[k][1]      + cr * np.sin(end_angle)
            sx = cc[next_k][0] + cr * np.cos(arc_starts[next_k])
            sy = cc[next_k][1] + cr * np.sin(arc_starts[next_k])
            pieces_x.append(np.linspace(ex, sx, 50))
            pieces_y.append(np.linspace(ey, sy, 50))
 
        x = np.concatenate(pieces_x);  y = np.concatenate(pieces_y)
        x = np.append(x, x[0]);        y = np.append(y, y[0])
        x, y = _scale_to_chord(x, y)
 
        meta = {
            "smooth":        True,
            "shape":         "rounded_square",
            "theta":         0.0,    # no rotation applied
            "size":          size,
            "corner_radius": cr,
        }
 
    return x, y, meta
 
 
def triangle_coords(param):
    # param → apex half-angle in [π/6, 5π/12]
    # Capped at 75° (5π/12) — beyond that the triangle becomes so wide and
    # flat that after _scale_to_chord the height is negligible and the raster
    # is near-empty.  π/6 (30°) gives a sharp pointed triangle at param=0.
    apex_angle = np.pi / 6 + param * (5 * np.pi / 12 - np.pi / 6)
 
    h = 1.0
    b = h * np.tan(apex_angle)
 
    # Apex pointing upstream (−x), base on the downstream (+x) side
    v1 = np.array([-h,  0])
    v2 = np.array([ h,  b])
    v3 = np.array([ h, -b])
 
    pts  = np.array([v1, v2, v3, v1])
    x, y = pts[:, 0], pts[:, 1]
    x, y = _scale_to_chord(x, y)
    meta = {"smooth": False}
    return x, y, meta
 
 
def diamond_coords(param):
    # param → tip half-angle in [π/8, π/2]
    # Capped at π/2 (90°) — beyond that tan(tip_angle) grows large, a = b/tan
    # becomes very small, and the diamond degenerates into a near-vertical
    # sliver that produces a near-empty raster.
    # π/8 (22.5°) = very sharp elongated diamond at param=0
    # π/2 (90°)   = square diamond (equal axes, a=b) at param=1

    param = min(param, 0.9)

    tip_angle = np.pi / 8 + param * (np.pi / 2 - np.pi / 8)
 
    b = 1.0
    a = b / np.tan(tip_angle)
 
    pts  = np.array([[ a, 0], [0,  b], [-a, 0], [0, -b], [a, 0]])
    x, y = pts[:, 0], pts[:, 1]
    x, y = _scale_to_chord(x, y)
    meta = {"smooth": False}
    return x, y, meta
 
 
def airfoil_coords(param):
    # param → NACA thickness and camber
    # Low = thin symmetric (NACA 0008),  high = thick cambered (NACA ~6424).
    # Chord always along +x axis, TE at right — AoA applied by solver.
    m = param * 0.06
    p = 0.2 + param * 0.4
    t = 0.08 + param * 0.16
 
    n  = 300
    xc = np.linspace(0, 1, n)
 
    yt = 5 * t * (0.2969 * np.sqrt(xc)
                  - 0.1260 * xc
                  - 0.3516 * xc**2
                  + 0.2843 * xc**3
                  - 0.1015 * xc**4)
 
    yc     = np.zeros_like(xc)
    dyc_dx = np.zeros_like(xc)
 
    for i in range(len(xc)):
        if xc[i] < p and p != 0:
            yc[i]     = m / (p**2) * (2 * p * xc[i] - xc[i]**2)
            dyc_dx[i] = 2 * m / (p**2) * (p - xc[i])
        elif p != 0:
            yc[i]     = m / ((1 - p)**2) * ((1 - 2*p) + 2*p*xc[i] - xc[i]**2)
            dyc_dx[i] = 2 * m / ((1 - p)**2) * (p - xc[i])
 
    theta_c = np.arctan(dyc_dx)
 
    xu = xc - yt * np.sin(theta_c);  yu = yc + yt * np.cos(theta_c)
    xl = xc + yt * np.sin(theta_c);  yl = yc - yt * np.cos(theta_c)
 
    Xfoil = np.concatenate([xu[::-1], xl[1:]])
    Yfoil = np.concatenate([yu[::-1], yl[1:]])
    Xfoil -= 0.5       # centre at origin; chord runs [-0.5, +0.5] before scaling
    Xfoil, Yfoil = _scale_to_chord(Xfoil, Yfoil)
 
    # Trailing edge: junction of upper and lower surfaces
    te_idx = len(xu) - 1
    te     = np.array([Xfoil[te_idx], Yfoil[te_idx]])
 
    Xfoil = np.append(Xfoil, Xfoil[0])
    Yfoil = np.append(Yfoil, Yfoil[0])
 
    meta = {
        "smooth":        True,
        "shape":         "airfoil",
        "trailing_edge": te,
    }
    return Xfoil, Yfoil, meta
 
 
def generate_geometry_physical(family, param):
    """
    Dispatch to the appropriate physical-space generator.
    Returns (x, y, meta): closed boundary arrays in physical units,
    object centred at origin with characteristic length = CHORD.
    """
    dispatch = {
        "circle":   circle_coords,
        "ellipse":  ellipse_coords,
        "square":   square_coords,
        "triangle": triangle_coords,
        "diamond":  diamond_coords,
        "airfoil":  airfoil_coords,
    }
    return dispatch[family](param)
 
 
# ============================================================
# RASTERISATION
# Convert physical boundary → N×N pixel mask spanning the full
# CFD domain [DOMAIN_X_MIN, DOMAIN_X_MAX] × [DOMAIN_Y_MIN, DOMAIN_Y_MAX].
# The object sits at the origin, so 1L/5L layout is automatic.
# ============================================================
 
def rasterise(x_phys, y_phys, N_grid=N):
    """
    Rasterise a closed physical boundary onto an N×N pixel grid that
    spans the full CFD domain.  Returns a float mask.
 
    Pixel (col, row) = (0,0) is top-left:
        col maps DOMAIN_X_MIN → 0,  DOMAIN_X_MAX → N-1
        row maps DOMAIN_Y_MAX → 0,  DOMAIN_Y_MIN → N-1  (image convention)
    """
    # Map physical → pixel
    px = (x_phys - DOMAIN_X_MIN) / DOMAIN_W * (N_grid - 1)
    py = (DOMAIN_Y_MAX - y_phys) / DOMAIN_H * (N_grid - 1)   # flip Y
 
    vertices = np.vstack([px, py]).T
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
 
    path = Path(vertices)
 
    col_idx = np.arange(N_grid)
    row_idx = np.arange(N_grid)
    Cols, Rows = np.meshgrid(col_idx, row_idx)
    pts = np.vstack([Cols.flatten(), Rows.flatten()]).T
 
    mask = path.contains_points(pts).reshape(N_grid, N_grid)
    return mask.astype(float)
 
 
# ============================================================
# ANSYS EXPORT — physical coordinates, per-shape boundary treatment
# Also writes a companion domain file for SpaceClaim enclosure.
# ============================================================
 
def _smooth_segment(xs, ys, n_out=None):
    if n_out is None:
        n_out = max(len(xs), 20)
 
    # Need at least 4 points for a cubic spline; return raw points below that.
    if len(xs) < 4:
        return np.column_stack((xs, ys))
 
    # savgol_filter requires window_length <= len(x) AND window must be odd.
    # Compute the largest valid odd window, capped at 15.
    max_window = len(xs) if len(xs) % 2 == 1 else len(xs) - 1
    window     = min(15, max_window)
 
    # Polynomial order must be less than window_length
    poly_order = min(3, window - 1)
 
    xs = savgol_filter(xs, window, poly_order)
    ys = savgol_filter(ys, window, poly_order)
 
    try:
        tck, _ = splprep([xs, ys], s=0.002, per=False, k=min(3, len(xs) - 1))
        u = np.linspace(0, 1, n_out)
        xn, yn = splev(u, tck)
        return np.column_stack((xn, yn))
    except Exception:
        return np.column_stack((xs, ys))
 
 
def _process_rounded_square(x, y, meta):
    """
    The rounded-square generator builds the boundary as an explicit sequence
    of 4 corner arcs interleaved with 4 straight sides.  Each corner arc has
    exactly 20 points and each side has exactly 50 points, giving a repeating
    pattern of length 70 per quarter (before the closing vertex).
 
    We exploit this known structure directly instead of trying to back-rotate
    and reclassify — that was the source of the high-kappa artefacts.
    """
    # Drop the closing duplicate point for processing
    pts = np.column_stack((x, y))
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
 
    ARC_PTS  = 20
    SIDE_PTS = 50
    UNIT     = ARC_PTS + SIDE_PTS   # 70 points per quarter
 
    pieces = []
 
    for k in range(4):
        base = k * UNIT
 
        # Corner arc — smooth with spline
        arc_idx = list(range(base, base + ARC_PTS))
        arc_seg = pts[arc_idx]
        pieces.append(_smooth_segment(arc_seg[:, 0], arc_seg[:, 1],
                                      n_out=ARC_PTS * 2))
 
        # Straight side — just two endpoint vertices (perfectly straight)
        side_idx  = [base + ARC_PTS, (base + UNIT - 1) % len(pts)]
        side_pts  = pts[side_idx]
        pieces.append(side_pts)
 
    coords = np.vstack(pieces)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    return coords
 
 
def _process_airfoil(x, y, meta):
    """
    Smooth upper and lower surfaces independently; pin trailing edge.
    """
    te     = meta["trailing_edge"]
    points = np.column_stack((x, y))
 
    dists  = np.linalg.norm(points - te, axis=1)
    te_idx = int(np.argmin(dists))
    le_idx = int(np.argmax(np.linalg.norm(points - points[te_idx], axis=1)))
 
    if te_idx < le_idx:
        upper = points[te_idx : le_idx + 1]
        lower = np.vstack([points[le_idx:], points[: te_idx + 1]])
    else:
        upper = np.vstack([points[te_idx:], points[: le_idx + 1]])
        lower = points[le_idx : te_idx + 1]
 
    upper_s = _smooth_segment(upper[:, 0], upper[:, 1], 150)
    lower_s = _smooth_segment(lower[:, 0], lower[:, 1], 150)
 
    te_pt  = te.reshape(1, 2)
    coords = np.vstack([te_pt, upper_s[1:], lower_s[1:-1], te_pt])
    return coords
 
 
def _process_smooth_generic(x, y):
    window = min(21, len(x) // 2 * 2 + 1)
    xs = savgol_filter(x, window, 3)
    ys = savgol_filter(y, window, 3)
    try:
        tck, _ = splprep([xs, ys], s=0.001, per=True)
        u = np.linspace(0, 1, 300)
        xn, yn = splev(u, tck)
        return np.column_stack((xn, yn))
    except Exception:
        return np.column_stack((xs, ys))
 
 
def export_ansys(x_phys, y_phys, meta, case_id, family, out_dir):
    """
    Write two files per case:
      ansys_curve_<id>_<family>.dat   — object boundary (x, y, z=0)
      ansys_domain_<id>.dat           — rectangular CFD domain enclosure
    All coordinates are in physical units (CHORD = 1.0).
    """
    shape = meta.get("shape", None)
 
    if not meta["smooth"]:
        pts = np.column_stack((x_phys, y_phys))
        coords = measure.approximate_polygon(pts, tolerance=0.005 * CHORD)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
 
    elif shape == "rounded_square":
        coords = _process_rounded_square(x_phys, y_phys, meta)
 
    elif shape == "airfoil":
        coords = _process_airfoil(x_phys, y_phys, meta)
 
    else:   # generic smooth: circle, ellipse
        coords = _process_smooth_generic(x_phys, y_phys)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
 
    # Object boundary file
    obj_out = np.column_stack((coords[:, 0], coords[:, 1], np.zeros(len(coords))))
    np.savetxt(
        os.path.join(out_dir, f"ansys_curve_{case_id}_{family}.dat"),
        obj_out, fmt="%.6f", delimiter=","
    )
 
    # Domain enclosure file — independent of object size, always correct
    domain = np.array([
        [DOMAIN_X_MIN, DOMAIN_Y_MIN, 0],
        [DOMAIN_X_MAX, DOMAIN_Y_MIN, 0],
        [DOMAIN_X_MAX, DOMAIN_Y_MAX, 0],
        [DOMAIN_X_MIN, DOMAIN_Y_MAX, 0],
        [DOMAIN_X_MIN, DOMAIN_Y_MIN, 0],
    ])
    np.savetxt(
        os.path.join(out_dir, f"ansys_domain_{case_id}.dat"),
        domain, fmt="%.6f", delimiter=","
    )
 
 
# ============================================================
# SPECTRAL COMPLEXITY (unchanged — operates on pixel mask)
# ============================================================
 
def spectral_complexity(mask):
    fft       = np.fft.fft2(mask)
    fft_shift = np.fft.fftshift(fft)
    power     = np.abs(fft_shift)**2
    N_g       = mask.shape[0]
    center    = N_g // 2
    cutoff    = N_g // 8
    yy, xx    = np.ogrid[:N_g, :N_g]
    dist      = np.sqrt((xx - center)**2 + (yy - center)**2)
    high_freq = power[dist > cutoff]
    return high_freq.sum() / (power.sum() + 1e-12)
 
 
# ============================================================
# MAIN
# ============================================================
# Usage:
#   python Geometry_Generation.py --res 128          (default)
#   python Geometry_Generation.py --res 64
#   python Geometry_Generation.py --res 256
#   python Geometry_Generation.py --res 128 --skip-ansys
#
# ANSYS .dat files are resolution-independent (physical coords).
# They are written only for the first / canonical run.
# Subsequent resolution runs use --skip-ansys to avoid
# redundantly overwriting the same files.
#
# Directory layout:
#   simulation/
#     cfd_dataset_cases.csv              ← LHS sampling (shared)
#     generated_geometries/              ← ANSYS .dat files (shared, written once)
#     res_064/                           ← 64×64  masks + metadata
#     res_128/                           ← 128×128 masks + metadata
#     res_256/                           ← 256×256 masks + metadata
# ============================================================
 
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--res",        type=int, default=128,
                    help="Pixel grid resolution (64 / 128 / 256)")
parser.add_argument("--skip-ansys", action="store_true",
                    help="Skip writing ANSYS .dat files (use for 2nd/3rd resolution runs)")
args = parser.parse_args()
 
N           = args.res
write_ansys = not args.skip_ansys
 
ansys_dir  = "simulation/generated_geometries"
output_dir = f"simulation/res_{N:03d}"
 
os.makedirs(ansys_dir,  exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
 
df = pd.read_csv("simulation/cfd_dataset_cases.csv")
 
geometries = []
all_meta   = []
 
print("Generating geometries...")
print(f"  Resolution      : {N}×{N}")
print(f"  Physical domain : x=[{DOMAIN_X_MIN}, {DOMAIN_X_MAX}]  "
      f"y=[{DOMAIN_Y_MIN}, {DOMAIN_Y_MAX}]  (CHORD={CHORD})")
print(f"  Pixel output    : {output_dir}/")
print(f"  ANSYS export    : {'yes → ' + ansys_dir + '/' if write_ansys else 'skipped (--skip-ansys)'}")
 
for row in df.itertuples():
 
    param   = row.geometry_parameter
    case_id = int(row.case_id)
 
    # 1. Generate closed boundary in physical space (same for all resolutions)
    x_phys, y_phys, meta = generate_geometry_physical(row.geometry_family, param)
 
    # 2. Rasterise onto this resolution's pixel grid
    mask = rasterise(x_phys, y_phys, N_grid=N)
 
    if np.sum(mask) < 10:
        print(f"  Warning: near-empty raster for case {case_id} "
              f"({row.geometry_family}, param={param:.4f})")
 
    if (np.any(mask[0, :]) or np.any(mask[-1, :]) or
            np.any(mask[:, 0]) or np.any(mask[:, -1])):
        print(f"  Warning: raster touches pixel boundary for case {case_id}")
 
    geometries.append(mask)
    all_meta.append(meta)
 
    # 3. Save pixel mask to resolution-specific directory
    np.save(
        os.path.join(output_dir, f"geometry_{case_id}_{row.geometry_family}.npy"),
        mask
    )
 
    # 4. ANSYS export — physical coords, resolution-independent, write once
    if write_ansys:
        export_ansys(x_phys, y_phys, meta, case_id, row.geometry_family, ansys_dir)
 
geometries = np.array(geometries)
 
print("Done.")
print(f"Geometry tensor shape: {geometries.shape}")
 
# Spectral complexity (resolution-dependent — recomputed per run)
df["spectral_complexity"] = [spectral_complexity(m) for m in geometries]
df["smooth_boundary"]     = [meta["smooth"] for meta in all_meta]
df["resolution"]          = N
 
# Train / test split — use same random_state so splits are identical
# across resolutions, which is essential for the FNO resolution experiment.
train_df, test_df = train_test_split(df, test_size=126, random_state=42, shuffle=True)
train_df["split"] = "train"
test_df["split"]  = "test"
 
final_df = pd.concat([train_df, test_df])
final_df.to_csv(os.path.join(output_dir, "cfd_dataset_metadata.csv"), index=False)
 
print("Dataset preparation complete.")
print(f"  Resolution : {N}×{N}")
print(f"  Output dir : {output_dir}/")
print(f"  Total      : {len(final_df)}")
print(f"  Train      : {len(train_df)}")
print(f"  Test       : {len(test_df)}")