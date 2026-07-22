import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re
import glob
import random
from pathlib import Path
 
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=128,
                    help="Resolution to verify (64 / 128 / 256)")
args = parser.parse_args()
 
# ============================================================
# Paths
# ============================================================
 
RES          = args.res
PROJECT_DIR  = Path(__file__).resolve().parent
GEOMETRY_DIR = str(PROJECT_DIR / "simulation" / "generated_geometries")
RES_DIR      = str(PROJECT_DIR / f"simulation" / f"res_{RES:03d}")
METADATA_PATH = os.path.join(RES_DIR, "cfd_dataset_metadata.csv")
 
# Physical domain constants — must match Geometry_Generation.py
CHORD        = 1.0
DOMAIN_X_MIN = -1.0
DOMAIN_X_MAX =  5.0
DOMAIN_Y_MIN = -2.0
DOMAIN_Y_MAX =  2.0
DOMAIN_W     = DOMAIN_X_MAX - DOMAIN_X_MIN   # 6.0
DOMAIN_H     = DOMAIN_Y_MAX - DOMAIN_Y_MIN   # 4.0
 
# ============================================================
# Load metadata and pixel masks
# ============================================================
 
df = pd.read_csv(METADATA_PATH)
print(f"Loaded metadata: {len(df)} cases  (resolution {RES}×{RES})")
 
geometries = []
for row in df.itertuples(index=False):
    path = os.path.join(RES_DIR, f"geometry_{row.case_id}_{row.geometry_family}.npy")
    geometries.append(np.load(path))
 
geometries = np.array(geometries)
print(f"Geometry tensor shape: {geometries.shape}")
 
N = geometries.shape[1]
 
# ============================================================
# PIXEL MASK CHECKS
# ============================================================
 
print("\n--- Pixel mask checks ---")
 
empty_cases    = [i for i, m in enumerate(geometries) if np.sum(m) < 10]
boundary_cases = [
    i for i, m in enumerate(geometries)
    if np.any(m[0, :]) or np.any(m[-1, :]) or np.any(m[:, 0]) or np.any(m[:, -1])
]
 
if empty_cases:
    print(f"WARNING: Empty raster masks: {len(empty_cases)} cases")
    for i in empty_cases:
        print(f"    index {i} — case {df.iloc[i]['case_id']} "
              f"({df.iloc[i]['geometry_family']}, "
              f"param={df.iloc[i]['geometry_parameter']:.4f})")
else:
    print("  No empty masks.")
 
if boundary_cases:
    print(f"WARNING: Masks touching pixel boundary: {len(boundary_cases)} cases")
    for i in boundary_cases:
        print(f"    case {df.iloc[i]['case_id']} ({df.iloc[i]['geometry_family']})")
else:
    print("  No masks touching pixel boundary.")
 
# Check that the object centroid pixel column matches the expected position.
# The object sits at physical x=0. The domain runs from DOMAIN_X_MIN=-1 to
# DOMAIN_X_MAX=5 (width=6). So the object maps to pixel column:
#   col = (0 - DOMAIN_X_MIN) / DOMAIN_W * (N-1) = 1/6 * (N-1) ≈ N*0.167
# N*0.25 was wrong — it assumed the object was at x=0.5, not x=0.
expected_col = (0 - DOMAIN_X_MIN) / DOMAIN_W * (N - 1)
print(f"\n  Object centroid column positions (expected ≈ {expected_col:.1f}):")
col_positions = []
for m in geometries:
    ys, xs = np.where(m > 0)
    if xs.size > 0:
        col_positions.append((xs.max() + xs.min()) / 2)
col_positions = np.array(col_positions)
print(f"    mean={col_positions.mean():.1f}  "
      f"min={col_positions.min():.1f}  max={col_positions.max():.1f}  "
      f"(N={N}, expected≈{expected_col:.1f})")
 
# ============================================================
# ANSYS CURVE FILE CHECKS  (physical-space .dat files)
# ============================================================
 
print("\n--- ANSYS curve file checks ---")
 
CURVATURE_THRESHOLD = {
    "circle":   10,
    # Highly elongated ellipses (low param) have sharp tips — kappa naturally
    # runs 10-20 there. Raise threshold to only catch genuinely broken splines.
    "ellipse":  25,
    "airfoil":  None,   # LE curvature is physically real, not an error
    "square":   None,   # intentional corners
    "triangle": None,   # intentional corners
    "diamond":  None,   # intentional corners
}
 
def check_closed(data):
    return np.linalg.norm(data[0, :2] - data[-1, :2]) < 1e-4
 
def curvature_variation(x, y):
    dx  = np.gradient(x);  dy  = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    safe  = denom > 1e-10
    kappa = np.zeros_like(x)
    kappa[safe] = np.abs(dx[safe]*ddy[safe] - dy[safe]*ddx[safe]) / denom[safe]
    return np.std(kappa)
 
def family_from_filename(f):
    m = re.search(r"ansys_curve_\d+_(\w+)\.dat$", os.path.basename(f))
    return m.group(1) if m else None
 
curve_files = sorted(
    os.path.join(GEOMETRY_DIR, f)
    for f in os.listdir(GEOMETRY_DIR)
    if f.startswith("ansys_curve_") and f.endswith(".dat")
)
 
n_open = n_curve = n_small = n_outside_domain = 0
 
for f in curve_files:
    data   = np.loadtxt(f, delimiter=",")
    family = family_from_filename(f)
 
    # Closure
    if not check_closed(data):
        print(f"  Open curve   : {os.path.basename(f)}")
        n_open += 1
 
    # Curvature (family-aware)
    thresh = CURVATURE_THRESHOLD.get(family, 10)
    if thresh is not None:
        cv = curvature_variation(data[:, 0], data[:, 1])
        if cv > thresh:
            print(f"  High kappa ({cv:.2f}): {os.path.basename(f)}")
            n_curve += 1
 
    # Object size: use the longest bounding-box axis, not just x-width.
    # Diamonds at high param are nearly vertical — their x-extent is tiny
    # but they are full-size objects. After _scale_to_chord the longest
    # axis should always be ≈ CHORD; if both axes are tiny something is wrong.
    cx   = data[:, 0]
    cy   = data[:, 1]
    size = max(cx.max() - cx.min(), cy.max() - cy.min())
    if size < 0.5 * CHORD:   # genuine size failure — longest axis < half chord
        print(f"  Too small ({size:.3f}): {os.path.basename(f)}")
        n_small += 1
 
    # All object coords must be inside the physical domain
    if (data[:, 0].min() < DOMAIN_X_MIN or data[:, 0].max() > DOMAIN_X_MAX or
            data[:, 1].min() < DOMAIN_Y_MIN or data[:, 1].max() > DOMAIN_Y_MAX):
        print(f"  Outside domain: {os.path.basename(f)}")
        n_outside_domain += 1
 
print(f"\n  Curve files checked : {len(curve_files)}")
print(f"  Open curves         : {n_open}")
print(f"  High curvature      : {n_curve}")
print(f"  Too small           : {n_small}")
print(f"  Outside domain      : {n_outside_domain}")
 
# Check domain files exist for every curve file
domain_files = {
    re.search(r"ansys_domain_(\d+)\.dat$", f).group(1)
    for f in os.listdir(GEOMETRY_DIR)
    if re.search(r"ansys_domain_(\d+)\.dat$", f)
}
curve_ids = {
    re.search(r"ansys_curve_(\d+)_", f).group(1)
    for f in os.listdir(GEOMETRY_DIR)
    if re.search(r"ansys_curve_(\d+)_", f)
}
missing_domains = curve_ids - domain_files
if missing_domains:
    print(f"  Missing domain files for case IDs: {sorted(missing_domains)}")
else:
    print(f"  All {len(curve_ids)} domain enclosure files present.")
 
# ============================================================
# VISUALISATION 1 — random pixel masks
# ============================================================
 
print("\nDisplaying random pixel masks...")
plt.figure(figsize=(8, 8))
indices = np.random.choice(len(geometries), 9, replace=False)
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(geometries[idx], cmap="gray", origin="upper",
               extent=[DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX])
    # Draw 1L upstream marker
    plt.axvline(x=DOMAIN_X_MIN + CHORD, color="cyan", linewidth=0.8, linestyle="--")
    plt.title(df.iloc[idx]["geometry_family"], fontsize=8)
    plt.axis("off")
plt.suptitle("Pixel masks (dashed = 1L upstream boundary)", fontsize=10)
plt.tight_layout()
plt.show()
 
# ============================================================
# VISUALISATION 2 — random ANSYS curves overlaid on domain
# ============================================================
 
print("Displaying random ANSYS curves...")
sample_files = random.sample(curve_files, min(9, len(curve_files)))
 
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, f in zip(axes.flatten(), sample_files):
    data   = np.loadtxt(f, delimiter=",")
    family = family_from_filename(f)
 
    # Draw domain rectangle
    domain_rect = mpatches.Rectangle(
        (DOMAIN_X_MIN, DOMAIN_Y_MIN), DOMAIN_W, DOMAIN_H,
        linewidth=1, edgecolor="steelblue", facecolor="aliceblue"
    )
    ax.add_patch(domain_rect)
 
    # 1L upstream and 5L downstream markers
    ax.axvline(x=0 - CHORD, color="cyan",   linewidth=0.8, linestyle="--", label="1L upstream")
    ax.axvline(x=0 + 5*CHORD, color="orange", linewidth=0.8, linestyle="--", label="5L downstream")
 
    # Object boundary
    ax.plot(data[:, 0], data[:, 1], "k-", linewidth=1.2)
    ax.fill(data[:, 0], data[:, 1], alpha=0.3, color="gray")
 
    ax.set_xlim(DOMAIN_X_MIN - 0.2, DOMAIN_X_MAX + 0.2)
    ax.set_ylim(DOMAIN_Y_MIN - 0.2, DOMAIN_Y_MAX + 0.2)
    ax.set_aspect("equal")
    ax.set_title(os.path.basename(f).replace("ansys_curve_", "").replace(".dat", ""),
                 fontsize=7)
    ax.tick_params(labelsize=6)
 
plt.suptitle("ANSYS curves in physical domain  (cyan=1L, orange=5L)", fontsize=10)
plt.tight_layout()
plt.show()
 
# ============================================================
# VISUALISATION 3 — spectral complexity distribution
# ============================================================
 
if "spectral_complexity" in df.columns:
    plt.figure(figsize=(6, 4))
    for fam in df["geometry_family"].unique():
        vals = df.loc[df["geometry_family"] == fam, "spectral_complexity"]
        plt.hist(vals, bins=15, alpha=0.6, label=fam)
    plt.title("Spectral Complexity by Family")
    plt.xlabel("Spectral Complexity")
    plt.ylabel("Count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
 
# ============================================================
# VISUALISATION 4 — geometry family distribution
# ============================================================
 
print("\nGeometry family distribution:")
print(df["geometry_family"].value_counts())
 
plt.figure(figsize=(6, 4))
df["geometry_family"].value_counts().plot(kind="bar")
plt.title("Geometry Family Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
 
print("\nVerification complete.")