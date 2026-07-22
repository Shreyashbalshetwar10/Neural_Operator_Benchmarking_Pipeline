import numpy as np
import pandas as pd
from scipy.stats import qmc
 
# ------------------------------------
# Dataset configuration
# ------------------------------------
 
n_total  = 630
families = ["circle", "ellipse", "triangle", "square", "diamond", "airfoil"]
 
n_per_family = n_total // len(families)   # 105 per family
 
AoA_range   = (-15, 15)    # degrees — angle of attack, used by CFD solver
Re_range    = (50, 800)    # Reynolds number
 
# geometry_parameter is a normalised shape descriptor in [0, 1].
# Its physical meaning is family-specific:
#
#   circle   — unused (circle has no shape dof)
#   ellipse  — aspect ratio b/a:  0 → very elongated,  1 → nearly circular
#   square   — corner radius ratio cr/size:  0 → sharp,  1 → maximally rounded
#   triangle — apex half-angle:   0 → very pointed,    1 → wide/blunt
#   diamond  — tip sharpness:     0 → very sharp tips, 1 → squat diamond
#   airfoil  — thickness/camber:  0 → thin symmetric,  1 → thick cambered
#
# Using [0, 1] ensures LHS samples the full shape space of each family.
geom_param_range = (0.0, 1.0)
 
all_cases = []
case_id   = 0
 
# ------------------------------------
# Generate cases per geometry family
# ------------------------------------
 
for family in families:
 
    sampler = qmc.LatinHypercube(d=3, seed=42)
    samples = sampler.random(n=n_per_family)
 
    AoA        = AoA_range[0]        + samples[:, 0] * (AoA_range[1]        - AoA_range[0])
    Re         = Re_range[0]         + samples[:, 1] * (Re_range[1]         - Re_range[0])
    geom_param = geom_param_range[0] + samples[:, 2] * (geom_param_range[1] - geom_param_range[0])
 
    for i in range(n_per_family):
        all_cases.append({
            "case_id":            case_id,
            "geometry_family":    family,
            "geometry_parameter": geom_param[i],
            "AoA":                AoA[i],
            "Re":                 Re[i],
        })
        case_id += 1
 
# ------------------------------------
# Save dataset metadata
# ------------------------------------
 
df = pd.DataFrame(all_cases)
df.to_csv("simulation/cfd_dataset_cases.csv", index=False)
 
print(f"Dataset created with {len(df)} cases")
print(df.groupby("geometry_family")["geometry_parameter"].describe().round(3))
print(df.head())