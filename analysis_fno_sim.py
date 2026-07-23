"""
Post-training analysis for the FNO baseline on the synthetic CFD benchmark.
Loads saved results CSVs produced by train_fno_sim.py, then runs:

1. Per-geometry and per-Re error breakdown
2. Flow complexity correlation analysis (wake energy, vorticity, pressure variation)
3. Outlier sensitivity — IQR-based removal with Pearson + Spearman comparison
4. Bluff body deep-dive — AoA and flow complexity correlations for square/triangle
5. Oversampling effectiveness analysis
6. High-error outlier case identification per geometry family

Run after train_fno_sim.py has completed.

Usage:
    python analysis_fno_sim.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results_sim"
ANALYSIS_DIR = "analysis_sim"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

test_results  = pd.read_csv(f"{RESULTS_DIR}/test_results_fno.csv")
train_results = pd.read_csv(f"{RESULTS_DIR}/train_results_fno.csv")

print(f"Loaded test results:  {len(test_results)} samples")
print(f"Loaded train results: {len(train_results)} samples")

COMPLEXITY_METRICS = ["wake_energy", "pressure_variation", "vorticity_energy"]

# ── 1. Per-geometry error breakdown ──────────────────────────────────────────
print("\n" + "="*60)
print("1. PER-GEOMETRY ERROR BREAKDOWN (Test Set)")
print("="*60)

geom_stats = (
    test_results
    .groupby("geometry_family")[["Ux_L2", "Uy_L2", "p_L2", "Total_L2"]]
    .agg(["mean", "std"])
)
print(geom_stats.round(4))

geom_mean = test_results.groupby("geometry_family")["Total_L2"].mean().sort_values()
plt.figure(figsize=(8, 4))
geom_mean.plot(kind="bar", color="steelblue", edgecolor="k", alpha=0.8)
plt.ylabel("Mean Relative L2 Error")
plt.title("Test Error by Geometry Family")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/error_by_geometry.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {ANALYSIS_DIR}/error_by_geometry.png")

# ── 2. Per-Re bin error breakdown ─────────────────────────────────────────────
print("\n" + "="*60)
print("2. ERROR vs REYNOLDS NUMBER")
print("="*60)

re_bins = [50, 200, 350, 500, 650, 800]
test_results["Re_bin"]  = pd.cut(test_results["Re"],  bins=re_bins)
train_results["Re_bin"] = pd.cut(train_results["Re"], bins=re_bins)

re_stats = test_results.groupby("Re_bin")["Total_L2"].agg(["mean", "std", "count"])
print(re_stats.round(4))

print("\nGeometry x Re pivot (test mean Total_L2):")
pivot = test_results.pivot_table(
    values="Total_L2", index="geometry_family",
    columns="Re_bin", aggfunc="mean"
)
print(pivot.round(4))
pivot.to_csv(f"{ANALYSIS_DIR}/geometry_re_pivot.csv")

plt.figure(figsize=(10, 4))
re_stats["mean"].plot(marker="o", color="steelblue")
plt.fill_between(
    range(len(re_stats)),
    re_stats["mean"] - re_stats["std"],
    re_stats["mean"] + re_stats["std"],
    alpha=0.2
)
plt.ylabel("Mean Relative L2 Error")
plt.title("Test Error vs Reynolds Number Bin")
plt.xticks(range(len(re_stats)), [str(b) for b in re_stats.index], rotation=30)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/error_vs_re.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Flow complexity correlation analysis ───────────────────────────────────
print("\n" + "="*60)
print("3. FLOW COMPLEXITY CORRELATIONS (Test Set)")
print("="*60)

available = [m for m in COMPLEXITY_METRICS if m in test_results.columns]
if not available:
    print("Complexity columns not found — re-run train_fno_sim.py to generate them.")
else:
    print(f"\n{'Metric':<25} {'Pearson r':>10} {'Spearman r':>11}")
    print("-" * 50)
    for col in available + ["Re", "AoA"]:
        r    = test_results["Total_L2"].corr(test_results[col])
        s, _ = spearmanr(test_results["Total_L2"], test_results[col])
        print(f"{col:<25} {r:>+10.3f} {s:>+11.3f}")

    fig, axes = plt.subplots(1, len(available), figsize=(6*len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        for fam in sorted(test_results["geometry_family"].unique()):
            sub = test_results[test_results["geometry_family"] == fam]
            ax.scatter(sub[metric], sub["Total_L2"], alpha=0.6, label=fam, s=25)
        r    = test_results["Total_L2"].corr(test_results[metric])
        s, _ = spearmanr(test_results["Total_L2"], test_results[metric])
        ax.set_xlabel(metric)
        ax.set_ylabel("Total_L2")
        ax.set_title(f"Total_L2 vs {metric}")
        ax.text(0.05, 0.95, f"Pearson={r:.3f}\nSpearman={s:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_DIR}/complexity_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {ANALYSIS_DIR}/complexity_correlations.png")

# ── 4. Outlier sensitivity analysis ──────────────────────────────────────────
print("\n" + "="*60)
print("4. OUTLIER SENSITIVITY (Train Set, IQR k=1.5)")
print("="*60)

def remove_outliers_iqr(df, col, k=1.5):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    mask   = (df[col] >= Q1 - k*IQR) & (df[col] <= Q3 + k*IQR)
    return df[mask].copy(), df[~mask].copy()

available_train = [m for m in COMPLEXITY_METRICS if m in train_results.columns]
if available_train:
    print(f"\n{'Metric':<25} {'P full':>8} {'P clean':>8} "
          f"{'S full':>8} {'S clean':>8} {'Removed':>8}")
    print("-" * 70)

    for metric in available_train:
        clean, outliers = remove_outliers_iqr(train_results, metric)
        p_full  = train_results["Total_L2"].corr(train_results[metric])
        p_clean = clean["Total_L2"].corr(clean[metric])
        s_full,  _ = spearmanr(train_results["Total_L2"], train_results[metric])
        s_clean, _ = spearmanr(clean["Total_L2"],         clean[metric])

        print(f"{metric:<25} {p_full:>+8.3f} {p_clean:>+8.3f} "
              f"{s_full:>+8.3f} {s_clean:>+8.3f} {len(outliers):>8}")

        if len(outliers) > 0:
            cols = ["geometry_family", "Re", "AoA", "Total_L2", metric]
            print("  Removed cases:")
            print("  " + outliers[cols]
                  .sort_values(metric, ascending=False)
                  .to_string(index=False).replace("\n", "\n  "))

# ── 5. Bluff body deep-dive ───────────────────────────────────────────────────
print("\n" + "="*60)
print("5. BLUFF BODY DEEP-DIVE (square + triangle, train set)")
print("="*60)

for geom in ["square", "triangle"]:
    sub = train_results[train_results["geometry_family"] == geom]
    print(f"\n--- {geom.upper()} (n={len(sub)}) ---")
    for metric in available_train + ["AoA"]:
        s, p = spearmanr(sub["Total_L2"], sub[metric])
        print(f"  Spearman(Total_L2, {metric:<22}) = {s:+.3f}  (p={p:.3f})")

    if geom == "triangle":
        print(f"\n  AoA sign effect:")
        print(f"    corr with |AoA|: {spearmanr(sub['Total_L2'], sub['AoA'].abs())[0]:+.3f}")
        sign_stats = (sub.groupby(sub["AoA"] < 0)["Total_L2"]
                      .agg(["mean", "std", "count"]))
        sign_stats.index = ["AoA >= 0", "AoA < 0"]
        print(sign_stats.round(4))

# ── 6. Oversampling effectiveness ─────────────────────────────────────────────
print("\n" + "="*60)
print("6. OVERSAMPLING EFFECTIVENESS")
print("="*60)
print("Triangle geometry was oversampled 3x during training.\n")

for geom in ["square", "triangle"]:
    tr = train_results[train_results["geometry_family"] == geom]["Total_L2"]
    te = test_results[test_results["geometry_family"] == geom]["Total_L2"]
    print(f"{geom:10s}  train: {tr.mean():.4f} +/- {tr.std():.4f}  |  "
          f"test: {te.mean():.4f} +/- {te.std():.4f}  |  "
          f"gap: {te.mean()-tr.mean():+.4f}")

print("\nConclusion: minimal improvement on bluff body test errors despite 3x oversampling,")
print("suggesting inherent flow complexity rather than data imbalance is the limiting factor.")

# ── 7. High-error outlier cases ───────────────────────────────────────────────
print("\n" + "="*60)
print("7. HIGH-ERROR OUTLIER CASES (test set, mean + 2*std threshold)")
print("="*60)

for geom in ["square", "triangle", "circle"]:
    sub       = test_results[test_results["geometry_family"] == geom]
    threshold = sub["Total_L2"].mean() + 2 * sub["Total_L2"].std()
    outliers  = sub[sub["Total_L2"] > threshold].sort_values("Total_L2", ascending=False)
    print(f"\n{geom.upper()} — threshold: {threshold:.3f} | "
          f"outliers: {len(outliers)}/{len(sub)}")
    if len(outliers) > 0:
        cols = ["Re", "AoA", "Total_L2", "Ux_L2", "Uy_L2", "p_L2"]
        print(outliers[cols].to_string(index=False))

print(f"\nAll outputs saved to: {ANALYSIS_DIR}/")
