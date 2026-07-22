import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from models.fno_baseline import FNO2d, UnitGaussianNormalizer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "simulation/dataset/dataset_128.npz"
SAVE_DIR    = "checkpoints_sim"
SAVE_PATH   = f"{SAVE_DIR}/fno_baseline_best.pth"
RESULTS_DIR = "results_sim"

MODES         = 16
WIDTH         = 32
BATCH_SIZE    = 8
EPOCHS        = 500
LEARNING_RATE = 1e-3

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
d        = np.load(DATA_PATH)
data     = torch.from_numpy(d["data"]).float()   # (N, 4, H, W)
Re_vals  = torch.from_numpy(d["Re"].astype(np.float32))
AoA_vals = torch.from_numpy(d["AoA"].astype(np.float32))
families = d["families"]

meta_df = pd.DataFrame({
    "geometry_family": families,
    "Re":  Re_vals.numpy(),
    "AoA": AoA_vals.numpy(),
})

N, C, H, W = data.shape
mask  = data[:, 0:1, :, :]   # (N,1,H,W) — 1=solid, 0=fluid
y_out = data[:, 1:, :, :]    # (N,3,H,W) — [Ux, Uy, p]

# Input features: SDF, x/y coordinates, log(Re), sin/cos(AoA)
Re_log  = torch.log10(Re_vals).view(N,1,1,1).expand(N,1,H,W)
sin_AoA = torch.sin(AoA_vals * np.pi / 180.0).view(N,1,1,1).expand(N,1,H,W)
cos_AoA = torch.cos(AoA_vals * np.pi / 180.0).view(N,1,1,1).expand(N,1,H,W)

yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
xx = xx.unsqueeze(0).unsqueeze(0).expand(N,1,H,W)
yy = yy.unsqueeze(0).unsqueeze(0).expand(N,1,H,W)

print("Computing signed distance fields...")
sdf_list = []
for i in range(N):
    solid = mask[i,0].numpy().astype(bool)
    sdf_list.append(distance_transform_edt(~solid) - distance_transform_edt(solid))
sdf = torch.tensor(np.array(sdf_list)).unsqueeze(1).float() / max(H, W)

x_in = torch.cat([sdf, xx, yy, Re_log, sin_AoA, cos_AoA], dim=1)  # (N,6,H,W)

# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(x_in, y_out, mask, meta_df,
                     n_train=75, n_val=15, n_test=15, seed=42):
    """
    Stratified split ensuring each geometry family and Re bin
    is proportionally represented across train/val/test.
    """
    rng     = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    train_idx, val_idx, test_idx = [], [], []

    for fam in sorted(meta_df["geometry_family"].unique()):
        fam_df = meta_df[meta_df["geometry_family"] == fam].copy()
        fam_df["re_bin"] = pd.qcut(fam_df["Re"], q=4, labels=False)

        for bin_id in range(4):
            bin_idx  = fam_df.index[fam_df["re_bin"] == bin_id].tolist()
            shuffled = rng.permutation(bin_idx).tolist()
            n        = len(shuffled)
            n_t      = round(n * n_train / n_total)
            n_v      = round(n * n_val   / n_total)
            train_idx.extend(shuffled[:n_t])
            val_idx.extend(shuffled[n_t:n_t+n_v])
            test_idx.extend(shuffled[n_t+n_v:])

    def make_ds(idx):
        return TensorDataset(x_in[idx], y_out[idx], mask[idx],
                             torch.tensor(idx, dtype=torch.long))

    n_fam = len(meta_df["geometry_family"].unique())
    print(f"\nSplit: train={len(train_idx)} ({len(train_idx)//n_fam}/fam) | "
          f"val={len(val_idx)} | test={len(test_idx)}")

    return (make_ds(train_idx), make_ds(val_idx), make_ds(test_idx),
            train_idx, val_idx, test_idx)


train_ds, val_ds, test_ds, train_idx, val_idx, test_idx = stratified_split(
    x_in, y_out, mask, meta_df)

# Normalise outputs on fluid regions only
y_normalizer = UnitGaussianNormalizer(train_ds.tensors[2], train_ds.tensors[1])

def encode_ds(ds):
    x, y, m, idx = ds.tensors
    return TensorDataset(x, y_normalizer.encode(m, y), m, idx)

train_dataset = encode_ds(train_ds)
val_dataset   = encode_ds(val_ds)
test_dataset  = encode_ds(test_ds)

# Oversample bluff bodies (square, triangle) 3x to address class imbalance
train_families   = meta_df.loc[train_idx, "geometry_family"].values
geometry_weights = {fam: 3.0 if fam in ["square", "triangle"] else 1.0
                    for fam in meta_df["geometry_family"].unique()}
sample_weights   = [geometry_weights[f] for f in train_families]

sampler      = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

y_normalizer.to(device)
model = FNO2d(modes=MODES, width=WIDTH, num_in_channels=6, num_out_channels=3).to(device)

def fluid_mse(out, y, m):
    """MSE computed only on fluid regions (mask < 0.5 = fluid)."""
    fluid = (m < 0.5).float()
    return ((out - y)**2 * fluid).sum() / (fluid.sum() * out.shape[1] + 1e-8)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

# ── Training loop ─────────────────────────────────────────────────────────────
print("Starting training...")
best_val_loss = float("inf")
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y, m, _ in train_loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        optimizer.zero_grad()
        loss = fluid_mse(model(x), y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y, m, _ in val_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            val_loss += fluid_mse(model(x), y, m).item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss, "epoch": epoch}, SAVE_PATH)

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:4d} | train {avg_train:.6f} | "
              f"val {avg_val:.6f} | best {best_val_loss:.6f}")

print(f"Training complete in {(time.time()-t0)/60:.1f} min")

# ── Evaluation ────────────────────────────────────────────────────────────────
def relative_l2(pred, true, m):
    fluid = (m < 0.5).float().to(pred.device)
    return (torch.norm((pred-true)*fluid) / (torch.norm(true*fluid) + 1e-8)).item()

ckpt = torch.load(SAVE_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def evaluate(dataset, name):
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y_enc, m, idx = dataset[i]
            x     = x.unsqueeze(0).to(device)
            y_enc = y_enc.unsqueeze(0).to(device)
            m_dev = m.unsqueeze(0).to(device)
            pred  = y_normalizer.decode(model(x))
            true  = y_normalizer.decode(y_enc)
            fam   = meta_df.iloc[idx.item()]["geometry_family"]
            re    = meta_df.iloc[idx.item()]["Re"]
            aoa   = meta_df.iloc[idx.item()]["AoA"]
            rows.append({
                "geometry_family": fam, "Re": re, "AoA": aoa,
                "Ux_L2":    relative_l2(pred[:,0:1], true[:,0:1], m_dev),
                "Uy_L2":    relative_l2(pred[:,1:2], true[:,1:2], m_dev),
                "p_L2":     relative_l2(pred[:,2:3], true[:,2:3], m_dev),
                "Total_L2": relative_l2(pred,        true,        m_dev),
                "_pred": pred.squeeze(0).cpu(),
                "_true": true.squeeze(0).cpu(),
                "_mask": m,
            })

    df = pd.DataFrame(rows)
    print(f"\n=== {name} Results ===")
    print(f"  Total L2:  {df['Total_L2'].mean():.4f} ± {df['Total_L2'].std():.4f}")
    print(f"  Ux L2:     {df['Ux_L2'].mean():.4f}")
    print(f"  Uy L2:     {df['Uy_L2'].mean():.4f}")
    print(f"  p  L2:     {df['p_L2'].mean():.4f}")
    print(f"\nPer-family breakdown:")
    print(df.groupby("geometry_family")["Total_L2"].agg(["mean","std","count"]).round(4))
    return df

test_results  = evaluate(test_dataset,  "Test")
train_results = evaluate(train_dataset, "Train")

# ── Flow complexity correlation analysis ──────────────────────────────────────
def compute_complexity(y_field, mask_np):
    """
    Compute flow-derived complexity metrics on fluid regions.
    y_field : (3, H, W) numpy — [Ux, Uy, p]
    mask_np : (H, W) numpy — 1=solid, 0=fluid
    """
    fluid    = mask_np < 0.5
    Ux, Uy   = y_field[0], y_field[1]
    dUy_dx   = np.gradient(Uy, axis=1)
    dUx_dy   = np.gradient(Ux, axis=0)
    vorticity = (dUy_dx - dUx_dy) ** 2
    return {
        "wake_energy":       float(np.mean(((Ux - 1.0)**2 + Uy**2)[fluid])),
        "pressure_variation": float(np.std(y_field[2][fluid])),
        "vorticity_energy":  float(np.mean(vorticity[fluid])),
    }

def add_complexity(results_df, dataset):
    metrics = []
    for i, row in enumerate(results_df.itertuples()):
        y_field  = results_df.iloc[i]["_true"].numpy()
        mask_np  = results_df.iloc[i]["_mask"].numpy()[0]
        metrics.append(compute_complexity(y_field, mask_np))
    return results_df.assign(**pd.DataFrame(metrics))

test_results  = add_complexity(test_results,  test_dataset)
train_results = add_complexity(train_results, train_dataset)

# Drop internal tensor columns before saving
drop_cols = ["_pred", "_true", "_mask"]
test_results.drop(columns=drop_cols).to_csv(
    f"{RESULTS_DIR}/test_results_fno.csv", index=False)
train_results.drop(columns=drop_cols).to_csv(
    f"{RESULTS_DIR}/train_results_fno.csv", index=False)

# Print correlation table
print("\n=== Test: Pearson correlation with Total_L2 ===")
for col in ["wake_energy", "pressure_variation", "vorticity_energy", "Re", "AoA"]:
    r = test_results["Total_L2"].corr(test_results[col])
    s, _ = spearmanr(test_results["Total_L2"], test_results[col])
    print(f"  {col:22s}  Pearson={r:+.3f}  Spearman={s:+.3f}")

# Plot Total_L2 vs complexity metrics coloured by geometry family
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes, ["wake_energy", "pressure_variation", "vorticity_energy"]):
    for fam in sorted(test_results["geometry_family"].unique()):
        sub = test_results[test_results["geometry_family"] == fam]
        ax.scatter(sub[metric], sub["Total_L2"], alpha=0.6, label=fam, s=25)
    r = test_results["Total_L2"].corr(test_results[metric])
    ax.set_xlabel(metric); ax.set_ylabel("Total_L2")
    ax.set_title(f"Total_L2 vs {metric}")
    ax.text(0.05, 0.95, f"r={r:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/error_vs_complexity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {RESULTS_DIR}/error_vs_complexity.png")