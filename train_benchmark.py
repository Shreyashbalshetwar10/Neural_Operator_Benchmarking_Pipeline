"""
train_benchmark.py
------------------
Trains and evaluates FNO, WNO, or Attention-FNO on the synthetic
CFD benchmark dataset under identical conditions for fair comparison.

Usage:
    python train_benchmark.py --model fno
    python train_benchmark.py --model wno
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import distance_transform_edt

from models.fno_baseline import FNO2d, UnitGaussianNormalizer
from models.wno_baseline import WNO2d

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["fno", "wno"],
                    required=True, help="Architecture to train")
args = parser.parse_args()
MODEL_NAME = args.model

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "simulation/dataset/dataset_128.npz"
SAVE_DIR   = "checkpoints_sim"
SAVE_PATH  = f"{SAVE_DIR}/{MODEL_NAME}_benchmark_best.pth"
RESULTS_DIR = "results_sim"

MODES      = 16
WIDTH      = 64
BATCH_SIZE = 8
EPOCHS     = 500
LR         = 1e-3

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
d        = np.load(DATA_PATH)
data     = torch.from_numpy(d["data"]).float()
Re_vals  = torch.from_numpy(d["Re"].astype(np.float32))
AoA_vals = torch.from_numpy(d["AoA"].astype(np.float32))
families = d["families"]

meta_df = pd.DataFrame({
    "geometry_family": families,
    "Re":  Re_vals.numpy(),
    "AoA": AoA_vals.numpy(),
})

N, C, H, W = data.shape
mask  = data[:, 0:1, :, :]
y_out = data[:, 1:, :, :]

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

    print(f"Split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    return (make_ds(train_idx), make_ds(val_idx), make_ds(test_idx),
            train_idx, val_idx, test_idx)


train_ds, val_ds, test_ds, train_idx, val_idx, test_idx = stratified_split(
    x_in, y_out, mask, meta_df)

y_normalizer = UnitGaussianNormalizer(train_ds.tensors[2], train_ds.tensors[1])

def encode_ds(ds):
    x, y, m, idx = ds.tensors
    return TensorDataset(x, y_normalizer.encode(m, y), m, idx)

train_dataset = encode_ds(train_ds)
val_dataset   = encode_ds(val_ds)
test_dataset  = encode_ds(test_ds)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training [{MODEL_NAME}] on {device}")

y_normalizer.to(device)

if MODEL_NAME == "fno":
    model = FNO2d(modes=MODES, width=WIDTH, num_in_channels=6, num_out_channels=3)
elif MODEL_NAME == "wno":
    model = WNO2d(width=WIDTH, level=2, num_in_channels=6, num_out_channels=3)
    
model.to(device)

def fluid_mse(out, y, m):
    fluid = (m < 0.5).float()
    return ((out - y)**2 * fluid).sum() / (fluid.sum() * out.shape[1] + 1e-8)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                  factor=0.5, patience=20)

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
        out = model(x)
        loss = fluid_mse(out, y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y, m, _ in val_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            out = model(x)
            val_loss += fluid_mse(out, y, m).item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss, "epoch": epoch}, SAVE_PATH)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:4d} | train {avg_train:.6f} | "
              f"val {avg_val:.6f} | best {best_val_loss:.6f}")

print(f"Training done in {(time.time()-t0)/60:.1f} min")

# ── Evaluation ────────────────────────────────────────────────────────────────
def relative_l2(pred, true, m):
    fluid = (m < 0.5).float().to(pred.device)
    return (torch.norm((pred-true)*fluid) / (torch.norm(true*fluid) + 1e-8)).item()

ckpt = torch.load(SAVE_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def eval_loader(loader, name):
    ux_l2, uy_l2, p_l2, tot_l2 = [], [], [], []
    with torch.no_grad():
        for x, y_enc, m, _ in loader:
            x, y_enc, m = x.to(device), y_enc.to(device), m.to(device)
            out = model(x)
            pred = y_normalizer.decode(out)
            true = y_normalizer.decode(y_enc)
            for i in range(pred.shape[0]):
                mi = m[i:i+1]
                ux_l2.append(relative_l2(pred[i,0:1], true[i,0:1], mi))
                uy_l2.append(relative_l2(pred[i,1:2], true[i,1:2], mi))
                p_l2.append( relative_l2(pred[i,2:3], true[i,2:3], mi))
                tot_l2.append(relative_l2(pred[i:i+1], true[i:i+1], mi))

    print(f"\n=== {name} [{MODEL_NAME}] ===")
    print(f"  Ux    : {np.mean(ux_l2):.4f} ± {np.std(ux_l2):.4f}")
    print(f"  Uy    : {np.mean(uy_l2):.4f} ± {np.std(uy_l2):.4f}")
    print(f"  p     : {np.mean(p_l2):.4f}  ± {np.std(p_l2):.4f}")
    print(f"  Total : {np.mean(tot_l2):.4f} ± {np.std(tot_l2):.4f}")
    return np.mean(tot_l2)

eval_loader(test_loader,  "Test")
eval_loader(train_loader, "Train")

# Per-geometry breakdown
print(f"\n=== Per-geometry test error [{MODEL_NAME}] ===")
all_fam, all_err = [], []
with torch.no_grad():
    for i in range(len(test_dataset)):
        x, y_enc, m, idx = test_dataset[i]
        x     = x.unsqueeze(0).to(device)
        y_enc = y_enc.unsqueeze(0).to(device)
        m_dev = m.unsqueeze(0).to(device)
        out   = model(x)
        pred = y_normalizer.decode(out)
        true = y_normalizer.decode(y_enc)
        all_fam.append(meta_df.iloc[idx.item()]["geometry_family"])
        all_err.append(relative_l2(pred, true, m_dev))

results = pd.DataFrame({"geometry_family": all_fam, "Total_L2": all_err})
print(results.groupby("geometry_family")["Total_L2"].agg(["mean","std","count"]).round(4))
results.to_csv(f"{RESULTS_DIR}/{MODEL_NAME}_test_results.csv", index=False)
print(f"\nResults saved to {RESULTS_DIR}/{MODEL_NAME}_test_results.csv")