"""
Ablation analysis: Covariance structure under MAD vs z-standardize normalization.

Generates a figure with 6 panels (2 rows × 3 cols):
  Row 1: z-standardize — covariance matrix, eigenvalue spectrum, variance ratios
  Row 2: global MAD     — covariance matrix, eigenvalue spectrum, variance ratios

The key argument: z-standardize forces unit variance per channel, collapsing
the covariance diagonal to identity and destroying the spatial geometry that
Riemannian attention exploits. MAD preserves relative channel variance ratios.

Usage:
    python analysis_cov_normalization.py
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

# ── Normalization functions (copied from MAE_pretraining/utils.py) ──

def standardize_channel(eeg):
    """Per-channel z-standardization. Destroys channel variance ratios."""
    mean = np.mean(eeg, axis=-1, keepdims=True)
    std = np.std(eeg, axis=-1, keepdims=True) + 1e-8
    return (eeg - mean) / std


UNIT_SCALE = {
    "mumtaz": 1e6,  # Volts → µV
}

def normalize_global(eeg, dataset_name=None):
    """Global MAD normalization. Preserves channel variance ratios."""
    if dataset_name is not None:
        scale = UNIT_SCALE.get(dataset_name, 1.0)
        eeg = eeg * scale
    eeg = eeg - np.mean(eeg)
    mad = np.median(np.abs(eeg - np.median(eeg))) + 1e-8
    eeg = eeg / (mad * 1.4826)
    return eeg


# ── Load data ──
data_path = "downstream/data/mumtaz/train.h5"
if not os.path.exists(data_path):
    print(f"Data not found at {data_path}")
    sys.exit(1)

with h5py.File(data_path, "r") as f:
    X = f["x"][:200]  # Take 200 samples for statistics
    participants = f["participant"][:200]

print(f"Loaded {X.shape[0]} samples, shape per sample: {X.shape[1:]}")
print(f"Raw value range: [{X.min():.6e}, {X.max():.6e}]")

n_samples, C, T = X.shape

# ── Compute covariance matrices under both normalizations ──

cov_zscore_list = []
cov_mad_list = []
var_ratios_zscore = []
var_ratios_mad = []

for i in range(n_samples):
    eeg_raw = X[i].copy()  # (C, T)

    # z-standardize
    eeg_z = standardize_channel(eeg_raw.copy())
    cov_z = np.cov(eeg_z)  # (C, C)
    cov_zscore_list.append(cov_z)
    ch_var_z = np.diag(cov_z)
    if ch_var_z.min() > 0:
        var_ratios_zscore.append(ch_var_z.max() / ch_var_z.min())

    # MAD
    eeg_m = normalize_global(eeg_raw.copy(), dataset_name="mumtaz")
    cov_m = np.cov(eeg_m)  # (C, C)
    cov_mad_list.append(cov_m)
    ch_var_m = np.diag(cov_m)
    if ch_var_m.min() > 0:
        var_ratios_mad.append(ch_var_m.max() / ch_var_m.min())

# Average covariance matrices
mean_cov_z = np.mean(cov_zscore_list, axis=0)
mean_cov_m = np.mean(cov_mad_list, axis=0)

# Eigenvalues
eig_z = np.sort(np.linalg.eigvalsh(mean_cov_z))[::-1]
eig_m = np.sort(np.linalg.eigvalsh(mean_cov_m))[::-1]

# Normalize eigenvalues to sum to 1 for comparison
eig_z_norm = eig_z / eig_z.sum()
eig_m_norm = eig_m / eig_m.sum()

# Effective rank (exponential of Shannon entropy of normalized eigenvalues)
def effective_rank(eigenvalues):
    e = eigenvalues / eigenvalues.sum()
    e = e[e > 1e-10]
    return np.exp(-np.sum(e * np.log(e)))

erank_z = effective_rank(eig_z)
erank_m = effective_rank(eig_m)

# ── Condition number ──
cond_z = eig_z[0] / (eig_z[-1] + 1e-10)
cond_m = eig_m[0] / (eig_m[-1] + 1e-10)

# ── Print statistics ──
print(f"\n{'='*60}")
print(f"Z-STANDARDIZE:")
print(f"  Diagonal range: [{np.diag(mean_cov_z).min():.4f}, {np.diag(mean_cov_z).max():.4f}]")
print(f"  Mean variance ratio (max/min per sample): {np.mean(var_ratios_zscore):.2f}")
print(f"  Condition number: {cond_z:.2f}")
print(f"  Effective rank: {erank_z:.2f} / {C}")
print(f"  Top-3 eigenvalue %: {eig_z_norm[:3].sum()*100:.1f}%")

print(f"\nGLOBAL MAD:")
print(f"  Diagonal range: [{np.diag(mean_cov_m).min():.4f}, {np.diag(mean_cov_m).max():.4f}]")
print(f"  Mean variance ratio (max/min per sample): {np.mean(var_ratios_mad):.2f}")
print(f"  Condition number: {cond_m:.2f}")
print(f"  Effective rank: {erank_m:.2f} / {C}")
print(f"  Top-3 eigenvalue %: {eig_m_norm[:3].sum()*100:.1f}%")
print(f"{'='*60}")

# ── Figure ──
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Channel labels (Mumtaz uses 19 channels, standard 10-20)
ch_labels = [f"Ch{i}" for i in range(C)]

# --- Row 1: Z-standardize ---
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(mean_cov_z, cmap="RdBu_r", aspect="equal",
                  vmin=-1.0, vmax=1.0)
ax1.set_title("Z-standardize: Covariance matrix", fontsize=12, fontweight="bold")
ax1.set_xlabel("Channel")
ax1.set_ylabel("Channel")
plt.colorbar(im1, ax=ax1, shrink=0.8)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(range(C), eig_z_norm, color="#d62728", alpha=0.8, edgecolor="black", linewidth=0.5)
ax2.set_title(f"Z-std: Eigenvalue spectrum\n(eff. rank = {erank_z:.1f}/{C})", fontsize=12, fontweight="bold")
ax2.set_xlabel("Component index")
ax2.set_ylabel("Normalized eigenvalue")
ax2.set_ylim(0, max(eig_z_norm.max(), eig_m_norm.max()) * 1.1)

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(range(C), np.diag(mean_cov_z), color="#d62728", alpha=0.8, edgecolor="black", linewidth=0.5)
ax3.set_title(f"Z-std: Channel variances\n(ratio max/min = {np.mean(var_ratios_zscore):.1f}×)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Channel index")
ax3.set_ylabel("Variance")
ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Unit variance")
ax3.legend(fontsize=9)

# --- Row 2: Global MAD ---
vmax_m = np.abs(mean_cov_m).max()
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(mean_cov_m, cmap="RdBu_r", aspect="equal",
                  vmin=-vmax_m, vmax=vmax_m)
ax4.set_title("Global MAD: Covariance matrix", fontsize=12, fontweight="bold")
ax4.set_xlabel("Channel")
ax4.set_ylabel("Channel")
plt.colorbar(im4, ax=ax4, shrink=0.8)

ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(C), eig_m_norm, color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
ax5.set_title(f"MAD: Eigenvalue spectrum\n(eff. rank = {erank_m:.1f}/{C})", fontsize=12, fontweight="bold")
ax5.set_xlabel("Component index")
ax5.set_ylabel("Normalized eigenvalue")
ax5.set_ylim(0, max(eig_z_norm.max(), eig_m_norm.max()) * 1.1)

ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(range(C), np.diag(mean_cov_m), color="#1f77b4", alpha=0.8, edgecolor="black", linewidth=0.5)
ax6.set_title(f"MAD: Channel variances\n(ratio max/min = {np.mean(var_ratios_mad):.1f}×)", fontsize=12, fontweight="bold")
ax6.set_xlabel("Channel index")
ax6.set_ylabel("Variance")

fig.suptitle("Effect of normalization on covariance geometry (Mumtaz, N=200 windows)",
             fontsize=14, fontweight="bold", y=0.98)

out_path = "analysis_cov_normalization.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
