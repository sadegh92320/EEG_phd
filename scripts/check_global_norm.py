"""
Diagnostic script: verify that normalize_global produces sensible values
for every pretraining dataset.

Run on Colab:
    !cd /content/drive/MyDrive/paper_1_code && python scripts/check_global_norm.py

What to look for:
    - "raw |mean|" varies wildly across datasets (different units/DC offsets)
    - "raw std" also varies (different amplifiers, units)
    - AFTER normalize_global:
        * "post |mean|" should be ~0 for all datasets (centering works)
        * "post std" should be ~1-3 for all datasets (MAD scaling works)
        * "post var_ratio" is the KEY metric: this is what Riemannian attention
          sees. It should be >1 (channels have different variances). If it's
          ~1.0 for every dataset, something is wrong (same as z-standardization).
          Values of 2-50× are healthy. Values >1000× suggest a bad channel.
    - Compare "z-std var_ratio" (always ~1.0) vs "post var_ratio" (preserved)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import h5py
import yaml
from pathlib import Path
from MAE_pretraining.utils import normalize_global, standardize_channel, UNIT_SCALE
from MAE_pretraining.pretrain_dataset import get_pretrain_dataset

# Load config to get dataset list
with open("MAE_pretraining/setting_pretraining.yaml") as f:
    config = yaml.safe_load(f)

datasets = config["data_use"]
N_SAMPLES = 10  # check 10 random samples per dataset

print(f"{'Dataset':<18} {'raw |mean|':>10} {'raw std':>10} {'post |mean|':>11} {'post std':>10} {'post var_ratio':>14} {'z-std var_ratio':>15}")
print("-" * 100)

for ds_name in datasets:
    try:
        dataset = get_pretrain_dataset(ds_name, type="train", use_global_norm=False)
    except Exception as e:
        print(f"{ds_name:<18} SKIP: {e}")
        continue

    raw_means, raw_stds = [], []
    post_means, post_stds, post_ratios = [], [], []
    zstd_ratios = []

    n = min(N_SAMPLES, len(dataset))
    indices = np.random.RandomState(42).choice(len(dataset), n, replace=False)

    for idx in indices:
        eeg_tensor, _ = dataset[idx]
        # Undo the z-standardization that __getitem__ applied
        # We need raw data, so read directly from h5
        pass

    # Read raw from h5 directly
    h5_path = dataset.h5_file_path
    try:
        with h5py.File(h5_path, "r") as f:
            total = f["x"].shape[0]
            indices = np.random.RandomState(42).choice(total, min(N_SAMPLES, total), replace=False)

            for idx in indices:
                eeg = f["x"][idx].astype(np.float64)  # (C, T)

                # Raw stats (before any normalization)
                raw_means.append(np.abs(np.mean(eeg)))
                raw_stds.append(np.std(eeg))

                # After global MAD normalization
                eeg_norm = normalize_global(eeg.copy(), dataset_name=ds_name)
                post_means.append(np.abs(np.mean(eeg_norm)))
                post_stds.append(np.std(eeg_norm))
                ch_var = np.var(eeg_norm, axis=1)
                if np.min(ch_var) > 0:
                    post_ratios.append(np.max(ch_var) / np.min(ch_var))
                else:
                    post_ratios.append(float('inf'))

                # After z-standardization (for comparison)
                eeg_z = standardize_channel(eeg.copy())
                ch_var_z = np.var(eeg_z, axis=1)
                if np.min(ch_var_z) > 0:
                    zstd_ratios.append(np.max(ch_var_z) / np.min(ch_var_z))
                else:
                    zstd_ratios.append(float('inf'))

    except Exception as e:
        print(f"{ds_name:<18} ERROR reading h5: {e}")
        continue

    print(f"{ds_name:<18} {np.mean(raw_means):>10.1f} {np.mean(raw_stds):>10.1f} "
          f"{np.mean(post_means):>11.4f} {np.mean(post_stds):>10.2f} "
          f"{np.median(post_ratios):>14.1f} {np.median(zstd_ratios):>15.2f}")

print()
print("KEY:")
print("  raw |mean|    — before normalization (shows DC offset / unit differences)")
print("  raw std       — before normalization (shows scale differences)")
print("  post |mean|   — after global MAD (should be ~0)")
print("  post std      — after global MAD (should be ~1-3)")
print("  post var_ratio — max/min channel variance AFTER global MAD (SHOULD BE >1)")
print("  z-std var_ratio — max/min channel variance after z-standardization (always ~1)")
print()
print("If post var_ratio >> 1 and z-std var_ratio ≈ 1, global MAD preserves geometry.")
print("If im post std is wildly different from others, its UNIT_SCALE is wrong.")
