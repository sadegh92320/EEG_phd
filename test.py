import h5py
import numpy as np

with h5py.File("downstream/data/faced/train.h5", "r") as f:
    eeg = f["x"][0]  # first sample
    print(f"Shape: {eeg.shape}")
    print(f"Min: {eeg.min():.4f}, Max: {eeg.max():.4f}")
    print(f"Mean: {eeg.mean():.4f}, Std: {eeg.std():.4f}")
    print(f"Abs range: {np.abs(eeg).max():.4f}")