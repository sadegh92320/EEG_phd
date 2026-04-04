from scipy.signal import resample
import numpy as np

def resample_eeg(eeg, previous_freq, new_freq):
    """Resample data with new frequency"""
    B, C, T = eeg.shape
    new_t = int(round((new_freq*T)/previous_freq))
    resample_data = resample(x=eeg,num=new_t,axis=2)
    return resample_data


def standardize_channel(eeg):
    """Per-channel z-standardization. Destroys channel variance ratios."""
    #size of eeg C,T (single sample from __getitem__)
    mean = np.mean(eeg, axis=-1, keepdims=True)
    std = np.std(eeg, axis=-1, keepdims=True) + 1e-8
    eeg = (eeg - mean)/std
    return eeg


# ── Conversion factors: dataset_name → multiply raw data to get µV ──
# Determined empirically from check_data_units.py
UNIT_SCALE = {
    "bci_comp_iv2a": 1e6,   # Volts → µV
    "bci_comp_iv2b": 1e6,   # Volts → µV
    "hgd":           1e6,   # Volts → µV
    "seed2":         1e6,   # Volts → µV (SEED stores in V)
    "physionet":     1e6,   # Volts → µV
    "p300":          1.0,   # already µV — BioSemi ActiveTwo, .mat in µV (Won et al. 2022)
    "ssvep":         1.0,   # already µV — BETA dataset, Neuroscan SynAmps2, .mat in µV
    "eeg_mi_bci":    1.0,   # already µV (but large — ~399 µV mean, check for artifacts)
    "auditory":      1.0,   # already µV — KU/SparrKULee, .npy provided by authors in µV
    "mi":            1.0,   # already µV — Lee2019, BrainAmp, .mat in µV (MOABB confirms)
    "online":        1.0,   # already µV — Neuroscan SynAmps RT, .mat in µV (var_ratio=5453 = bad channels)
    "im":            1.0,   # likely µV with DC offset (|mean|≈1260 from baseline shift, removed by centering)
}


def normalize_global(eeg, dataset_name=None, clamp_channels=False):
    """
    Global normalization that PRESERVES channel variance ratios.

    1. Convert to microvolts (if dataset_name provided)
    2. (Optional) Clamp outlier channels whose variance > 10× median
    3. Remove global mean (not per-channel)
    4. Divide by global robust scale (MAD across all channels + time)

    This keeps the relative variance structure across channels intact,
    which is the geometric signal that Riemannian attention exploits.

    Args:
        eeg: (C, T) single sample
        dataset_name: str — used to look up unit conversion factor
        clamp_channels: if True, scale down channels with variance > 10×
                        the median channel variance. Use if training is
                        unstable due to noisy electrodes; leave off by
                        default to let the model learn robustness.
    Returns:
        eeg: (C, T) normalized, values roughly in [-10, 10]
    """
    # Step 1: Convert to µV
    if dataset_name is not None:
        scale = UNIT_SCALE.get(dataset_name, 1.0)
        eeg = eeg * scale

    # Step 2 (optional): Clamp outlier channels (bad electrodes / impedance drift)
    # Ceiling: scale down channels with variance > 10× median (noisy electrodes)
    # Floor:   scale up channels with variance < 0.1× median (near-dead electrodes)
    # This bounds the max/min variance ratio to ~100, preventing rank-1 covariances
    if clamp_channels:
        ch_var = np.var(eeg, axis=1)               # (C,)
        median_var = np.median(ch_var)
        if median_var > 0:
            ceil = 10.0 * median_var
            floor = 0.1 * median_var
            for c in range(eeg.shape[0]):
                if ch_var[c] > ceil:
                    eeg[c] *= np.sqrt(ceil / ch_var[c])
                elif ch_var[c] < floor and ch_var[c] > 0:
                    eeg[c] *= np.sqrt(floor / ch_var[c])

    # Step 3: Global centering (same offset for all channels)
    eeg = eeg - np.mean(eeg)

    # Step 4: Global robust scaling (same scale for all channels)
    # MAD (median absolute deviation) is robust to outliers/artifacts
    mad = np.median(np.abs(eeg - np.median(eeg))) + 1e-8
    eeg = eeg / (mad * 1.4826)  # 1.4826 makes MAD comparable to std for Gaussian

    return eeg