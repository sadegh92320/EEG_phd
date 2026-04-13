from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly
from math import gcd


MODEL_PREPROCESS_CONFIG = {
    "steegformer":  {"norm": {"method": "z_standardize"},            "sfreq": 128},
    "labram":       {"norm": {"method": "rescale", "scale": 1e-4},  "sfreq": 200},
    "biot":         {"norm": {"method": "percentile_95"},            "sfreq": 200},
    "cbramod":      {"norm": {"method": "rescale", "scale": 1e-3},   "sfreq": 200},  # ×1e-3: µV → mV 
    "eegpt":        {"norm": {"method": "rescale", "scale": 1e-3},  "sfreq": 256},  # µV → mV (V→µV conversion handled by data_unit flag)
    "bendr":        {"norm": {"method": "minmax_neg1_1"},            "sfreq": 256},
    # Your own pretrained models (pretrained at 128 Hz)
    # Riemannian models use global_mad to preserve channel variance ratios
    # (per-channel z-std collapses covariance → eigenvalues ≈ 1 → Padé ≈ S-I)
    "baseline":          {"norm": {"method": "global_mad"},  "sfreq": 128},
    "encoder_gnn":       {"norm": {"method": "global_mad"},  "sfreq": 128},
    "riemann_loss":      {"norm": {"method": "global_mad"},  "sfreq": 128},
    "riemann_para":      {"norm": {"method": "global_mad"},  "sfreq": 128},
    "riemann_adaptive":  {"norm": {"method": "global_mad"},  "sfreq": 128},
    "riemann_ema":       {"norm": {"method": "global_mad"},  "sfreq": 128},
    "riemann_seq":       {"norm": {"method": "global_mad"},  "sfreq": 128},
    # Classic NN baselines run at the baseline 256 Hz with z-standardization
    "default":      {"norm": {"method": "z_standardize"},            "sfreq": 256},
}


class Downstream_Dataset(Dataset):
    def __init__(
        self,
        X,
        y,
        fold=0,
        config_path=None,
        classification_task=None,
        data_length=None,
        normalize=True,
        norm_mode="default",
        base_sfreq=256,
    ):
        self.classification_task = classification_task
        self.data_length = data_length
        self.data = np.array(X)
        self.class_label = np.array(y)
        self.normalize = normalize
        self.fold = fold

        # Per-model preprocessing (normalization + resampling)
        preprocess = MODEL_PREPROCESS_CONFIG.get(norm_mode, MODEL_PREPROCESS_CONFIG["default"])
        self.norm_config = preprocess["norm"]
        self.target_sfreq = preprocess["sfreq"]
        self.base_sfreq = base_sfreq

        # Precompute resample ratio (only resample if target != base)
        self._needs_resample = (self.target_sfreq != self.base_sfreq)
        if self._needs_resample:
            g = gcd(self.target_sfreq, self.base_sfreq)
            self._resample_up = self.target_sfreq // g
            self._resample_down = self.base_sfreq // g

        if norm_mode == "steegformer":
            with open(Path("downstream/info_dataset/steegformer_channel_info.yaml"), "r") as file:
                self.channel_config = yaml.safe_load(file)
        else:
            with open(Path("downstream/info_dataset/channel_info.yaml"), "r") as file:
                self.channel_config = yaml.safe_load(file)
            

        if config_path is None:
            raise ValueError("config_path must be provided")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.channel_list = self.config.get("channel_list", [])
        self.channel_id = self.get_chan_idx()
        self.channel_num = len(self.channel_id)

        # Unit conversion: if data is stored in Volts, convert to µV
        # so that model-specific rescaling (e.g. ×1e-3 for EEGPT = µV→mV) works as intended.
        data_unit = self.config.get("data_unit", "uV")  # default assumes µV
        self._v_to_uv = (data_unit.upper() == "V")

    def get_chan_idx(self):
        channel_id = []
        channel_list = [ch.lower() for ch in self.channel_list]
        chan_map = {
            key.lower(): val
            for key, val in self.channel_config["channels_mapping"].items()
        }

        next_id = max(chan_map.values()) + 1 if chan_map else 0
        for ch in channel_list:
            idx = chan_map.get(ch)
            if idx is None:
                # Bipolar derivations (e.g. Fpz-Cz) or unknown channels:
                # assign sequential IDs beyond the known mapping.
                # These won't match any pretrained positional embedding,
                # which is correct — models will use default/no pos embed.
                idx = next_id
                next_id += 1
            channel_id.append(idx)
        return channel_id

    def __len__(self):
        return len(self.class_label)

    def _normalize(self, trial_data):
        """Apply model-specific normalization"""
        method = self.norm_config["method"]

        if method == "global_mad":
            # Global MAD normalization — one scale factor for all channels.
            # Preserves inter-channel variance ratios (critical for Riemannian
            # attention: covariance eigenvalues stay meaningful for Padé log map).
            # Channel clamping: floor (0.1× median var) + ceiling (10× median var)
            # to bound max variance ratio at 100, handling dead/noisy electrodes.
            ch_var = np.var(trial_data, axis=1)
            median_var = np.median(ch_var)
            if median_var > 0:
                ceil = 10.0 * median_var
                floor = 0.1 * median_var
                for c in range(trial_data.shape[0]):
                    if ch_var[c] > ceil:
                        trial_data[c] *= np.sqrt(ceil / ch_var[c])
                    elif ch_var[c] < floor and ch_var[c] > 0:
                        trial_data[c] *= np.sqrt(floor / ch_var[c])
            trial_data = trial_data - np.mean(trial_data)
            mad = np.median(np.abs(trial_data - np.median(trial_data))) + 1e-8
            return trial_data / (mad * 1.4826)

        elif method == "z_standardize":
            # Zero mean, unit variance per channel
            mean = np.mean(trial_data, axis=1, keepdims=True)
            std = np.std(trial_data, axis=1, keepdims=True) + 1e-6
            return (trial_data - mean) / std

        elif method == "rescale":
            # Multiply by scale factor (e.g. µV → mV)
            scale = self.norm_config["scale"]
            return trial_data * scale

        elif method == "percentile_95":
            # Divide each channel by its 95th percentile of absolute values
            p95 = np.percentile(np.abs(trial_data), 95, axis=1, keepdims=True) + 1e-6
            return trial_data / p95

        elif method == "minmax_neg1_1":
            # Scale each channel to [-1, 1]
            cmin = trial_data.min(axis=1, keepdims=True)
            cmax = trial_data.max(axis=1, keepdims=True)
            denom = (cmax - cmin) + 1e-6
            return 2.0 * (trial_data - cmin) / denom - 1.0

        else:
            return trial_data

    def _resample(self, trial_data):
        """
        Resample from base_sfreq to target_sfreq using polyphase filtering.
        Input/output shape: (C, T) → (C, T').
        E.g. 256→128 Hz: (32, 7680) → (32, 3840)
             256→200 Hz: (32, 7680) → (32, 6000)
        """
        return resample_poly(trial_data, self._resample_up, self._resample_down, axis=1)

    def __getitem__(self, idx):
        trial_data = self.data[idx].copy()
        label = self.class_label[idx]

        if self.data_length is not None:
            trial_data = trial_data[:, :self.data_length]

        # Convert Volts → µV if needed (before clipping and normalization)
        if self._v_to_uv:
            trial_data = trial_data * 1e6

        # Resample to model's native rate BEFORE normalization
        if self._needs_resample:
            trial_data = self._resample(trial_data)

        # No hard clipping — let each model's normalization handle the scale.
        # Previous ±500µV clip was too aggressive for datasets like Mumtaz (±3400µV range)
        # and not used by CBraMod/EEGPT papers in their downstream evaluation.

        if self.normalize:
            trial_data = self._normalize(trial_data)

        # Regression labels (e.g. PERCLOS float) must stay float;
        # classification labels are cast to long.
        if isinstance(label, (float, np.floating)):
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return (
            torch.from_numpy(trial_data).float(),
            label_tensor,
            torch.tensor(self.channel_id, dtype=torch.long),
        )