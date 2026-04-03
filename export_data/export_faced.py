"""
FACED dataset exporter  →  train.h5 / val.h5

Source: Processed_data/sub###.pkl  (28 trials, 32 ch, 7500 samples @ 250 Hz)
        Trial i = video i+1, all trials are exactly 30 s.

Split:  80 % of each participant's trials → train
        20 % of each participant's trials → val
        Every participant appears in BOTH splits.

Preprocessing applied on top of the pkl data:
    - band-pass 0.1–128 Hz  (FIR via MNE, or 4th-order Butterworth fallback)
    - notch 50 Hz + 60 Hz
    - resample 250 → 256 Hz  (baseline rate; per-model resampling at dataset fetch)

Output per row: (32, 7680)  =  32 channels × 30 s × 256 Hz
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.signal import resample, iirnotch, filtfilt
import h5py
from export_data.export_data import DataImport


# ── Video index (1-28) → emotion label (0-8) ──────────────
VIDEO_TO_EMOTION = {
    1: 0, 2: 0, 3: 0,               # Anger
    4: 1, 5: 1, 6: 1,               # Disgust
    7: 2, 8: 2, 9: 2,               # Fear
    10: 3, 11: 3, 12: 3,            # Sadness
    13: 4, 14: 4, 15: 4, 16: 4,     # Neutral
    17: 5, 18: 5, 19: 5,            # Amusement
    20: 6, 21: 6, 22: 6,            # Inspiration
    23: 7, 24: 7, 25: 7,            # Joy
    26: 8, 27: 8, 28: 8,            # Tenderness
}


class ImportFACED(DataImport):

    def import_data(self):
        return None

    def get_config(self):
        self.config = "downstream/info_dataset/faced.yaml"

    # ─────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────
    def import_data_to_hdf5(
        self,
        input_dir="/Volumes/Elements/EEG_data/downstream/FACED/Processed_data",
        output_dir="downstream/data/faced",
        val_ratio=0.2,
        random_seed=92,
        target_sfreq=256.0,
        num_channels=32,
        window_sec=10,
    ):
        """
        Read every sub###.pkl, apply notch + resample,
        split 80/20 PER PARTICIPANT, write train.h5 and val.h5.

        Args:
            window_sec: If set (e.g. 10), segment each 30 s trial into
                        non-overlapping windows of this duration.
                        Output shape per row: (C, window_sec * target_sfreq).
                        If None (default), keep full 30 s trials.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5 = output_dir / "train.h5"
        val_h5 = output_dir / "val.h5"

        # ── Discover pkl files ──
        pkls = sorted(input_dir.glob("sub*.pkl"))
        if len(pkls) == 0:
            raise FileNotFoundError(
                f"No sub###.pkl in {input_dir}. "
                "Download Processed_data from Synapse (syn50615881)."
            )
        print(f"Found {len(pkls)} subjects")

        rng = np.random.default_rng(random_seed)

        # ── Process each subject and split trials ──
        train_rows = []   # list of (eeg, label, participant, video_id)
        val_rows = []

        for pkl_path in pkls:
            part_nb = int(pkl_path.stem[3:])

            trials = self._load_and_preprocess(
                pkl_path, target_sfreq, num_channels,
            )
            if len(trials) == 0:
                continue

            # ── Optional windowing (e.g. 30s → 3 × 10s) ──
            if window_sec is not None:
                trials = self._window_trials(trials, target_sfreq, window_sec)

            # 80/20 split of this participant's trials
            n_trials = len(trials)
            n_val = max(1, int(round(n_trials * val_ratio)))
            perm = rng.permutation(n_trials)
            val_idx = set(perm[:n_val].tolist())

            for i, (eeg, label, vid) in enumerate(trials):
                row = (eeg, label, part_nb, vid)
                if i in val_idx:
                    val_rows.append(row)
                else:
                    train_rows.append(row)

            print(f"  {pkl_path.name}: {n_trials - n_val} train, {n_val} val")

        # ── Write HDF5 ──
        self._write_h5(train_rows, train_h5, "train", target_sfreq, num_channels)
        self._write_h5(val_rows, val_h5, "val", target_sfreq, num_channels)

        print(f"\nDone.  train: {len(train_rows)} rows  |  val: {len(val_rows)} rows")
        print(f"  {train_h5}")
        print(f"  {val_h5}")

    # ─────────────────────────────────────────────────
    # Load + preprocess one subject
    # ─────────────────────────────────────────────────
    def _load_and_preprocess(self, pkl_path, target_sfreq, num_channels):
        """
        Returns list of (eeg, emotion_label, video_id).
        eeg shape: (num_channels, T_resampled)
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="iso-8859-1")
        data = np.asarray(data, dtype=np.float64)     # (28, 32, N)

        n_trials, n_ch, N = data.shape

        # The pkl files are ALREADY preprocessed:
        #   - resampled to 250 Hz (regardless of original recording rate)
        #   - scaled to µV (regardless of original unit)
        # So we always use 250 Hz as the source sfreq, no unit conversion.
        orig_sfreq = 250.0

        # Keep first num_channels (32 = 30 EEG + A1 + A2)
        data = data[:, :num_channels, :]

        # ── Band-pass 0.1–128 Hz (ST-EEGFormer paper, Table D.2) ──
        # Applied before notch to remove DC drift and high-frequency noise
        try:
            from mne.filter import filter_data
            for t in range(n_trials):
                data[t] = filter_data(
                    data[t], sfreq=orig_sfreq,
                    l_freq=0.1, h_freq=124.0,
                    method="iir"
                )
        except ImportError:
            from scipy.signal import butter, filtfilt as _filtfilt
            # Fallback: 4th-order Butterworth band-pass
            nyq = orig_sfreq / 2.0
            b_bp, a_bp = butter(4, [0.1 / nyq, min(128.0, nyq - 1) / nyq], btype="band")
            for t in range(n_trials):
                data[t] = _filtfilt(b_bp, a_bp, data[t], axis=-1)

        # ── Notch 50 Hz + 60 Hz (applied per trial) ──
        for freq in [50.0, 60.0]:
            if freq < orig_sfreq / 2:   # only if below Nyquist
                b, a = iirnotch(freq, Q=30.0, fs=orig_sfreq)
                for t in range(n_trials):
                    data[t] = filtfilt(b, a, data[t], axis=-1)

        # ── Resample to target_sfreq ──
        if orig_sfreq != target_sfreq:
            n_out = int(round(N * target_sfreq / orig_sfreq))
            resampled = np.zeros((n_trials, num_channels, n_out), dtype=np.float32)
            for t in range(n_trials):
                resampled[t] = resample(data[t], n_out, axis=-1)
            data = resampled

        # ── Build trial list ──
        trials = []
        for t in range(n_trials):
            video_id = t + 1                        # trial 0 = video 1
            label = VIDEO_TO_EMOTION.get(video_id)
            if label is None:
                continue
            trials.append((data[t].astype(np.float32), label, video_id))

        return trials

    # ─────────────────────────────────────────────────
    # Windowing
    # ─────────────────────────────────────────────────
    @staticmethod
    def _window_trials(trials, sfreq, window_sec):
        """
        Segment each trial into non-overlapping windows.
        E.g. 30 s trial at 128 Hz with window_sec=10 → 3 windows of (C, 1280).
        Each window inherits the original trial's label and video_id.
        """
        win_samples = int(round(sfreq * window_sec))
        windowed = []
        for eeg, label, vid in trials:
            C, T = eeg.shape
            n_windows = T // win_samples
            for w in range(n_windows):
                start = w * win_samples
                end = start + win_samples
                windowed.append((eeg[:, start:end].copy(), label, vid))
        return windowed

    # ─────────────────────────────────────────────────
    # HDF5 writer
    # ─────────────────────────────────────────────────
    @staticmethod
    def _write_h5(rows, h5_path, split, target_sfreq, num_channels):
        if len(rows) == 0:
            print(f"  [{split}] nothing to write")
            return

        # All rows have the same shape after resampling
        C, T = rows[0][0].shape
        n = len(rows)

        with h5py.File(h5_path, "w") as f:
            x_ds = f.create_dataset("x", shape=(n, C, T), dtype="f4")
            y_ds = f.create_dataset("y", shape=(n,), dtype="i8")
            part_ds = f.create_dataset("participant", shape=(n,), dtype="i8")
            vid_ds = f.create_dataset("video_id", shape=(n,), dtype="i8")

            for i, (eeg, label, part_nb, vid) in enumerate(rows):
                x_ds[i] = eeg
                y_ds[i] = label
                part_ds[i] = part_nb
                vid_ds[i] = vid

            f.attrs["split"] = split
            f.attrs["n_samples"] = n
            f.attrs["sampling_rate"] = target_sfreq
            f.attrs["n_channels"] = num_channels
            f.attrs["n_classes"] = 9

        print(f"  [{split}] wrote {n} trials to {h5_path}")

    # ─────────────────────────────────────────────────
    # Base class requirement
    # ─────────────────────────────────────────────────
    def apply_preprocessing_pretrain(self, array):
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw_mne_object.resample(sfreq=128.0)
        return raw_mne_object.get_data()


if __name__ == "__main__":
    data_import = ImportFACED()
    data_import.import_data_to_hdf5(input_dir="/Users/sadeghemami/Downloads/FACEDprocess", window_sec=None)
    for split in ['train', 'val']:
        with h5py.File(f'downstream/data/faced/{split}.h5', 'r') as f:
            print(f'=== {split}.h5 ===')
            print(f'  x:           {f["x"].shape}')
            print(f'  y:           {f["y"].shape}   unique: {np.unique(f["y"][:])}')
            print(f'  participant: {f["participant"].shape}   unique: {np.unique(f["participant"][:])}')