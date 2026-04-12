"""
SEED-VIG vigilance estimation dataset exporter  →  train.h5 / val.h5

Source: Raw EEG .mat files + PERCLOS label .mat files
        23 subjects, 17 EEG channels, 200 Hz native (kept as-is).
        Continuous driving ~2 h per subject.

Data format (from .mat inspection):
    EEG.data         : (T, 17)  float64   — raw EEG samples
    EEG.chn          : (1, 17)  cell      — channel name strings
    EEG.sample_rate  : scalar   200       — sampling rate in Hz
    EEG.node_number  : scalar   17        — number of channels
    perclos          : (885, 1) float64   — PERCLOS vigilance score ∈ [0, 1]

Each file is named like: {subject_id}_{date}_{time}_{session}.mat
One PERCLOS label per 8 seconds of recording at 200 Hz (1600 samples).

Split:  80 % of each participant's segments → train (random)
        20 % of each participant's segments → val  (random)
        Every participant appears in BOTH splits (LOO done downstream).

Preprocessing:
    - band-pass 0.1–50 Hz (IIR via MNE)
    - no resampling (kept at native 200 Hz; per-model resampling done at load time)

Segmentation: 5-second non-overlapping windows (matching STEEGFormer paper).
              Labels interpolated from PERCLOS to match window centers.
"""

import re
import numpy as np
import mne
import h5py
import scipy.io as sio
from pathlib import Path
from export_data.export_data import DataImport



# 17 EEG channels in SEED-VIG (10-20 montage, no CPZ in actual data)
KEEP_CHANNELS = [
    "FT7", "FT8", "T7", "T8", "TP7", "TP8",
    "CP1", "CP2", "P1", "PZ", "P2",
    "PO3", "POZ", "PO4", "O1", "OZ", "O2",
]
assert len(KEEP_CHANNELS) == 17

TARGET_SFREQ = 200.0   # Keep native 200 Hz (same as STEEGFormer); per-model resampling at dataset fetch
SEGMENT_SEC = 5         # STEEGFormer paper: "predict vigilance level from 5-s EEG epochs"


class ImportSeedVig(DataImport):

    def import_data(self):
        return None

    def get_config(self):
        self.config = "downstream/info_dataset/seed_vig.yaml"

    # ─────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────
    def import_data_to_hdf5(
        self,
        raw_dir,
        label_dir,
        output_dir="downstream/data/seed_vig",
        val_ratio=0.2,
        random_seed=92,
    ):
        """
        Read raw .mat files + PERCLOS .mat labels, preprocess, segment,
        split 80/20 PER PARTICIPANT (temporal order), write train.h5 and val.h5.

        Parameters
        ----------
        raw_dir : str or Path
            Folder containing raw EEG .mat files (e.g. 1_20151124_noon_2.mat).
        label_dir : str or Path
            Folder containing PERCLOS label .mat files (same naming convention).
        output_dir : str or Path
            Output directory for train.h5 / val.h5.
        val_ratio : float
            Fraction of each participant's segments for validation (random).
        random_seed : int
            Random seed for reproducibility.
        """
        raw_dir = Path(raw_dir)
        label_dir = Path(label_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5 = output_dir / "train.h5"
        val_h5 = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # ── Discover .mat files ──
        raw_files = sorted(raw_dir.glob("*.mat"))
        label_files = sorted(label_dir.glob("*.mat"))

        if len(raw_files) == 0:
            raise FileNotFoundError(f"No .mat files found in {raw_dir}")
        if len(label_files) == 0:
            raise FileNotFoundError(f"No .mat label files found in {label_dir}")

        # ── Match raw ↔ label files by stem (identical filenames) ──
        raw_by_stem = {f.stem: f for f in raw_files}
        label_by_stem = {f.stem: f for f in label_files}

        common_stems = sorted(set(raw_by_stem.keys()) & set(label_by_stem.keys()))
        if len(common_stems) == 0:
            raise ValueError(
                f"No matching filenames between raw ({raw_dir}) and labels ({label_dir}).\n"
                f"  Raw stems:   {sorted(raw_by_stem.keys())[:5]}...\n"
                f"  Label stems: {sorted(label_by_stem.keys())[:5]}..."
            )

        # Group files by subject ID (leading number before first '_')
        subject_files = {}  # subject_id → list of (raw_path, label_path)
        for stem in common_stems:
            sid = self._extract_subject_id(stem)
            if sid not in subject_files:
                subject_files[sid] = []
            subject_files[sid].append((raw_by_stem[stem], label_by_stem[stem]))

        subject_ids = sorted(subject_files.keys())
        print(f"Found {len(common_stems)} recordings from {len(subject_ids)} subjects: {subject_ids}")

        win_samples = int(round(TARGET_SFREQ * SEGMENT_SEC))  # 1280 at 256 Hz
        print(f"Segment: {SEGMENT_SEC}s × {TARGET_SFREQ} Hz = {win_samples} samples")

        train_rows = []  # (eeg, label, participant_id)
        val_rows = []

        for part_idx, sid in enumerate(subject_ids):
            recordings = subject_files[sid]
            print(f"\n── Subject {sid} (idx={part_idx}): {len(recordings)} recording(s) ──")

            all_segments = []
            all_labels = []

            for raw_path, label_path in recordings:
                print(f"  Loading {raw_path.name}")

                try:
                    segs, labs = self._load_recording(raw_path, label_path, win_samples)
                except Exception as e:
                    print(f"  ERROR: {e}, skipping {raw_path.name}")
                    continue

                all_segments.extend(segs)
                all_labels.extend(labs)

            if len(all_segments) == 0:
                print(f"  No segments extracted, skipping subject {sid}")
                continue

            # 80/20 random split of this participant's segments
            n_seg = len(all_segments)
            n_val = max(1, int(round(n_seg * val_ratio)))
            perm = rng.permutation(n_seg)
            val_idx = set(perm[:n_val].tolist())

            for i in range(n_seg):
                row = (all_segments[i], all_labels[i], part_idx)
                if i in val_idx:
                    val_rows.append(row)
                else:
                    train_rows.append(row)

            print(f"  {n_seg} segments ({n_seg - n_val} train, {n_val} val)")

        # ── Write HDF5 ──
        self._write_h5(train_rows, train_h5, "train")
        self._write_h5(val_rows, val_h5, "val")

        print(f"\nDone.  train: {len(train_rows)} rows  |  val: {len(val_rows)} rows")
        print(f"  {train_h5}")
        print(f"  {val_h5}")

    # ─────────────────────────────────────────────────
    # Subject ID extraction
    # ─────────────────────────────────────────────────
    @staticmethod
    def _extract_subject_id(stem):
        """Extract numeric subject ID from filename stem (leading number)."""
        m = re.match(r"^(\d+)", stem)
        return int(m.group(1)) if m else stem

    # ─────────────────────────────────────────────────
    # Load one recording: raw EEG .mat + PERCLOS .mat
    # ─────────────────────────────────────────────────
    def _load_recording(self, raw_path, label_path, win_samples):
        """
        Returns:
            segments: list of np.array (C, T) in float32, C=17, T=win_samples
            labels:   list of float (PERCLOS values)
        """
        # ── Load PERCLOS labels ──
        lab_mat = sio.loadmat(str(label_path))
        if "perclos" not in lab_mat:
            # Fallback: first non-metadata key
            data_keys = [k for k in lab_mat.keys() if not k.startswith("__")]
            raise ValueError(
                f"Key 'perclos' not found in {label_path.name}. "
                f"Available keys: {data_keys}"
            )
        perclos = lab_mat["perclos"].flatten().astype(np.float64)
        print(f"    PERCLOS: {len(perclos)} labels, range [{perclos.min():.3f}, {perclos.max():.3f}]")

        # ── Load raw EEG from .mat ──
        raw_mat = sio.loadmat(str(raw_path))
        if "EEG" not in raw_mat:
            raise ValueError(
                f"Key 'EEG' not found in {raw_path.name}. "
                f"Available keys: {[k for k in raw_mat.keys() if not k.startswith('__')]}"
            )

        eeg_struct = raw_mat["EEG"][0, 0]
        data = eeg_struct["data"]             # (T, n_channels) float64
        chn_raw = eeg_struct["chn"][0]        # array of arrays of strings
        sfreq = float(eeg_struct["sample_rate"].flat[0])

        # Parse channel names
        ch_names = [str(c[0]) for c in chn_raw]
        n_samples, n_ch = data.shape

        print(f"    EEG: {n_ch} channels, {n_samples} samples, {sfreq} Hz "
              f"({n_samples / sfreq:.0f}s)")
        print(f"    Channels: {ch_names}")

        # ── Validate channels ──
        # Map loaded channel names (uppercase) to our expected list
        ch_upper = [c.upper() for c in ch_names]
        keep_upper = [c.upper() for c in KEEP_CHANNELS]

        # Find indices for our target channels in order
        ch_indices = []
        matched_names = []
        for target in keep_upper:
            if target in ch_upper:
                ch_indices.append(ch_upper.index(target))
                matched_names.append(target)

        if len(ch_indices) != len(KEEP_CHANNELS):
            missing = set(keep_upper) - set(ch_upper)
            print(f"    WARNING: Missing channels: {missing}")
            print(f"    Matched {len(ch_indices)}/{len(KEEP_CHANNELS)}")

        # Reorder data to standard channel order: (T, n_matched) → (n_matched, T)
        data = data[:, ch_indices].T.astype(np.float64)  # (C, T)
        C = data.shape[0]

        # ── Create MNE Raw for preprocessing ──
        info = mne.create_info(
            ch_names=matched_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info, verbose="ERROR")

        # ── Preprocessing ──
        nyq = sfreq / 2.0
        h_freq = min(50.0, nyq - 1.0)
        raw.filter(l_freq=0.1, h_freq=h_freq, method="iir", verbose="ERROR")

        # No resampling — keep native 200 Hz (per-model resampling at dataset fetch)
        data = raw.get_data()  # (C, T)
        C, T_total = data.shape
        effective_sfreq = sfreq

        print(f"    After filtering: {C} channels, {T_total} samples, {effective_sfreq} Hz")

        # ── Compute label timing ──
        # Each PERCLOS label covers (T_original / n_labels) original samples
        # = 1600 samples at 200 Hz = 8 seconds per label
        n_labels = len(perclos)
        total_duration_sec = n_samples / sfreq  # from original sample count
        label_interval_sec = total_duration_sec / n_labels
        print(f"    Label interval: {label_interval_sec:.2f}s ({n_labels} labels over {total_duration_sec:.0f}s)")

        # ── Extract one 5-second segment per PERCLOS label ──
        # Each label covers label_interval_sec (~8s). We take a 5s window
        # from the start of each label period → one segment per label,
        # matching STEEGFormer's ~750 trials/participant.
        label_interval_samples = int(round(label_interval_sec * effective_sfreq))
        segments = []
        seg_labels = []

        for i in range(n_labels):
            start = i * label_interval_samples
            end = start + win_samples  # 5s window from start of each 8s period

            if end > T_total:
                break

            seg = data[:, start:end].astype(np.float32)
            segments.append(seg)
            seg_labels.append(float(perclos[i]))

        print(f"    Extracted {len(segments)} segments of {SEGMENT_SEC}s "
              f"(1 per PERCLOS label, {n_labels} labels)")

        return segments, seg_labels

    # ─────────────────────────────────────────────────
    # HDF5 writer (stores y as float for regression)
    # ─────────────────────────────────────────────────
    @staticmethod
    def _write_h5(rows, h5_path, split):
        if len(rows) == 0:
            print(f"  [{split}] nothing to write")
            return

        C, T = rows[0][0].shape
        n = len(rows)

        with h5py.File(h5_path, "w") as f:
            x_ds = f.create_dataset("x", shape=(n, C, T), dtype="f4")
            # y is float32 for regression (PERCLOS ∈ [0, 1])
            y_ds = f.create_dataset("y", shape=(n,), dtype="f4")
            part_ds = f.create_dataset("participant", shape=(n,), dtype="i8")

            for i, (eeg, label, part_id) in enumerate(rows):
                x_ds[i] = eeg
                y_ds[i] = label
                part_ds[i] = part_id

            f.attrs["split"] = split
            f.attrs["n_samples"] = n
            f.attrs["sampling_rate"] = TARGET_SFREQ
            f.attrs["n_channels"] = C
            f.attrs["task"] = "regression"

        print(f"  [{split}] wrote {n} rows → {h5_path}")


if __name__ == "__main__":
    exporter = ImportSeedVig()
    exporter.import_data_to_hdf5(
        raw_dir="/Users/sadeghemami/Downloads/Raw_Data",
        label_dir="/Users/sadeghemami/Downloads/perclos_labels",
        output_dir="downstream/data/seed_vig",
    )

    # Quick verification
    for split in ["train", "val"]:
        path = f"downstream/data/seed_vig/{split}.h5"
        with h5py.File(path, "r") as f:
            print(f"\n=== {split}.h5 ===")
            print(f"  x:           {f['x'].shape}")
            print(f"  y:           {f['y'].shape}   range: [{f['y'][:].min():.3f}, {f['y'][:].max():.3f}]")
            print(f"  participant: unique count: {len(np.unique(f['participant'][:]))}")
