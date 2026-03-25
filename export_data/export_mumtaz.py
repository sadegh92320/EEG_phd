"""
Mumtaz depression dataset exporter  →  train.h5 / val.h5

Source: EDF files from Figshare #4244171
        Naming:  "{H|MDD} S{number} {EC|EO|TASK}.edf"
        19 channels (10-20 montage), 256 Hz, ~5 min continuous per condition.

Conditions used: EC (eyes-closed) and EO (eyes-open).  TASK (P300) is skipped.
Labels: 0 = Healthy, 1 = MDD

Split:  80 % of each participant's windows → train
        20 % of each participant's windows → val
        Every participant appears in BOTH splits (cross-subject eval done downstream).

Preprocessing:
    - band-pass 0.1–128 Hz (IIR via MNE)
    - notch 50 Hz + 60 Hz
    - NO resampling (already at 256 Hz baseline; per-model resampling at dataset fetch)

Segmentation: 10 s non-overlapping windows → (19, 2560) per row.
"""

import re
import numpy as np
import mne
import h5py
from pathlib import Path
from export_data.export_data import DataImport


# EEG channels to keep (standard 10-20, 19 channels)
KEEP_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Mapping from EDF channel name → standard name
# Mumtaz EDF uses names like "EEG Fp1-A1", "EEG F3-A1", etc.
def _normalize_ch_name(edf_name: str) -> str:
    """
    'EEG Fp1-A1' → 'Fp1'
    'EEG T3-A1'  → 'T3'
    Also handles plain names like 'Fp1'.
    """
    name = edf_name.strip()
    # Remove "EEG " prefix
    if name.upper().startswith("EEG "):
        name = name[4:]
    # Remove reference suffix like "-A1", "-A2", "-LE", "-REF"
    name = re.split(r"[-]", name)[0].strip()
    return name


# File name parsing
_FILE_RE = re.compile(
    r"^(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$", re.IGNORECASE
)


def _parse_filename(filename: str):
    """
    Returns (group, subject_number, condition) or None if not a valid file.
    group: 'H' or 'MDD'
    condition: 'EC', 'EO', or 'TASK'
    """
    m = _FILE_RE.match(filename)
    if m is None:
        return None
    group = m.group(1).upper()
    subject_nb = int(m.group(2))
    condition = m.group(3).upper()
    return group, subject_nb, condition


class ImportMumtaz(DataImport):

    def import_data(self):
        return None

    def get_config(self):
        self.config = "downstream/info_dataset/mumtaz.yaml"

    # ─────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────
    def import_data_to_hdf5(
        self,
        input_dir,
        output_dir="downstream/data/mumtaz",
        val_ratio=0.2,
        random_seed=92,
        window_sec=5,
    ):
        """
        Read EC and EO EDF files, preprocess, window into 10 s segments,
        split 80/20 PER PARTICIPANT, write train.h5 and val.h5.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5 = output_dir / "train.h5"
        val_h5 = output_dir / "val.h5"

        # ── Discover EDF files ──
        edf_files = sorted(input_dir.glob("*.edf"))
        if len(edf_files) == 0:
            raise FileNotFoundError(f"No .edf files in {input_dir}")

        # ── Group by participant, skip TASK ──
        # participant key = (group, subject_nb)  →  unique participant
        subject_files = {}  # {(group, nb): [(path, condition), ...]}

        for edf_path in edf_files:
            parsed = _parse_filename(edf_path.name)
            if parsed is None:
                print(f"  Skipping unrecognized file: {edf_path.name}")
                continue
            group, subject_nb, condition = parsed
            if condition == "TASK":
                continue  # skip P300
            key = (group, subject_nb)
            subject_files.setdefault(key, []).append((edf_path, condition))

        print(f"Found {len(subject_files)} participants (EC/EO only)")

        rng = np.random.default_rng(random_seed)

        # ── Process each participant ──
        sfreq = 256.0  # Mumtaz native rate
        num_channels = len(KEEP_CHANNELS)
        win_samples = int(round(sfreq * window_sec))

        train_rows = []  # list of (eeg, label, participant_id, condition_str)
        val_rows = []

        # Assign a unique integer ID to each participant
        sorted_keys = sorted(subject_files.keys(), key=lambda k: (k[0], k[1]))
        for part_idx, key in enumerate(sorted_keys):
            group, subject_nb = key
            label = 0 if group == "H" else 1  # 0=healthy, 1=MDD
            files = subject_files[key]

            all_windows = []

            for edf_path, condition in files:
                windows = self._load_and_segment(
                    edf_path, sfreq, win_samples,
                )
                for win in windows:
                    all_windows.append((win, label, part_idx, condition))

            if len(all_windows) == 0:
                print(f"  {group} S{subject_nb}: no windows extracted, skipping")
                continue

            # 80/20 split of this participant's windows
            n_win = len(all_windows)
            n_val = max(1, int(round(n_win * val_ratio)))
            perm = rng.permutation(n_win)
            val_idx = set(perm[:n_val].tolist())

            for i, row in enumerate(all_windows):
                if i in val_idx:
                    val_rows.append(row)
                else:
                    train_rows.append(row)

            print(
                f"  {group} S{subject_nb}: "
                f"{len(files)} files, {n_win} windows "
                f"({n_win - n_val} train, {n_val} val)"
            )

        # ── Write HDF5 ──
        self._write_h5(train_rows, train_h5, "train", sfreq, num_channels)
        self._write_h5(val_rows, val_h5, "val", sfreq, num_channels)

        print(f"\nDone.  train: {len(train_rows)} rows  |  val: {len(val_rows)} rows")
        print(f"  {train_h5}")
        print(f"  {val_h5}")

    # ─────────────────────────────────────────────────
    # Load one EDF, preprocess, segment
    # ─────────────────────────────────────────────────
    def _load_and_segment(self, edf_path, target_sfreq, win_samples):
        """
        Returns list of np arrays, each shape (19, win_samples).
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

        # ── Map EDF channel names to standard names ──
        ch_name_map = {}
        for edf_name in raw.ch_names:
            std_name = _normalize_ch_name(edf_name)
            if std_name in KEEP_CHANNELS:
                ch_name_map[edf_name] = std_name

        # Pick only the channels we want
        picks = list(ch_name_map.keys())
        if len(picks) < len(KEEP_CHANNELS):
            found = {_normalize_ch_name(p) for p in picks}
            missing = set(KEEP_CHANNELS) - found
            print(
                f"    WARNING: {edf_path.name} missing channels {missing}, "
                f"found {len(picks)}/{len(KEEP_CHANNELS)}"
            )
            if len(picks) == 0:
                return []

        raw.pick(picks)
        raw.rename_channels(ch_name_map)

        # Reorder to KEEP_CHANNELS order
        available = [ch for ch in KEEP_CHANNELS if ch in raw.ch_names]
        raw.reorder_channels(available)

        # ── Preprocessing: bandpass + notch ──
        actual_sfreq = raw.info["sfreq"]
        nyq = actual_sfreq / 2.0
        h_freq = min(128.0, nyq - 1.0)

        raw.filter(l_freq=0.1, h_freq=h_freq, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=50, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=60, method="iir", verbose="ERROR")

        # ── Resample if needed (should already be 256 Hz) ──
        if abs(actual_sfreq - target_sfreq) > 0.5:
            raw.resample(target_sfreq, verbose="ERROR")

        data = raw.get_data()  # (C, T)

        # ── Segment into non-overlapping windows ──
        C, T = data.shape
        n_windows = T // win_samples
        windows = []
        for w in range(n_windows):
            start = w * win_samples
            end = start + win_samples
            windows.append(data[:, start:end].astype(np.float32))

        return windows

    # ─────────────────────────────────────────────────
    # HDF5 writer
    # ─────────────────────────────────────────────────
    @staticmethod
    def _write_h5(rows, h5_path, split, sfreq, num_channels):
        if len(rows) == 0:
            print(f"  [{split}] nothing to write")
            return

        C, T = rows[0][0].shape
        n = len(rows)

        with h5py.File(h5_path, "w") as f:
            x_ds = f.create_dataset("x", shape=(n, C, T), dtype="f4")
            y_ds = f.create_dataset("y", shape=(n,), dtype="i8")
            part_ds = f.create_dataset("participant", shape=(n,), dtype="i8")

            for i, (eeg, label, part_id, _condition) in enumerate(rows):
                x_ds[i] = eeg
                y_ds[i] = label
                part_ds[i] = part_id

            f.attrs["split"] = split
            f.attrs["n_samples"] = n
            f.attrs["sampling_rate"] = sfreq
            f.attrs["n_channels"] = num_channels
            f.attrs["n_classes"] = 2

        print(f"  [{split}] wrote {n} rows to {h5_path}")


if __name__ == "__main__":
    data_import = ImportMumtaz()
    data_import.import_data_to_hdf5(
        input_dir="/Volumes/Elements/EEG_data/downstream/mumtaz/4244171",
    )
    # Quick verification
    for split in ["train", "val"]:
        path = f"downstream/data/mumtaz/{split}.h5"
        with h5py.File(path, "r") as f:
            print(f"\n=== {split}.h5 ===")
            print(f"  x:           {f['x'].shape}")
            print(f"  y:           {f['y'].shape}   unique: {np.unique(f['y'][:])}")
            print(f"  participant: {f['participant'].shape}   unique count: {len(np.unique(f['participant'][:]))}")
