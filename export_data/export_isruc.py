"""
ISRUC-Sleep Subgroup 3 sleep staging exporter  →  train.h5 / val.h5

Source: 10 healthy subjects, 1 night each.
        PSG recordings (.rec, EDF format) + per-epoch annotations (.txt).
        6 EEG channels: F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1.
        200 Hz native (kept as-is).
        30-second epochs, 5-class: W(0), N1(1), N2(2), N3(3), REM(4).

Annotation format:
    - Two annotators per subject: {subject_id}_1.txt and {subject_id}_2.txt
    - One integer label per line, one line per 30s epoch
    - Label coding: 0=W, 1=N1, 2=N2, 3=N3, 5=REM
    - We use annotator 1 by default (standard convention).

Channel naming:
    - Raw channel names are referential (F3-A2, C3-A2, etc.)
    - Renamed to standard 10-20 names (F3, C3, O1, F4, C4, O2) for
      compatibility with channel_info.yaml and foundation model montages.

Split:  80 % of each participant's epochs → train (random)
        20 % of each participant's epochs → val  (random)
        Every participant appears in BOTH splits (cross-subject eval done downstream).
"""

import re
import shutil
import tempfile
import numpy as np
import mne
import h5py
from pathlib import Path
from export_data.export_data import DataImport


# 6 EEG channels (referential names in .rec → standard 10-20 names)
RAW_EEG_CHANNELS = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
STANDARD_NAMES = ["F3", "C3", "O1", "F4", "C4", "O2"]
RENAME_MAP = dict(zip(RAW_EEG_CHANNELS, STANDARD_NAMES))

NATIVE_SFREQ = 200.0   # Keep native 200 Hz
EPOCH_SEC = 30          # Standard 30-second sleep staging epochs

# Label mapping: ISRUC uses 0=W, 1=N1, 2=N2, 3=N3, 5=REM
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # remap 5→4 for REM


class ImportIsruc(DataImport):

    def import_data(self):
        return None

    def get_config(self):
        self.config = "downstream/info_dataset/isruc.yaml"

    # ─────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────
    def import_data_to_hdf5(
        self,
        input_dir,
        output_dir="downstream/data/isruc",
        val_ratio=0.2,
        random_seed=92,
        annotator=1,
    ):
        """
        Read .rec PSG files + .txt annotation files, segment into 30s epochs,
        split 80/20 PER PARTICIPANT, write train.h5 and val.h5.

        Parameters
        ----------
        input_dir : str or Path
            Path to the ISRUC Subgroup 3 folder. Expected structure:
            input_dir/
              1/1.rec, 1/1_1.txt, 1/1_2.txt
              2/2.rec, 2/2_1.txt, 2/2_2.txt
              ...
        output_dir : str or Path
            Output directory for train.h5 / val.h5.
        val_ratio : float
            Fraction of each participant's epochs for validation.
        random_seed : int
            Random seed for reproducibility.
        annotator : int
            Which annotator to use (1 or 2). Default: 1.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5 = output_dir / "train.h5"
        val_h5 = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # ── Discover subject folders ──
        # Each subject has a folder named by subject ID (1, 2, ..., 10)
        subject_dirs = sorted(
            [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )

        if len(subject_dirs) == 0:
            # Maybe files are directly in input_dir (flat structure)
            rec_files = sorted(input_dir.glob("*.rec"))
            if rec_files:
                # Flat structure: all files in one folder
                subject_dirs = [input_dir]
                print(f"Found {len(rec_files)} .rec files in flat structure")
            else:
                raise FileNotFoundError(
                    f"No subject folders or .rec files found in {input_dir}"
                )

        print(f"Found {len(subject_dirs)} subject folder(s)")

        win_samples = int(EPOCH_SEC * NATIVE_SFREQ)  # 6000

        train_rows = []
        val_rows = []

        for part_idx, subj_dir in enumerate(subject_dirs):
            sid = subj_dir.name if subj_dir != input_dir else "all"

            # Find .rec file
            if subj_dir == input_dir:
                rec_files = sorted(subj_dir.glob("*.rec"))
            else:
                rec_files = sorted(subj_dir.glob("*.rec"))

            if len(rec_files) == 0:
                print(f"  No .rec file found in {subj_dir}, skipping")
                continue

            for rec_path in rec_files:
                rec_stem = rec_path.stem  # e.g. "1"

                # Find annotation file
                annot_path = rec_path.parent / f"{rec_stem}_{annotator}.txt"
                if not annot_path.exists():
                    print(f"  No annotation {annot_path.name} found, skipping")
                    continue

                print(f"\n── Subject {sid} (idx={part_idx}): {rec_path.name} ──")

                try:
                    segs, labs = self._load_recording(rec_path, annot_path, win_samples)
                except Exception as e:
                    print(f"  ERROR: {e}, skipping")
                    continue

                if len(segs) == 0:
                    print(f"  No epochs extracted, skipping")
                    continue

                # 80/20 random split
                n_seg = len(segs)
                n_val = max(1, int(round(n_seg * val_ratio)))
                perm = rng.permutation(n_seg)
                val_idx = set(perm[:n_val].tolist())

                for i in range(n_seg):
                    row = (segs[i], labs[i], part_idx)
                    if i in val_idx:
                        val_rows.append(row)
                    else:
                        train_rows.append(row)

                # Class distribution
                labs_arr = np.array(labs)
                class_names = ["W", "N1", "N2", "N3", "REM"]
                dist = ", ".join(
                    f"{class_names[c]}:{np.sum(labs_arr == c)}"
                    for c in range(5) if np.sum(labs_arr == c) > 0
                )
                print(f"  {n_seg} epochs ({n_seg - n_val} train, {n_val} val) [{dist}]")

        # ── Write HDF5 ──
        self._write_h5(train_rows, train_h5, "train")
        self._write_h5(val_rows, val_h5, "val")

        # Summary
        print(f"\nDone.  train: {len(train_rows)} rows  |  val: {len(val_rows)} rows")
        class_names = ["W", "N1", "N2", "N3", "REM"]
        for split_name, rows in [("train", train_rows), ("val", val_rows)]:
            arr = np.array([r[1] for r in rows])
            dist = ", ".join(f"{class_names[c]}:{np.sum(arr == c)}" for c in range(5))
            print(f"  {split_name}: {dist}")
        print(f"  {train_h5}")
        print(f"  {val_h5}")

    # ─────────────────────────────────────────────────
    # Load one recording: .rec + .txt annotation
    # ─────────────────────────────────────────────────
    def _load_recording(self, rec_path, annot_path, win_samples):
        """
        Returns:
            segments: list of np.array (6, 6000) float32
            labels:   list of int (0-4)
        """
        # ── Load annotations ──
        with open(annot_path, "r") as f:
            raw_labels = [int(line.strip()) for line in f if line.strip()]

        print(f"  Annotations: {len(raw_labels)} epochs from {annot_path.name}")

        # ── Load PSG (.rec → rename to .edf for MNE) ──
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copy2(str(rec_path), tmp_path)

        try:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        sfreq = raw.info["sfreq"]

        # ── Pick EEG channels ──
        available = [ch for ch in RAW_EEG_CHANNELS if ch in raw.ch_names]
        if len(available) == 0:
            raise ValueError(
                f"No EEG channels found. Available: {raw.ch_names}"
            )

        raw.pick(available)

        # Rename to standard 10-20 names
        rename = {ch: RENAME_MAP[ch] for ch in available if ch in RENAME_MAP}
        raw.rename_channels(rename)

        # Reorder to standard order
        ordered = [RENAME_MAP[ch] for ch in RAW_EEG_CHANNELS if ch in available]
        raw.reorder_channels(ordered)

        # ── Preprocessing: notch 50/60 Hz + bandpass 0.1–64 Hz (IIR) ──
        raw.notch_filter(freqs=50, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=60, method="iir", verbose="ERROR")
        raw.filter(l_freq=0.1, h_freq=64.0, method="iir", verbose="ERROR")

        data = raw.get_data()  # (C, T)
        C, T_total = data.shape

        print(f"  EEG: {C} channels {ordered}, {T_total} samples, {sfreq} Hz "
              f"({T_total / sfreq:.0f}s)")

        # ── Extract 30s epochs ──
        n_possible = T_total // win_samples
        n_epochs = min(len(raw_labels), n_possible)

        segments = []
        labels = []

        for i in range(n_epochs):
            raw_label = raw_labels[i]
            if raw_label not in LABEL_MAP:
                continue  # skip unknown labels

            start = i * win_samples
            end = start + win_samples

            seg = data[:, start:end].astype(np.float32)
            if seg.shape[1] != win_samples:
                continue

            segments.append(seg)
            labels.append(LABEL_MAP[raw_label])

        print(f"  Extracted {len(segments)} valid epochs (from {n_epochs} annotated)")

        return segments, labels

    # ─────────────────────────────────────────────────
    # HDF5 writer
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
            y_ds = f.create_dataset("y", shape=(n,), dtype="i8")
            part_ds = f.create_dataset("participant", shape=(n,), dtype="i8")

            for i, (eeg, label, part_id) in enumerate(rows):
                x_ds[i] = eeg
                y_ds[i] = label
                part_ds[i] = part_id

            f.attrs["split"] = split
            f.attrs["n_samples"] = n
            f.attrs["sampling_rate"] = NATIVE_SFREQ
            f.attrs["n_channels"] = C
            f.attrs["task"] = "classification"
            f.attrs["classes"] = "W,N1,N2,N3,REM"

        print(f"  [{split}] wrote {n} rows → {h5_path}")


if __name__ == "__main__":
    exporter = ImportIsruc()
    exporter.import_data_to_hdf5(
        input_dir="/Users/sadeghemami/Downloads/ISRUC",
        output_dir="downstream/data/isruc",
    )

    # Quick verification
    for split in ["train", "val"]:
        path = f"downstream/data/isruc/{split}.h5"
        with h5py.File(path, "r") as f:
            print(f"\n=== {split}.h5 ===")
            print(f"  x:           {f['x'].shape}")
            print(f"  y:           {f['y'].shape}   unique: {np.unique(f['y'][:])}")
            print(f"  participant: unique count: {len(np.unique(f['participant'][:]))}")
