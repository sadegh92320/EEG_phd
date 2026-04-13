"""
Sleep-EDF Expanded (SleepEDFx) sleep staging exporter  →  train.h5 / val.h5

Source: Sleep Cassette (SC) subset — 78 subjects, up to 2 nights each.
        PSG recordings (.edf) + Hypnogram annotations (.edf).
        2 EEG channels: Fpz-Cz, Pz-Oz.  100 Hz native (kept as-is).
        30-second epochs, 5-class: W(0), N1(1), N2(2), N3(3), REM(4).
        N4 merged into N3.  Movement time and '?' epochs discarded.

Standard convention (following Supratak et al., 2017 DeepSleepNet):
    - Use only Sleep Cassette (SC) files
    - Trim recording: keep from 30 min before first sleep epoch
      to 30 min after last sleep epoch (reduces Wake dominance)
    - Use BOTH nights per subject (more data per subject)

Split:  80 % of each participant's epochs → train (random)
        20 % of each participant's epochs → val  (random)
        Every participant appears in BOTH splits (cross-subject eval done downstream).

Preprocessing:
    - No resampling (already 100 Hz; per-model resampling at dataset fetch)
    - No filtering (already filtered in PSG recording)
"""

import re
import numpy as np
import mne
import h5py
from pathlib import Path
from export_data.export_data import DataImport


# 2 EEG channels in SleepEDFx
KEEP_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
# Standardised names for YAML / channel_info
CHANNEL_NAMES = ["Fpz-Cz", "Pz-Oz"]

NATIVE_SFREQ = 100.0   # Keep native 100 Hz
EPOCH_SEC = 30          # Standard 30-second sleep staging epochs

# Sleep stage label mapping (AASM)
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,   # N1
    "Sleep stage 2": 2,   # N2
    "Sleep stage 3": 3,   # N3
    "Sleep stage 4": 3,   # N4 → N3
    "Sleep stage R": 4,   # REM
}
DISCARD_STAGES = {"Sleep stage ?", "Movement time"}


class ImportSleepEdfx(DataImport):

    def import_data(self):
        return None

    def get_config(self):
        self.config = "downstream/info_dataset/sleep_edfx.yaml"

    # ─────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────
    def import_data_to_hdf5(
        self,
        input_dir,
        output_dir="downstream/data/sleep_edfx",
        val_ratio=0.2,
        random_seed=92,
        trim_wake_min=30,
    ):
        """
        Read SC PSG + Hypnogram .edf files, segment into 30s epochs,
        trim wake, split 80/20 PER PARTICIPANT, write train.h5 and val.h5.

        Parameters
        ----------
        input_dir : str or Path
            Path to the sleep-cassette/ folder containing *-PSG.edf and *-Hypnogram.edf files.
        output_dir : str or Path
            Output directory for train.h5 / val.h5.
        val_ratio : float
            Fraction of each participant's epochs for validation.
        random_seed : int
            Random seed for reproducibility.
        trim_wake_min : int
            Minutes of wake to keep before first sleep and after last sleep.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5 = output_dir / "train.h5"
        val_h5 = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # ── Discover PSG files ──
        psg_files = sorted(input_dir.glob("SC*-PSG.edf"))
        if len(psg_files) == 0:
            raise FileNotFoundError(f"No SC*-PSG.edf files found in {input_dir}")

        # Group by subject ID: SC4{XX}{night}E0-PSG.edf → subject XX
        subject_files = {}  # subject_id → list of (psg_path, hyp_path)
        for psg_path in psg_files:
            m = re.match(r"SC4(\d{2})\d", psg_path.stem)
            if not m:
                print(f"  Skipping unrecognized file: {psg_path.name}")
                continue

            sid = int(m.group(1))

            # Find matching hypnogram: SC4XXYZO-PSG.edf → SC4XXY*-Hypnogram.edf
            # PSG stem is like "SC4631E0", we match on first 6 chars "SC4631"
            prefix = psg_path.stem[:6]  # e.g. "SC4631"
            hyp_candidates = list(input_dir.glob(f"{prefix}*-Hypnogram.edf"))

            if len(hyp_candidates) == 0:
                print(f"  No hypnogram found for {psg_path.name}, skipping")
                continue

            hyp_path = hyp_candidates[0]

            if sid not in subject_files:
                subject_files[sid] = []
            subject_files[sid].append((psg_path, hyp_path))

        subject_ids = sorted(subject_files.keys())
        total_recordings = sum(len(v) for v in subject_files.values())
        print(f"Found {total_recordings} recordings from {len(subject_ids)} subjects")

        win_samples = int(EPOCH_SEC * NATIVE_SFREQ)  # 3000

        train_rows = []
        val_rows = []

        for part_idx, sid in enumerate(subject_ids):
            recordings = subject_files[sid]
            print(f"\n── Subject {sid} (idx={part_idx}): {len(recordings)} night(s) ──")

            all_segments = []
            all_labels = []

            for psg_path, hyp_path in recordings:
                print(f"  Loading {psg_path.name} + {hyp_path.name}")

                try:
                    segs, labs = self._load_recording(
                        psg_path, hyp_path, win_samples, trim_wake_min,
                    )
                except Exception as e:
                    print(f"  ERROR: {e}, skipping")
                    continue

                all_segments.extend(segs)
                all_labels.extend(labs)

            if len(all_segments) == 0:
                print(f"  No epochs extracted, skipping subject {sid}")
                continue

            # 80/20 random split of this participant's epochs
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

            # Class distribution for this subject
            labs_arr = np.array(all_labels)
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
        all_train_y = [r[1] for r in train_rows]
        all_val_y = [r[1] for r in val_rows]
        print(f"\nDone.  train: {len(train_rows)} rows  |  val: {len(val_rows)} rows")
        class_names = ["W", "N1", "N2", "N3", "REM"]
        for split_name, labels in [("train", all_train_y), ("val", all_val_y)]:
            arr = np.array(labels)
            dist = ", ".join(f"{class_names[c]}:{np.sum(arr == c)}" for c in range(5))
            print(f"  {split_name}: {dist}")
        print(f"  {train_h5}")
        print(f"  {val_h5}")

    # ─────────────────────────────────────────────────
    # Load one recording: PSG + Hypnogram
    # ─────────────────────────────────────────────────
    def _load_recording(self, psg_path, hyp_path, win_samples, trim_wake_min):
        """
        Returns:
            segments: list of np.array (2, 3000) float32
            labels:   list of int (0-4)
        """
        # ── Load PSG ──
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="ERROR")
        raw.pick(KEEP_CHANNELS)
        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # (2, T)

        # ── Load Hypnogram annotations ──
        annot = mne.read_annotations(str(hyp_path))

        # ── Convert annotations to per-epoch labels ──
        all_epochs = []  # (onset_sec, label)
        for i in range(len(annot)):
            desc = annot.description[i]
            if desc in DISCARD_STAGES:
                continue
            if desc not in LABEL_MAP:
                continue

            onset = annot.onset[i]
            dur = annot.duration[i]
            label = LABEL_MAP[desc]
            n_30s = int(dur // EPOCH_SEC)

            for j in range(n_30s):
                epoch_onset = onset + j * EPOCH_SEC
                all_epochs.append((epoch_onset, label))

        if len(all_epochs) == 0:
            raise ValueError("No valid sleep epochs found")

        # ── Trim: keep from trim_wake_min before first sleep to trim_wake_min after last ──
        sleep_epochs = [(t, l) for t, l in all_epochs if l != 0]
        if sleep_epochs and trim_wake_min > 0:
            first_sleep = sleep_epochs[0][0]
            last_sleep = sleep_epochs[-1][0]
            trim_start = max(0, first_sleep - trim_wake_min * 60)
            trim_end = last_sleep + trim_wake_min * 60
            all_epochs = [(t, l) for t, l in all_epochs if trim_start <= t <= trim_end]

        # ── Extract 30s segments from raw data ──
        T_total = data.shape[1]
        max_onset = (T_total - win_samples) / sfreq

        segments = []
        labels = []
        for onset_sec, label in all_epochs:
            start_samp = int(round(onset_sec * sfreq))
            end_samp = start_samp + win_samples

            if end_samp > T_total:
                continue

            seg = data[:, start_samp:end_samp].astype(np.float32)
            if seg.shape[1] != win_samples:
                continue

            segments.append(seg)
            labels.append(label)

        print(f"    {len(segments)} valid epochs (trimmed from {len(all_epochs)} annotated)")

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
    exporter = ImportSleepEdfx()
    exporter.import_data_to_hdf5(
        input_dir="/Users/sadeghemami/Downloads/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        output_dir="downstream/data/sleep_edfx",
    )

    # Quick verification
    for split in ["train", "val"]:
        path = f"downstream/data/sleep_edfx/{split}.h5"
        with h5py.File(path, "r") as f:
            print(f"\n=== {split}.h5 ===")
            print(f"  x:           {f['x'].shape}")
            print(f"  y:           {f['y'].shape}   unique: {np.unique(f['y'][:])}")
            print(f"  participant: unique count: {len(np.unique(f['participant'][:]))}")
