import re
import os
from pathlib import Path
import numpy as np
import mne
from export_data.export_data_h5 import ImportDataDownstream
import h5py


class ImportPhysioP300(ImportDataDownstream):
    """
    PhysioNet ERP-based BCI (P300 speller) dataset.
    8 subjects: [2, 3, 4, 5, 6, 7, 9, 11]
    Binary classification: target (1) vs non-target (0).
    64 EEG channels, native 2048 Hz → resampled to 256 Hz at export.
    Epoch window: [-0.1, 2.0] s relative to stimulus onset.
    Source: erp-based-brain-computer-interface-recordings-1.0.0
    """

    NON_EEG = ["EARL", "EARR", "VEOGL", "VEOGR", "HEOGL", "HEOGR"]

    def get_config(self):
        self.config = "downstream/info_dataset/physio_P300.yaml"

    def condition(self, file: Path):
        if file.name.startswith("._"):
            return False
        return file.suffix.lower() == ".edf"

    def get_participant_number(self, file: Path):
        m = re.match(r"s(\d+)", file.parent.name)
        if m is None:
            raise ValueError(f"Cannot extract subject from path: {file}")
        return int(m.group(1))

    def import_data(
        self,
        input_dir,
        output_dir,
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression=None,
    ):
        """
        Override to handle nested subject-folder structure (sXX/rcXX.edf).
        All participants appear in BOTH train and val:
        80% of each participant's trials → train, 20% → val.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5_path = output_dir / "train.h5"
        val_h5_path = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # 1. Auto-discover subject folders (sXX) and collect EDF files
        subject_to_files = {}
        for sub_dir in sorted(input_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            m = re.match(r"s(\d+)", sub_dir.name)
            if m is None:
                continue
            sub = int(m.group(1))
            for file in sorted(sub_dir.iterdir()):
                if not self.condition(file):
                    continue
                subject_to_files.setdefault(sub, []).append(file)

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid EDF files found in {input_dir}")

        print(f"Found {len(subject_to_files)} subjects: {sorted(subject_to_files.keys())}")

        # 2. Extract all trials per subject, then split 80/20 per subject
        train_trials = []  # list of (participant_id, eeg, label)
        val_trials = []

        for pid in sorted(subject_to_files.keys()):
            # Collect all trials for this subject across all run files
            subject_trials = []
            for file_path in subject_to_files[pid]:
                print(f"Processing subject={pid}, file={file_path.name}")
                trial_label_pairs = self._extract_trials(file_path)
                subject_trials.extend(trial_label_pairs)

            if len(subject_trials) == 0:
                print(f"  Warning: no trials for subject {pid}, skipping.")
                continue

            # Shuffle and split this subject's trials
            n_trials = len(subject_trials)
            n_val = max(1, int(round(n_trials * val_ratio)))
            perm = rng.permutation(n_trials)

            for i, idx in enumerate(perm):
                eeg, label = subject_trials[idx]
                if i < n_val:
                    val_trials.append((pid, eeg, label))
                else:
                    train_trials.append((pid, eeg, label))

            print(f"  Subject {pid}: {n_trials} total → "
                  f"{n_trials - n_val} train, {n_val} val")

        # 3. Write HDF5
        self._write_trials_hdf5(train_trials, train_h5_path, "train", use_float16, compression)
        self._write_trials_hdf5(val_trials, val_h5_path, "val", use_float16, compression)

        print(f"\nSaved train HDF5: {train_h5_path} ({len(train_trials)} trials)")
        print(f"Saved val HDF5:   {val_h5_path} ({len(val_trials)} trials)")

    def _write_trials_hdf5(self, trials, h5_path, split_name, use_float16=False, compression=None):
        """Write pre-extracted trials to HDF5. trials = list of (pid, eeg, label)."""
        dtype_x = np.float16 if use_float16 else np.float32

        with h5py.File(h5_path, "w") as f:
            x_ds = None
            y_ds = f.create_dataset("y", shape=(0,), maxshape=(None,), dtype="i8")
            participant_ds = f.create_dataset("participant", shape=(0,), maxshape=(None,), dtype="i8")

            for count, (pid, eeg, label) in enumerate(trials):
                eeg = eeg.astype(dtype_x, copy=False)

                if x_ds is None:
                    x_shape = eeg.shape  # (C, T)
                    x_dtype = "f2" if use_float16 else "f4"
                    create_kwargs = dict(
                        shape=(0, *x_shape), maxshape=(None, *x_shape),
                        dtype=x_dtype, chunks=(1, *x_shape),
                    )
                    if compression is not None:
                        create_kwargs["compression"] = compression
                    x_ds = f.create_dataset("x", **create_kwargs)

                x_ds.resize(count + 1, axis=0)
                y_ds.resize(count + 1, axis=0)
                participant_ds.resize(count + 1, axis=0)

                x_ds[count] = eeg
                y_ds[count] = int(label)
                participant_ds[count] = int(pid)

            n = len(trials)
            f.attrs["split"] = split_name
            f.attrs["n_samples"] = n
            if x_ds is not None:
                f.attrs["n_channels"] = x_ds.shape[1]
                f.attrs["time_samples"] = x_ds.shape[2]

    def apply_preprocessing(self, array):
        """Not used — preprocessing is done inside _extract_trials on the Raw object."""
        raise NotImplementedError("P300 preprocessing is handled in _extract_trials")

    def _extract_trials(self, file_path):
        """
        Extract P300 epochs from a single EDF run file.

        Returns:
            list of (trial_array, label)
            - trial_array: shape (C, T) in Volts
            - label: 1 = target, 0 = non-target
        """
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")

        # Drop non-EEG channels
        drops = [ch for ch in self.NON_EEG if ch in raw.ch_names]
        if drops:
            raw.drop_channels(drops)

        # Keep only the 64 channels from config (in config order)
        ch_list = self.config["channel_list"]
        raw.pick_channels(ch_list, ordered=True)

        # Preprocess continuous data before epoching
        raw.filter(l_freq=0.1, h_freq=120.0, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=50, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=60, method="iir", verbose="ERROR")
        raw.resample(sfreq=256.0, verbose="ERROR")

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

        # Build reverse map: event_code → annotation string
        # Find the target character from #TgtX annotation
        event_map = {}
        tgt_char = None
        for annot_str, code in event_id.items():
            event_map[code] = annot_str
            if annot_str.startswith("#Tgt"):
                tgt_char = annot_str[4]  # e.g. "#TgtA" → "A"

        if tgt_char is None:
            print(f"  Warning: no #Tgt annotation in {file_path.name}, skipping.")
            return []

        # Create epochs for all annotated events
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=-0.1, tmax=2.0,
            event_repeated="drop",
            preload=True,
            proj=False,
            baseline=None,
            verbose="ERROR",
        )

        data = epochs.get_data()  # (n_epochs, C, T) in Volts
        stim_codes = [ev[2] for ev in epochs.events]

        trial_label_pairs = []
        for eeg, code in zip(data, stim_codes):
            annot = event_map.get(code, "")
            # Skip meta-annotations (#Tgt, #end, #start, etc.)
            if annot.startswith("#"):
                continue
            label = 1 if tgt_char in annot else 0
            trial_label_pairs.append((eeg, label))

        print(f"  {file_path.name}: {len(trial_label_pairs)} trials "
              f"(target={sum(1 for _, l in trial_label_pairs if l == 1)}, "
              f"non-target={sum(1 for _, l in trial_label_pairs if l == 0)}), "
              f"tgt_char='{tgt_char}'")

        return trial_label_pairs


if __name__ == "__main__":
    exporter = ImportPhysioP300()
    exporter.import_data(
        input_dir="/Users/sadeghemami/Downloads/files-2",
        output_dir="downstream/data/physio_P300",
    )

    # Quick sanity check
    for split in ["train", "val"]:
        h5_path = f"downstream/data/physio_P300/{split}.h5"
        with h5py.File(h5_path, "r") as f:
            print(f"\n{split}: x={f['x'].shape}, y={f['y'].shape}, "
                  f"participants={np.unique(f['participant'][:])}")
            labels = f["y"][:]
            print(f"  label distribution: target={np.sum(labels == 1)}, "
                  f"non-target={np.sum(labels == 0)}")
