import re
from pathlib import Path
import numpy as np
import mne
from scipy.io import loadmat
from export_data.export_data_h5 import ImportDataDownstream
import h5py
import sys



class ImportBCIComp2a(ImportDataDownstream):
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/bci_comp_2a.yaml"

    def import_data(
        self,
        input_dir,
        output_dir,
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression=None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5_path = output_dir / "train.h5"
        val_h5_path = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # 1. group files by participant
        subject_to_files = {}

        for file in sorted(input_dir.iterdir()):
            if not self.condition(file):
                continue

            participant_nb = self.get_participant_number(file)
            subject_to_files.setdefault(participant_nb, []).append(file)

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid files found in {input_dir}")

        # 2. split participants into train / val
        participant_ids = sorted(subject_to_files.keys())
        n_participants = len(participant_ids)
        n_val = max(1, int(round(n_participants * val_ratio)))

        perm = rng.permutation(n_participants)
        val_idx = set(perm[:n_val])

        train_files = []
        val_files = []

        for i, participant_nb in enumerate(participant_ids):
            files = subject_to_files[participant_nb]
            for file in files:
                if file.name.split(".")[0][-1] == "E":
                    val_files.append((participant_nb, file))
                elif file.name.split(".")[0][-1] == "T":
                    train_files.append((participant_nb, file))
                else:
                    raise Exception("pas le bon nom")

        # 3. write HDF5
        self._write_split_hdf5(
            file_tuples=train_files,
            h5_path=train_h5_path,
            split_name="train",
            use_float16=use_float16,
            compression=compression,
        )

        self._write_split_hdf5(
            file_tuples=val_files,
            h5_path=val_h5_path,
            split_name="val",
            use_float16=use_float16,
            compression=compression,
        )

        print(f"Saved train HDF5: {train_h5_path}")
        print(f"Saved val HDF5:   {val_h5_path}")


    def _write_split_hdf5(
        self,
        file_tuples,
        h5_path,
        split_name,
        use_float16=False,
        compression=None,
    ):
        dtype_x = np.float16 if use_float16 else np.float32

        with h5py.File(h5_path, "w") as f:
            x_ds = None
            y_ds = f.create_dataset(
                "y",
                shape=(0,),
                maxshape=(None,),
                dtype="i8",
            )
            participant_ds = f.create_dataset(
                "participant",
                shape=(0,),
                maxshape=(None,),
                dtype="i8",
            )

            count = 0
            for participant_nb, file_path in file_tuples:
                print(f"[{split_name}] processing subject={participant_nb}, file={file_path.name}")

                trial_label_pairs = self._extract_trials(file_path)

                for eeg, label in trial_label_pairs:
                    eeg = eeg.astype(dtype_x, copy=False)

                    if x_ds is None:
                        x_shape = eeg.shape  # (C, T)
                        x_dtype = "f2" if use_float16 else "f4"

                        create_kwargs = dict(
                            shape=(0, *x_shape),
                            maxshape=(None, *x_shape),
                            dtype=x_dtype,
                            chunks=(1, *x_shape),
                        )
                        if compression is not None:
                            create_kwargs["compression"] = compression

                        x_ds = f.create_dataset("x", **create_kwargs)

                    x_ds.resize(count + 1, axis=0)
                    y_ds.resize(count + 1, axis=0)
                    participant_ds.resize(count + 1, axis=0)

                    x_ds[count] = eeg
                    y_ds[count] = int(label)
                    participant_ds[count] = int(participant_nb)

                    count += 1

            f.attrs["split"] = split_name
            f.attrs["n_samples"] = count
            if x_ds is not None:
                f.attrs["n_channels"] = x_ds.shape[1]
                f.attrs["time_samples"] = x_ds.shape[2]

    def apply_preprocessing(self, array):
        """
        Preprocessing for BCI-IV-2a downstream (keep at 250 Hz baseline).
        Band-pass 0.1–124 Hz, notch 50/60 Hz, no resampling.
        Per-model resampling happens in Downstream_Dataset at training time.
        """
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        nyq = raw_mne_object.info["sfreq"] / 2.0
        h_freq = min(128.0, nyq - 1.0)  # 124 Hz at 250 Hz source
        raw_mne_object.filter(l_freq=0.1, h_freq=h_freq, method="iir")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        return raw_mne_object.get_data()

    def condition(self, file: Path):
        if file.name.startswith("._"):
            return False
        return file.suffix.lower() == ".gdf" and re.match(r"A\d{2}[TE]\.gdf$", file.name) is not None

    def get_participant_number(self, file: Path):
        # A01T.gdf -> 1
        m = re.match(r"A(\d{2})[TE]\.gdf$", file.name)
        if m is None:
            raise ValueError(f"Unexpected file name format: {file.name}")
        return int(m.group(1))

    def _find_label_file(self, gdf_path: Path):
        """
        Assumption:
        matching label file has the same stem, e.g.
            A01T.gdf -> A01T.mat
            A01E.gdf -> A01E.mat
        """
        mat_path = gdf_path.with_suffix(".mat")
        if not mat_path.exists():
            raise FileNotFoundError(f"Matching label file not found for {gdf_path.name}: {mat_path}")
        return mat_path

    def _read_labels_from_mat(self, mat_path: Path):
        """
        Assumption:
        the .mat file contains a 1D vector of trial labels in trial order.

        This function tries a few common key names.
        """
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        candidate_keys = ["classlabel", "labels", "y", "true_y", "label"]

        for key in candidate_keys:
            if key in mat:
                labels = np.asarray(mat[key]).squeeze()
                if labels.ndim == 1:
                    return labels.astype(int)

        # fallback: try to find a plausible 1D integer-like vector
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value).squeeze()
            if arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                unique_vals = np.unique(arr)
                # likely MI labels 1..4 or 0..3
                if len(unique_vals) <= 10:
                    return arr.astype(int)

        raise KeyError(f"Could not find label vector in {mat_path}")

    def _extract_trials(self, file_path):
        """
        Returns:
            list of (trial_array, label)

        Assumptions:
        1. EEG is in the GDF file.
        2. Labels are in the matching MAT file, in the same trial order.
        3. We extract one trial per cue event (769,770,771,772).
        4. We use the common MI window [2s, 6s] after cue onset.
        5. We keep the first 22 EEG channels only.
        """

        # ----- load EEG + events from GDF -----
        raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose="ERROR")
        events, _ = mne.events_from_annotations(raw, verbose="ERROR")

        data = raw.get_data()          # shape (C, T)
        sfreq = raw.info["sfreq"]

        # keep EEG channels only
        data = data[:22, :]

        # ----- filter continuous recording BEFORE epoching -----
        # Paper D.2: bandpass 0.1–128 Hz, notch 50/60 Hz on continuous data
        data = self.apply_preprocessing(data)

        # ----- get cue onsets from GDF -----
        # Official 2a cue codes: 769,770,771,772
       
        reject_code = 1

        if file_path.name.split(".")[0][-1] == "E":
            cue_codes = {7}

        if file_path.name.split(".")[0][-1] == "T":
            cue_codes = {7,8,9,10}

        cue_positions = []
        rejected_positions = set()

        for event in events:
            pos = int(event[0])
            code = int(event[2])

            if code == reject_code:
                rejected_positions.add(pos)

            if code in cue_codes:
                cue_positions.append((pos, code))

        # ----- load labels from matching MAT -----
        mat_path = self._find_label_file(file_path)
        labels = self._read_labels_from_mat(mat_path)
     

        if len(labels) < len(cue_positions):
            raise RuntimeError(
                f"Label count smaller than cue count for {file_path.name}: "
                f"{len(labels)} labels vs {len(cue_positions)} cues"
            )

        trial_label_pairs = []

        # window [0s, 4s] after cue onset — captures full MI period
        # (BCI-IV-2a: cue at t=2s, MI from t=2s to t=6s = 4s of MI starting at cue)
        start_offset = int(round(0.0 * sfreq))
        end_offset = int(round(4.0 * sfreq))

        label_idx = 0
        T = data.shape[1]
       

        for pos, cue_code in cue_positions:
            start = pos + start_offset
            end = pos + end_offset

            if end > T:
                label_idx += 1
                continue

            eeg = data[:, start:end]

            # Convert labels to 0..3 if they are 1..4
            label = int(labels[label_idx])
            if label in [1, 2, 3, 4]:
                label = label - 1

            trial_label_pairs.append((eeg, label))
            label_idx += 1

        return trial_label_pairs
    
if __name__ == "__main__":
    dataimport = ImportBCIComp2a()
    dataimport.import_data(input_dir="/Volumes/Elements/EEG_data/pretraining/BCICIV_2a_gdf", output_dir="downstream/data/bci_comp_2a")
    with h5py.File("/Users/sadeghemami/paper_1_code/downstream/data/bci_comp_2a/train.h5", "r") as f:
        print(np.unique(f["participant"][:]))

   