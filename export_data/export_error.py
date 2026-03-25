import mne


import re
from pathlib import Path
import numpy as np
import mne
from scipy.io import loadmat
from export_data.export_data_h5 import ImportDataDownstream
import h5py
import sys
from export_data.export_data import DataImport
import h5py

class ImportError(DataImport):

    def import_data(self):
        return None
    def get_config(self):
        self.config = "downstream/info_dataset/error.yaml"

    def participant_nb(self,file):
        name = file.name
        no_dot = name.split(".")[0]
        sli = no_dot.split("_")[1]
        nb = sli[2:4]
        return int(nb)

    def import_data_to_hdf5(
        self,
        input_dir="/Users/sadeghemami/Downloads/834976",
        output_dir="downstream/data/upper_limb",
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression="gzip",
    ):
        """
        Convert all .gdf files into two HDF5 files:
            - upper_limb_train.h5
            - upper_limb_val.h5

        Split is done by run/file within each subject.
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5_path = output_dir / "train.h5"
        val_h5_path = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # -------------------------------------------------
        # 1. Group files by subject
        # -------------------------------------------------
        subject_to_files = {}

        for file in sorted(input_dir.iterdir()):
            if file.suffix.lower() != ".vhdr" or file.name.startswith("._"):
                continue

    
            
            participant_nb = self.participant_nb(file)

            subject_to_files.setdefault(participant_nb, []).append(file)

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid .gdf files found in {input_dir}")

        # -------------------------------------------------
        # 2. Split files by subject into train/val runs
        # -------------------------------------------------
        train_files = []
        val_files = []

        for participant_nb, files in subject_to_files.items():
            files = sorted(files)
            n_files = len(files)

            if n_files == 1:
                # If only one run exists, keep it in train
                train_files.extend((participant_nb, f) for f in files)
                continue
            if n_files != 10:
                print("ok")
            else:
                print("not 10")
                print(n_files)
                print(participant_nb)

            n_val = max(1, int(round(n_files * val_ratio)))
            perm = rng.permutation(n_files)

            val_idx = set(perm[:n_val])
            for i, f in enumerate(files):
                if i in val_idx:
                    val_files.append((participant_nb, f))
                else:
                    train_files.append((participant_nb, f))

        # -------------------------------------------------
        # 3. Create HDF5 files and write incrementally
        # -------------------------------------------------
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
        compression="gzip",
    ):
        """
        Write one split (train or val) to HDF5 incrementally.
        """

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
            run_name_ds = f.create_dataset(
                "run_name",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            count = 0

            for participant_nb, path in file_tuples:
                print(f"[{split_name}] processing subject={participant_nb}, file={path.name}")

                trials, labels = self._extract_trials(path)

                if len(trials) == 0:
                    continue

                for eeg, label in zip(trials, labels):
                    eeg = eeg.astype(dtype_x, copy=False)

                    if x_ds is None:
                        x_shape = eeg.shape  # (C, T)
                        x_dtype = "f2" if use_float16 else "f4"
                        x_ds = f.create_dataset(
                            "x",
                            shape=(0, *x_shape),
                            maxshape=(None, *x_shape),
                            dtype=x_dtype,
                            chunks=(1, *x_shape),
                        )

                    # grow datasets by one sample
                    x_ds.resize(count + 1, axis=0)
                    y_ds.resize(count + 1, axis=0)
                    participant_ds.resize(count + 1, axis=0)
                    run_name_ds.resize(count + 1, axis=0)

                    x_ds[count] = eeg
                    y_ds[count] = int(label)
                    participant_ds[count] = int(participant_nb)
                    run_name_ds[count] = path.name

                    count += 1

            f.attrs["split"] = split_name
            f.attrs["n_samples"] = count
            f.attrs["sampling_rate"] = 128.0
            f.attrs["n_channels"] = 61

    def _extract_trials(self, path):
        raw = mne.io.read_raw_brainvision(path, preload=True)

        # keep EEG channels only if these are the first 64
        raw.pick(raw.ch_names[:64])

        # preprocess continuous signal first
        raw.notch_filter(freqs=50, method="iir")
        raw.notch_filter(freqs=60, method="iir")
        raw.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw.resample(sfreq=256.0)

        events, event_id = mne.events_from_annotations(raw)

        class_events = {48: 0, 96: 1}   # 0=correct, 1=incorrect

        selected = np.array([ev for ev in events if ev[2] in class_events], dtype=int)
        if len(selected) == 0:
            return [], []

        epochs = mne.Epochs(
            raw,
            selected,
            event_id=None,
            tmin=-0.1,
            tmax=1.0,
            baseline=(-0.1, 0.0),
            preload=True,
            verbose=False,
        )

        X = epochs.get_data()
        y = np.array([class_events[e] for e in epochs.events[:, 2]], dtype=int)

        trials = [eeg.astype(np.float32) for eeg in X]
        labels = y.tolist()

        return trials, labels

    def apply_preprocessing_pretrain(self, array):
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw_mne_object.resample(sfreq=128.0)
        eeg_data = raw_mne_object.get_data()
        return eeg_data
    

if __name__ == "__main__":
    path = "/Users/sadeghemami/Downloads/EEG/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr"

    # 1. Load the data
    #data_import = ImportError()
    #data_import.import_data_to_hdf5(input_dir="/Users/sadeghemami/Downloads/Error_dataset", output_dir="downstream/data/error")
    # 'events' is a NumPy array
    # 'event_id' is a dictionary mapping the name (e.g., 'S  1') to the integer ID
    path = "downstream/data/error/train.h5"

    with h5py.File(path, "r") as f:
        label = f["x"]
        print(label.shape)
