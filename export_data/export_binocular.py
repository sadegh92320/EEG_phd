import os
from pathlib import Path
import h5py
import numpy as np
import mne
from export_data.export_data import DataImport
import pandas as pd
import torch

class ImportBinocularBis(DataImport):

    def import_data(self):
        return None
    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/downstream/info_dataset/upperlimb.yaml"

    def get_participant_nb(self, file):
        participant_nb = file.name.split(".")[0].replace("Subject", "")
        return int(participant_nb)


    def import_data_to_hdf5(
        self,
        input_dir="/Users/sadeghemami/Downloads/834976",
        output_dir="downstream/data/upper_limb",
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression=None,
    ):
        """
        Convert all .gdf files into two HDF5 files:
            - train.h5
            - val.h5

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
            if file.suffix.lower() != ".csv" or file.name.startswith("._"):
                continue

            
            participant_nb = self.get_participant_nb(file)
            subject_to_files[participant_nb] = file

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid .csv files found in {input_dir}")

        # -------------------------------------------------
        # 2. Split files by subject into train/val runs
        # -------------------------------------------------
        train_files = []
        val_files = []

        for participant_nb, file in subject_to_files.items():
            df = pd.read_csv()
            df = df.to_numpy()
            df = df.T
            
            num_epochs = np.max(df[3])

            n_val = max(1, int(round(num_epochs * val_ratio)))
            perm = rng.permutation(num_epochs)

            val_idx = set(perm[:n_val])
            for epoch in enumerate(num_epochs):
                if epoch in val_idx:
                    val_files.append((participant_nb, epoch, file))
                else:
                    train_files.append((participant_nb, epoch, file))

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

            for participant_nb, epoch,path in file_tuples:
                print(f"[{split_name}] processing subject={participant_nb}, file={path.name}")

                trials, labels = self._extract_trials(path, epoch)

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
            f.attrs["n_channels"] = 64

    def _extract_trials(self, path, epoch):
        """
        Read one .gdf file, extract trials, preprocess them, return:
            trials: list of np.ndarray with shape (C, T)
            labels: list of int
        """
        df = pd.read_csv(path)
        df = df.to_numpy()
        df = df.T
        label = df[2]
        epochs = df[3]
        data = df[4:,:]
       
        data = data[:,np.where(epochs == epoch)[0]]
        label = label[np.where(epochs == epoch)]
        l = np.unique(label)
        l = (int(l.item()))
        data = self.apply_preprocessing_pretrain(data)
        trials.append(data)
        labels.append(l)
        trials = []
        labels = []
        return trials, labels

    def apply_preprocessing_pretrain(self, array):
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw_mne_object.resample(sfreq=128.0)
        eeg_data = raw_mne_object.get_data()
        return eeg_data


import os
from pathlib import Path
import h5py
import numpy as np
import mne
import pandas as pd

class ImportBinocular(DataImport): # Inherit from DataImport if needed


    def get_config(self):
        self.config = "downstream/info_dataset/binocular.yaml"

    def import_data(self):
        return None

    def get_participant_nb(self, file):
        participant_nb = file.name.split(".")[0].replace("Subject", "")
        return int(participant_nb)

    def import_data_to_hdf5(
        self,
        input_dir="/Users/sadeghemami/Downloads/834976",
        output_dir="downstream/data/upper_limb",
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

        train_files = []
        val_files = []

        # 1. Group and Split Logic
        for file in sorted(input_dir.iterdir()):
            if file.suffix.lower() != ".csv" or file.name.startswith("._"):
                continue

            participant_nb = self.get_participant_nb(file)
            
            # Read CSV once to find how many trials it has
            df = pd.read_csv(file)
            unique_epochs = df.iloc[:, 3].unique() # Get all unique Trial IDs
            
            n_epochs = len(unique_epochs)
            n_val = max(1, int(round(n_epochs * val_ratio)))
            
            # Shuffle the trial IDs
            perm_epochs = rng.permutation(unique_epochs)
            val_set = set(perm_epochs[:n_val])

            for epoch_id in unique_epochs:
                if epoch_id in val_set:
                    val_files.append((participant_nb, epoch_id, file))
                else:
                    train_files.append((participant_nb, epoch_id, file))

        # 2. Write Splits
        self._write_split_hdf5(train_files, train_h5_path, "train", use_float16, compression)
        self._write_split_hdf5(val_files, val_h5_path, "val", use_float16, compression)

    def _write_split_hdf5(self, file_tuples, h5_path, split_name, use_float16, compression):
        dtype_x = np.float16 if use_float16 else np.float32

        with h5py.File(h5_path, "w") as f:
            x_ds, y_ds, participant_ds, run_name_ds = None, None, None, None
            count = 0

            # Group by path to avoid re-reading the CSV 200 times for one subject
            # This is a major speed optimization
            current_path = None
            cached_df = None

            for p_nb, e_id, path in file_tuples:
                if path != current_path:
                    print(f"[{split_name}] Loading subject={p_nb}, file={path.name}")
                    cached_df = pd.read_csv(path).to_numpy().T
                    current_path = path

                # Extract specific trial from cached data
                trials, labels = self._extract_trials_from_data(cached_df, e_id)

                for eeg, label in zip(trials, labels):
                    if x_ds is None:
                        # Initialize datasets based on first trial shape
                        c, t = eeg.shape
                        x_ds = f.create_dataset("x", shape=(0, c, t), maxshape=(None, c, t), 
                                                dtype="f2" if use_float16 else "f4", chunks=(1, c, t), compression=compression)
                        y_ds = f.create_dataset("y", shape=(0,), maxshape=(None,), dtype="i8")
                        participant_ds = f.create_dataset("participant", shape=(0,), maxshape=(None,), dtype="i8")
                        run_name_ds = f.create_dataset("run_name", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

                    x_ds.resize(count + 1, axis=0)
                    y_ds.resize(count + 1, axis=0)
                    participant_ds.resize(count + 1, axis=0)
                    run_name_ds.resize(count + 1, axis=0)

                    x_ds[count] = eeg.astype(dtype_x)
                    y_ds[count] = int(label)
                    participant_ds[count] = int(p_nb)
                    run_name_ds[count] = path.name
                    count += 1

    def _extract_trials_from_data(self, data_array, epoch_id):
        """
        Processes a single trial from pre-loaded data.
        """
        labels = data_array[2]
        epochs = data_array[3]
        data   = data_array[4:, :]

        # Selection using the boolean mask logic we discussed
        mask = (epochs == epoch_id)
        trial_data = data[:, mask]
        
        # Get the label (assuming it's the same for the whole trial)
        trial_label = labels[mask][0]

        # Preprocessing
        processed_data = self.apply_preprocessing_pretrain(trial_data)

        return [processed_data], [trial_label]

    def apply_preprocessing_pretrain(self, array):
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw_mne_object.resample(sfreq=128.0)
        eeg_data = raw_mne_object.get_data()
        return eeg_data


if __name__ == "__main__":
    data_import = ImportBinocular()
    data_import.import_data_to_hdf5(input_dir="/Users/sadeghemami/Downloads/Binocular-Swap_Vision", output_dir="downstream/data/binocular")