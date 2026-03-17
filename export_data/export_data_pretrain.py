from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch
import sys
import os
from typing import Any
import numpy as np
from abc import ABC, abstractmethod
import yaml
import mne
from scipy.signal import resample
from process_data.mne_constructor import MNEMethods
from pathlib import Path



class ImportDataPre(ABC):
    def __init__(self, num_chan):
        self.get_config()
        with open(self.config) as f:
            self.config = yaml.safe_load(f)
        
        self.num_chan = num_chan
        self.mne_process = MNEMethods(self.config)

        self.data_dir = None

    @abstractmethod
    def get_config(self):
        pass


    @abstractmethod
    def get_participant_number(self, file: Path):
        pass

    @abstractmethod
    def condition_file_name(self, file_name):
        pass

    def import_data(
        self,
        input_dir,
        output_dir,
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression="gzip",
    ):
        """
        Convert raw .mat files into:
            - train.h5
            - val.h5

        Split is done by participant.
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5_path = output_dir / "train.h5"
        val_h5_path = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # 1. Group files by participant
        subject_to_files = {}

        for file in sorted(input_dir.iterdir()):
            if self.condition_file_name(file):
                continue

            participant_nb = self.get_participant_number(file)
            subject_to_files.setdefault(participant_nb, []).append(file)

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid .mat files found in {input_dir}")

        # 2. Split participants into train / val
        participant_ids = sorted(subject_to_files.keys())
        n_participants = len(participant_ids)
        n_val = max(1, int(round(n_participants * val_ratio)))

        perm = rng.permutation(n_participants)
        val_idx = set(perm[:n_val])

        train_files = []
        val_files = []

        for i, participant_nb in enumerate(participant_ids):
            files = subject_to_files[participant_nb]
            if i in val_idx:
                val_files.extend((participant_nb, f) for f in files)
            else:
                train_files.extend((participant_nb, f) for f in files)

        # 3. Write HDF5
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
        dtype_x = np.float16 if use_float16 else np.float32

        with h5py.File(h5_path, "w") as f:
            x_ds = None
            participant_ds = f.create_dataset(
                "participant",
                shape=(0,),
                maxshape=(None,),
                dtype="i8",
            )

            count = 0

            for participant_nb, file_path in file_tuples:
                print(f"[{split_name}] processing subject={participant_nb}, file={file_path.name}")

                trials = self._extract_trials(file_path)   # list of arrays (C, T)

                for data in trials:
                    split_data = self.split_with_hops(
                        data=data,
                        window_s=6,
                        hop_s=0.5,
                        sampling_rate=128,
                        channels_expected=self.num_chan,
                    )

                    if len(split_data) == 0:
                        continue

                    for eeg, _ in split_data:
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
                        participant_ds.resize(count + 1, axis=0)

                        x_ds[count] = eeg
                        participant_ds[count] = participant_nb

                        count += 1

            f.attrs["split"] = split_name
            f.attrs["n_samples"] = count
            f.attrs["sampling_rate"] = 128.0
            f.attrs["n_channels"] = self.num_chan
            f.attrs["window_s"] = 6.0
            f.attrs["hop_s"] = 0.5

    @abstractmethod
    def _extract_trials(self):
        pass


    def split_with_hops(self, data, label = None, window_s=6.0, hop_s=0.5, sampling_rate=128,
                        drop_last=True, channels_expected=62):
        """
        Always returns: list of (participant, segment, label)
        segment shape: (C, win_samples)
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        # Ensure (C, T)
        if data.shape[0] != channels_expected and data.shape[1] == channels_expected:
            data = data.T

        C, T = data.shape
        if C != channels_expected:
            raise ValueError(f"Expected {channels_expected} channels, got shape {data.shape}")

        win = int(round(window_s * sampling_rate))
        hop = int(round(hop_s * sampling_rate))
        if win <= 0 or hop <= 0:
            raise ValueError(f"Bad win/hop: win={win}, hop={hop}")

        out = []

        if T < win:
            if not drop_last:
                out.append((data, label))
            return out

        last_start = T - win
        for start in range(0, last_start + 1, hop):
            out.append((data[:, start:start + win], label))

        if not drop_last and (last_start % hop) != 0:
            out.append((data[:, -win:], label))

        return out

    

    def apply_preprocessing_pretrain(self, array):
        raw_mne_object = self.mne_process.create_mne_object(array, "dataset")
        raw_mne_object.notch_filter(freqs=50, method="iir")
        raw_mne_object.notch_filter(freqs=60, method="iir")
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method="iir")
        raw_mne_object.resample(sfreq=128.0)
        eeg_data = raw_mne_object.get_data()
        return eeg_data

if __name__ == "__main__":
    data_import = ImportSEED()
    data_import()
    #file_path = "/Volumes/Elements/EEG_data/pretraining/SEED/5_20140411.mat"
    #mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    #print(mat.keys())
    
   


    