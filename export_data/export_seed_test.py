from export_data.export_data_pretrain import ImportDataPre
from pathlib import Path
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

class ImportSEED(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = file.stem.split("_")[0]
        return int(participant_nb)

    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.name.startswith("label"):
            return True

        if file.suffix.lower() != ".mat":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/seed2.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        trials = []

        for key, value in mat.items():
            # skip matlab metadata
            if key.startswith("__"):
                continue

            # adjust this condition depending on actual SEED key names
            
            data = value

            if not isinstance(data, np.ndarray):
                continue

            if data.ndim != 2:
                continue

            if data.shape[0] != 62 and data.shape[1] == 62:
                data = data.T

            data = self.apply_preprocessing_pretrain(data)
            trials.append(data)

        return trials
    
if __name__ == "__main__":
    dataimport = ImportSEED()
    dataimport.import_data(input_dir="/Volumes/Elements/EEG_data/pretraining/SEED", output_dir="downstream/data/seed")



