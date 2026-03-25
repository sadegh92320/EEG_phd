from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
from export_data.export_data_pretrain import ImportDataPre
from pathlib import Path
import torch
import mne



class ImportKU(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = int(file.name[4:7])
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".npy":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/auditory.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        data = np.load(file_path)
        if data.shape[0] != 64 and data.shape[1] == 64:
            data = data.T
        data = self.apply_preprocessing_pretrain(data)
        trials.append(data)

        return trials


if __name__ == "__main__":
    data_import = ImportKU(num_chan=64)
    data_import.import_data(input_dir="/Volumes/Elements/EEG_data/pretraining/KU", output_dir="MAE_pretraining/data/KU")
    



