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



class ImportSEED(ImportDataPre):

    def get_participant_number(self, file: Path):
        s = file.name.split("_")[0]
        participant_nb = int(s)
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".cnt":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/seed2.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        raw = mne.io.read_raw_cnt(file_path, preload=True)
    
        raw.drop_channels(['VEO', 'HEO'])
        data = raw.get_data()
        data = self.apply_preprocessing_pretrain(data)
       
        trials.append(data)
        return trials

if __name__ == "__main__":
    data_import = ImportSEED(num_chan=64)
    data_import.import_data(input_dir="/Users/sadeghemami/Downloads/EEG_raw", output_dir="MAE_pretraining/data/seed")

    

    