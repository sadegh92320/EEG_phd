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




class ImportMIBCI(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = int(file.name.split(".")[0][1:])
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".mat":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/eeg_mi_bci.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
        data = np.asarray(mat["eeg"].movement_left[:64,:], dtype=np.float32)
        trials.append(data)
        data = np.asarray(mat["eeg"].movement_right[:64,:], dtype=np.float32)
        trials.append(data)

        return trials




if __name__ == "__main__":
    pass
    




    



