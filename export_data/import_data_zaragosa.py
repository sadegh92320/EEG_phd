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


class ImportDataZaragoza(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = int(file.name)
        return int(participant_nb)
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".mat":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/im_lab.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        file_path = os.path.join(file_path,f"{file_path.name}.mat")
        with h5py.File(file_path, "r") as f:
                y = np.array(f["y"])
                eeg = y[:,1:33]
                trials.append(eeg.T)
        return trials




if __name__ == "__main__":
    pass
    




    



