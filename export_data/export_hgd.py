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




class ImportHGD(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = file.name.split("_")[0]
        participant_nb = participant_nb.replace("sub", "")
        participant_nb = int(participant_nb)
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".edf":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/hgd.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []

       
        data = mne.io.read_raw_edf(file_path)
            
        channels_to_remove = ['EOG EOGh', 'EOG EOGv', 
                            'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF']

        data.drop_channels(channels_to_remove)
        data = (data.get_data())
                    
        if data.shape[0] != 128 and data.shape[1] == 128:
            data = data.T

        trials.append(data)

        return trials


if __name__ == "__main__":
    pass
    



