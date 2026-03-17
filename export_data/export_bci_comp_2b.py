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



class ImportBCIComp2b(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = int(file[2])
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".gdf":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/bci_comp_2b.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        raw = mne.io.read_raw_gdf(file_path, preload=True)
        raw = raw.get_data()
        if raw.shape[0] != 6 and raw.shape[1] == 6:
                raw = raw.T
        data = raw[:3,:]
        trials.append(data)

        return trials


if __name__ == "__main__":
    pass
    



