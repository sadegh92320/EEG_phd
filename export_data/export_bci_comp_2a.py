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





class ImportBCIComp2a(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = int(file.name[1:3])
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".gdf":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/bci_comp_2a.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []
        raw = mne.io.read_raw_gdf(file_path, preload=True)
        raw = raw.get_data()
        

        if raw.shape[0] != 25 and raw.shape[1] == 25:
                raw = raw.T

        raw = raw[:22,:]
        raw = self.apply_preprocessing_pretrain(raw)
        trials.append(raw)
        return trials


if __name__ == "__main__":
    data_import = ImportBCIComp2a(num_chan=22)
    data_import.import_data(input_dir="/Volumes/Elements/EEG_data/pretraining/BCICIV_2a_gdf",output_dir= "MAE_pretraining/data/bci_comp_2a")