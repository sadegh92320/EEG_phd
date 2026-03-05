from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportKU(DataImport):

    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/auditory.yaml"
    
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = self.config["input_data_path"]
        to_check = sorted(os.listdir(path))
        for file in to_check:
            participant_nb = file[4:7]
            if not file.endswith(".npy") or file.startswith("._"):
                continue
            path = os.path.join(path, f"{file}")
            data = np.load(path)
            
            if data.shape[0] != 32 and data.shape[1] == 32:
                data = data.T


            data_eeg.extend([(participant_nb, data), (participant_nb, data)])



        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=6, hop_s=0.5,
                                                              sampling_rate=128, channels_expected=64)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self


if __name__ == "__main__":
    data = np.load('/Volumes/Elements/EEG_data/pretraining/KU/sub-001_ses-shortstories01_task-listeningActive_run-01_desc-preproc-audio-audiobook_5_1_eeg.npy')
    print(data.shape)