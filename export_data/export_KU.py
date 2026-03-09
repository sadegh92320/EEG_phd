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
        self.config = r"MAE_pretraining\info_dataset\auditory.yaml"
    
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = "D:\EEG_data\pretraining\KU"
        to_check = sorted(os.listdir(path))
        for file in to_check:
            participant_nb = file[4:7]
            if not file.endswith(".npy") or file.startswith("._"):
                continue
            path_data = os.path.join(path, f"{file}")
            data = np.load(path_data)
            
            if data.shape[0] != 64 and data.shape[1] == 64:
                data = data.T
            print(data.dtype)

            
            data_eeg.append((participant_nb, data))



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
    data_import = ImportKU()
    data_import().preprocessing().split_train_val().save_data_pretrain()