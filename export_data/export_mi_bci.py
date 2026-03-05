from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportMIBCI(DataImport):

    def get_config(self):
        self.config = "MAE_pretraining\info_dataset\eeg_mi_bci.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "D:\EEG_data\pretraining\ EEG-MI-BCI"

        for file in sorted(os.listdir(path)):
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            mat_path = os.path.join(path, file)

            # Extract subject number (S10_Session_1.mat → 10)
            participant_nb = file.split(".")[0][1:]

            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        
            data_eeg.append((participant_nb, np.asarray(mat["eeg"].movement_left[:64,:], dtype=np.float32)))
            data_eeg.append((participant_nb, np.asarray(mat["eeg"].movement_right[:64,:], dtype=np.float32)))
        
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
    data_import = ImportMIBCI()
    data_import().preprocessing().split_train_val().save_data_pretrain()