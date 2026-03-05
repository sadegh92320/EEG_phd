from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportP300(DataImport):

    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/p300.yaml"
    
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = self.config["input_data_path"]
        to_check = sorted(os.listdir(path))
        for file in to_check:
            participant_nb = file[1:3]
            if not file.endswith(".mat") or file.startswith("._"):
                continue
            mat_path = os.path.join(path, f"{file}")
            with h5py.File(mat_path, "r") as f:
                train = f["train"][()].reshape(-1)
                test = f["test"][()].reshape(-1)
                g0 = f[train[0]]
                g1 = f[test[0]]

                
                X1 = g0["data"][()]
                X2 = g1["data"][()]
                if X1.shape[0] != 32 and X1.shape[1] == 32:
                    X1 = X1.T

                if X2.shape[0] != 32 and X2.shape[1] == 32:
                    X2 = X2.T

                data_eeg.extend([(participant_nb, X1), (participant_nb, X2)])



        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=6, hop_s=0.5,
                                                              sampling_rate=128, channels_expected=32)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self


if __name__ == "__main__":
    path = "/Volumes/Elements/EEG_data/pretraining/P300/s01.mat"
   


    with h5py.File(path, "r") as f:
        refs = f["test"][()].reshape(-1)
        g0 = f[refs[0]]

        
        X = g0["data"][()]
        print(X.shape)