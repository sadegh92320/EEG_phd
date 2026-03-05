from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportSSVEP(DataImport):

    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/ssvep.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "/Volumes/Elements/EEG_data/pretraining/Online_MI_BCI_Classification"

        for file in sorted(os.listdir(path)):
            s = file.split(".")[0][1:]
            participant_nb = int(s)
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            mat_path = os.path.join(path, file)

        

            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            eeg = (mat["data"].EEG)
            ch = eeg.shape[0]
            eeg = eeg.reshape(ch, -1)

            
            if eeg.shape[0] != 64 and eeg.shape[1] == 64:
                eeg = eeg.T

            data_eeg.append((participant_nb, eeg))

        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=6, hop_s=2.5,
                                                              sampling_rate=128, channels_expected=62)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self
    


if __name__ == "__main__":
    path = "/Volumes/Elements/EEG_data/pretraining/SSVEP/S1.mat"
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    eeg = (mat["data"].EEG)
    ch = eeg.shape[0]
    eeg = eeg.reshape(ch, -1)
    print(eeg.shape)