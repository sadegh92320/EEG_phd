from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportLargeMI(DataImport):

    def get_config(self):
        self.config = r"MAE_pretraining\info_dataset\LMI_C.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "D:\EEG_data\pretraining\Experiment_CLA"

        for file in sorted(os.listdir(path)):
            s = file.split("-")[1][-1]
            participant_nb = ord(s.upper()) - ord('A') + 1
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            mat_path = os.path.join(path, file)

        

            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            data = mat["o"]
            data = data.data[:,:21]

            
            if data.shape[0] != 21 and data.shape[1] == 21:
                data = data.T

            data_eeg.append((participant_nb, data))

        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=6, hop_s=2.5,
                                                              sampling_rate=128, channels_expected=21)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self
    



if __name__ == "__main__":
    data_import = ImportLargeMI()
    data_import().preprocessing().split_train_val().save_data_pretrain()
    
    #data =  loadmat(path, struct_as_record=False, squeeze_me=True)
    #data = (data["o"])
    #print(data.data.shape)
    #data = data.data[:, :21]
    #print(data.shape)