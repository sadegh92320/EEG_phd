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
        self.config = r"MAE_pretraining\info_dataset\ssvep.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "D:\EEG_data\pretraining\SSVEP"

        for file in sorted(os.listdir(path)):
            
        
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            s = file.split(".")[0][1:]
           
            participant_nb = int(s)

            mat_path = os.path.join(path, file)

        
            print(mat_path)
            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
           
            eeg = (mat["data"].EEG)
            eeg = np.delete(eeg, [60,63], axis=0)
            
            if eeg.shape[0] != 62 and eeg.shape[1] == 62:
                print(eeg.shape)
                eeg = eeg.T

            C,T,B,K = eeg.shape
            
            eeg = np.transpose(eeg, (2,3,0,1))
            eeg = eeg.reshape(-1,C,T)
            list_eeg = [(participant_nb, eeg[i]) for i in range(B*K)]
            data_eeg.extend(list_eeg)

        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=2, hop_s=0.125,
                                                              sampling_rate=128, channels_expected=62)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self
    


if __name__ == "__main__":
    data_import = ImportSSVEP()
    data_import().preprocessing().split_train_val().save_data_pretrain()