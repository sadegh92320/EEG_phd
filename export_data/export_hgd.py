from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch
import mne


class ImportHGD(DataImport):

    def get_config(self):
        self.config = "MAE_pretraining\info_dataset\hgd.yaml"
    
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = "D:\EEG_data\pretraining\HGD"
        to_check = sorted(os.listdir(path))
        for file in to_check:
            participant_nb = file.split("_")[0]
            participant_nb = participant_nb.replace("sub", "")
            participant_nb = int(participant_nb)
            if not file.endswith(".edf") or file.startswith("._"):
                continue
            
            edf_path = os.path.join(path, file)
            
            
            data = mne.io.read_raw_edf(edf_path)
            print(data.ch_names)
            
            channels_to_remove = ['EOG EOGh', 'EOG EOGv', 
                            'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF']

            data.drop_channels(channels_to_remove)
            data = (data.get_data())
                    
            if data.shape[0] != 128 and data.shape[1] == 128:
                data = data.T


            data_eeg.extend([(participant_nb, data), (participant_nb, data)])



        return data_eeg
    
    def preprocessing(self):
        preprocess_data = self.apply_preprocessing_pretrain()
        data_splitted = []

        for p, d in preprocess_data:
            split_data = self.split_with_hops(data=d, participant=p,window_s=6, hop_s=0.5,
                                                              sampling_rate=128, channels_expected=128)
            zip_data = [(x[0], x[1]) for x in split_data]
            data_splitted.extend(zip_data)
        self.data = data_splitted
        return self


if __name__ == "__main__":
    data_import = ImportHGD()
    data_import().preprocessing().split_train_val().save_data_pretrain()