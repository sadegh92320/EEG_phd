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


class ImportBCIComp2a(DataImport):

    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/bci_comp_2a.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "/Volumes/Elements/EEG_data/pretraining/BCICIV_2b_gdf"

        for file in sorted(os.listdir(path)):
            if not file.endswith(".gdf") or file.startswith("._"):
                continue

            gdf_path = os.path.join(path, file)

            # Extract subject number (S10_Session_1.mat → 10)
            participant_nb = int(file[2])

            raw = mne.io.read_raw_gdf(gdf_path, preload=True)
            if arr.shape[0] != 6 and arr.shape[1] == 6:
                    arr = arr.T
            data_eeg.append((participant_nb, raw[:3,:]))
            
        
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
    path = "/Volumes/Elements/EEG_data/pretraining/BCICIV_2b_gdf/B0101T.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True)
    data = raw.get_data()   # shape: (n_channels, n_samples)
    print(data[:3,:].shape)