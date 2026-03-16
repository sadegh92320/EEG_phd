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


class ImportP300(DataImport):

    def get_config(self):
        self.config = r"MAE_pretraining\info_dataset\p300.yaml"
    
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = "D:\EEG_data\pretraining\P300"
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

class ImportP300(ImportDataPre):

    def get_participant_number(self, file: Path):
        
        participant_nb = file.name[1:3]
        return int(participant_nb)

    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".mat":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/p300.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        
        trials = []

        with h5py.File(file_path, "r") as f:
                train = f["train"][()].reshape(-1)
                test = f["test"][()].reshape(-1)
                g0 = f[train[0]]
                g1 = f[test[0]]

                
                X1 = g0["data"][()]
                X2 = g1["data"][()]
                if not isinstance(X1, np.ndarray):
                    print(X1)
                    raise Exception("not array")
                
                if not isinstance(X2, np.ndarray):
                    print(X2)
                    raise Exception("not array")

                if X1.ndim != 2:
                    raise Exception("wrong dim")
                
                if X2.ndim != 2:
                    raise Exception("wrong dim")

                if X1.shape[0] != 32 and X1.shape[1] == 32:
                    X1 = X1.T

                if X2.shape[0] != 32 and X2.shape[1] == 32:
                    X2 = X2.T
                
                X1 = self.apply_preprocessing_pretrain(X1)
                X2 = self.apply_preprocessing_pretrain(X2)
                trials.extend([X1,X2])

        return trials
    


if __name__ == "__main__":
    data_import = ImportP300()
    data_import.import_data(input_dir="/Volumes/Elements/EEG_data/pretraining/P300", output_dir="MAE_pretraining/data/p300")
   


    