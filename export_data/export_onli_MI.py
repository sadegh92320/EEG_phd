from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
import torch


class ImportOnlineMI(DataImport):

    def get_config(self):
        self.config = "/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/online_bci_cla.yaml"
    
    def import_data(self):
        data_eeg = []
        path = "/Volumes/Elements/EEG_data/pretraining/Online_MI_BCI_Classification"

        for file in sorted(os.listdir(path)):
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            mat_path = os.path.join(path, file)

            # Extract subject number (S10_Session_1.mat → 10)
            m = re.match(r"S(\d+)_Session_", file)
            participant_nb = int(m.group(1)) if m else file.split("_")[0]

            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            BCI = mat["BCI"]

            for d in BCI.data:
                arr = np.asarray(d, dtype=np.float32)

                # safety: ensure (62, nTime)
                if arr.shape[0] != 62 and arr.shape[1] == 62:
                    arr = arr.T

                data_eeg.append((participant_nb, arr))

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
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    #data = ImportOnlineMI(config=config, mne_process=MNEMethods(config=config))
    #data()
    file = "/Users/sadeghemami/Downloads/S1_Session_1.mat"
    data =  loadmat(file, struct_as_record=False, squeeze_me=True)
    print(data["BCI"].data.shape)

    