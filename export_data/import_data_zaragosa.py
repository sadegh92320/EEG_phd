from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os

class ImportDataZaragoza(DataImport):

    def get_config(self):
        self.config = self.config = r"MAE_pretraining\info_dataset\im_lab.yaml"
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = "D:\EEG_data\pretraining\Zaragoza"
        to_check = sorted(os.listdir(path))
        for file in to_check:
            mat_path = os.path.join(path, file,f"{file}.mat")
            part_nb = int(file)
            with h5py.File(mat_path, "r") as f:
                y = np.array(f["y"])
                eeg = y[:,1:33]
                print(eeg.shape)
                data_eeg.append((part_nb, eeg.T))

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
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    data = ImportDataZaragoza(config=config, mne_process=MNEMethods(config=config))
    data()