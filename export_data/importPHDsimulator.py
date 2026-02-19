from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os

class ImportPhdSimulator(DataImport):
    
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = self.config["input_data_path"]
        to_check = sorted(os.listdir(path))
        for file in to_check:
            mat_path = os.path.join(path, f"{file}")
            seg_path = file.split("_")
            if "old" in seg_path or "time" in seg_path:
                print(seg_path)
                continue
            with h5py.File(mat_path, "r") as f:
                y = np.array(f["y"])
                eeg = y[:,1:33]
                print(eeg.shape)
                data_eeg.extend(eeg.T)

        return data_eeg
    
    
    
if __name__ == "__main__":
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    data = ImportPhdSimulator(config=config, mne_process=MNEMethods(config=config))
    data()