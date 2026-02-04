from export_data.export_data import DataImport
import os
import numpy as np
import yaml
from process_data.mne_constructor import MNEMethods
from math import ceil

class StewImport(DataImport):
    def import_data(self):
        """Import the stew dataset which is stored in txt file"""
        data_eeg = []
        l = sorted(os.listdir(self.config["input_data_path"]))
        rating = l.pop(0)
        participant = {}
        with open(self.config["input_data_path"] + "/" + rating) as f:
            for line in f:
                rating_split = line.strip().split(",")
                participant[str(int(rating_split[0]))] = [int(rating_split[1]), int(rating_split[2])]
        for file in l:
            filename = os.fsdecode(file)
            array = np.loadtxt(self.config["input_data_path"] + "/" + filename, dtype=float)
            sli = filename.replace(".","_").replace("sub","").split("_")
            if int(sli[0]) in [42, 5, 24]:
                continue
            if sli[1] == "hi":
                label = participant[str(int(sli[0]))][0]
            if sli[1] == "lo":
                label = participant[str(int(sli[0]))][1]
            
            if not os.path.isdir(self.config["output_data_path"]): 
                os.mkdir(self.config["output_data_path"])
            label = ceil(label/3) - 1
            data_eeg.append((array, label))
        return data_eeg
    
    def label_converter(self, data):
        scale = lambda x: (x/3) 
        data = scale(data)

        return np.ceil(data) - 1



if __name__ == "__main__":
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    data = StewImport(config=config, mne_process=MNEMethods(config=config))
    print(data.label_converter(np.array([1,2,3,4,5,6,7,8,9])))