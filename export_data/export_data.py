import os
from typing import Any
import numpy as np
from abc import ABC, abstractmethod
import yaml
import mne

class DataImport(ABC):
    def __init__(self, config, mne_process):
        self.config = config

        self.mne_process = mne_process

        self.data_dir = None

    def __call__(self):
        self.data = self.import_data()
        return self
    
    @abstractmethod
    def import_data(self):
        """import the specific data changes for every dataset"""
        pass




    def partition_data(self):
        """partition each recording in several slices"""
        data_partition = []
        for part_number, array, label in self.data:
            partitions = np.vsplit(array, self.config["num_slice"])
            for partition in partitions:
                data_partition.append((part_number, partition, label))
        self.data = data_partition
        return self
    
    def remove_artifacts(self):
        """Removes occular artifacts using eeg channels"""
        data_clean = []
        for part_number ,array, label in self.data:
            
            clean = self.mne_process.clean_raw_with_ica(array)
            data_clean.append((part_number, clean, label))
            
        self.data = data_clean
        return self
    
    def apply_bandpass(self, low, high):
        """apply bandpass filter to eeg recording"""
        raw_data = []
        for array, label in self.data:
            raw = self.mne_process.create_mne_object(array, 'dataset')
            raw.filter(low, high, fir_design='firwin')
            raw_filter = raw.get_data().T
            raw_data.append((raw_filter, label))
        self.data = raw_data
        return self
    
    def resamplesample(self, new_freq):
        """changes the frequency of the eeg recording"""
        raw_data = []
        for array, label in self.data:
            raw = self.mne_process.create_mne_object(array, 'dataset')
            raw.resample(new_freq)
            raw_filter = raw.get_data().T
            raw_data.append((raw_filter, label))
        self.data = raw_data
        return self
    
    
    def save_data(self, particition = False):
        """save the data in the folder"""
        out_dir = os.path.join(
            self.config["output_data_path"],
            self.config["experiment_folder"],
        )
        os.makedirs(out_dir, exist_ok=True)
        self.data_dir = out_dir

        base = self.config["output_data_file"]
        for i, (part_number,x, y) in enumerate(self.data):
            filename = os.path.join(out_dir, part_number)
            os.makedirs(filename, exist_ok=True)

            filename  = os.path.join(filename,f"{base}_{i}_{y}.npz")
            np.savez(filename, x=x, y=y)
        return self


if __name__ == "__main__":
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
   
    l = sorted(os.listdir("/Users/sadeghemami/Downloads/STEW Dataset"))
    rating = l.pop(0)
    participant = {}
    with open(f"/Users/sadeghemami/Downloads/STEW Dataset/{rating}") as f:
        for line in f:
            r = line.strip().split(",")
            participant[str(int(r[0]))] = [int(r[1]), int(r[2])]
    for file in l:
        filename = os.fsdecode(file)
        array = np.loadtxt(f'/Users/sadeghemami/Downloads/STEW Dataset/{filename}', dtype=float)
        sli = filename.replace(".","_").replace("sub","").split("_")
        if int(sli[0]) in [42, 5, 24]:
            continue
        if sli[1] == "hi":
            label = participant[str(int(sli[0]))][0]
        if sli[1] == "lo":
            label = participant[str(int(sli[0]))][1]
        
        if not os.path.isdir(config["data_path"]): 
            os.mkdir(config["data_path"])
        np.savez(config["data_path"] + "/stew_data.npz", x=array, y=label)
        
        
        