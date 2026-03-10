import os
from typing import Any
import numpy as np
from abc import ABC, abstractmethod
import yaml
import mne
from scipy.signal import resample
from process_data.mne_constructor import MNEMethods

class DataImport(ABC):
    def __init__(self):
        self.get_config()
        with open(self.config) as f:
            self.config = yaml.safe_load(f)
        

        self.mne_process = MNEMethods(self.config)

        self.data_dir = None

    def __call__(self):
        self.data = self.import_data()
        return self
    
    @abstractmethod
    def import_data(self):
        """import the specific data changes for every dataset"""
        pass

    @abstractmethod
    def get_config(self):
        pass


    def split_with_hops(self, data, participant, label = None, window_s=6.0, hop_s=0.5, sampling_rate=128,
                        drop_last=True, channels_expected=62):
        """
        Always returns: list of (participant, segment, label)
        segment shape: (C, win_samples)
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        # Ensure (C, T)
        if data.shape[0] != channels_expected and data.shape[1] == channels_expected:
            data = data.T

        C, T = data.shape
        if C != channels_expected:
            raise ValueError(f"Expected {channels_expected} channels, got shape {data.shape}")

        win = int(round(window_s * sampling_rate))
        hop = int(round(hop_s * sampling_rate))
        if win <= 0 or hop <= 0:
            raise ValueError(f"Bad win/hop: win={win}, hop={hop}")

        out = []

        if T < win:
            if not drop_last:
                out.append((participant, data, label))
            return out

        last_start = T - win
        for start in range(0, last_start + 1, hop):
            out.append((int(participant), data[:, start:start + win], label))

        if not drop_last and (last_start % hop) != 0:
            out.append((int(participant), data[:, -win:], label))

        return out
    
    def resample_eeg(eeg, previous_freq, new_freq):
        """Resample data with new frequency"""
        B, C, T = eeg.shape
        new_t = int(round((new_freq*T)/previous_freq))
        resample_data = resample(x=eeg,num=new_t,axis=2)
        return resample_data

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
    
    

    def apply_preprocessing_array(self, data):
       
        raw_mne_object = self.mne_process.create_mne_object(data, 'dataset')
        raw_mne_object.notch_filter(freqs=50, method = "iir")
        raw_mne_object.notch_filter(freqs=60, method = "iir")
            
        raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method='iir')
            
        raw_mne_object.resample(sfreq=128.0)
            
        eeg_data = raw_mne_object.get_data()
        
        
        return data
        
    def apply_preprocessing_pretrain(self):
        raw_data = []
        for participant, array in self.data:
            raw_mne_object = self.mne_process.create_mne_object(array, 'dataset')
            raw_mne_object.notch_filter(freqs=50, method = "iir")
            raw_mne_object.notch_filter(freqs=60, method = "iir")
            
            raw_mne_object.filter(l_freq=0.1, h_freq=64.0, method='iir')
            
            raw_mne_object.resample(sfreq=128.0)
            
            eeg_data = raw_mne_object.get_data()
            raw_data.append((participant, eeg_data))
        
        return raw_data
    
    
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
    
    def split_train_val(self, val_ratio=0.2):
        self.val_data = []
        self.train_data = []

        participants = sorted({p for p, _ in self.data})
        total_participants = len(participants)

        n_val = int(round(val_ratio * total_participants))
        n_train = total_participants - n_val

        train_participants = set(participants[:n_train])
        val_participants   = set(participants[n_train:])

        for p, seg in self.data:
            if p in train_participants:
                self.train_data.append((p,seg))  
            else:
                self.val_data.append((p,seg))

        return self

    def save_data_pretrain(self):
        out_dir_val = os.path.join(
            self.config["data_file"],
            "val"
        )
        out_dir_train = os.path.join(
            self.config["data_file"],
            "train"
        )
        os.makedirs(out_dir_train, exist_ok=True)
        os.makedirs(out_dir_val, exist_ok=True)
        self.data_dir_train = out_dir_train
        self.data_dir_val = out_dir_val

        for i, (part_number, x) in enumerate(self.val_data):
            filename = os.path.join(out_dir_val, f"{part_number}_{i}")
            np.savez(filename, x=x)
        print("save train")

        for i, (part_number, x) in enumerate(self.train_data):
            filename = os.path.join(out_dir_train, f"{part_number}_{i}")
            np.savez(filename, x=x)


    
    
    def save_data_downstream(self):
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
        
        
        