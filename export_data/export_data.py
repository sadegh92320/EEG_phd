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

    def split(self, data, label = None, segment_length=3, overlap=0, sampling_rate=250):
        """
        Split the eeg in several segment
        data:  shape (C, T) after transpose
        label: single label for this trial (scalar or 0-d/1-d)
        """
        # make sure it's (C, T)
        data = data.transpose() if data.shape[0] > data.shape[1] else data
        C, T = data.shape
        print(f"data shape (C, T): {data.shape}")

        
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length 

        if step <= 0:
            raise ValueError(
                f"step computed as {step}. Check segment_length={segment_length} and overlap={overlap}."
            )

        # handle short signals: if T < data_segment, we still create 1 segment
        if T <= data_segment:
            number_segment = 0
        else:
            number_segment = (T - data_segment) // step

        segments = []
        new_labels = []

        for i in range(number_segment + 1):
            start = i * step
            end = start + data_segment

            # safety in case of boundary issues
            if end > T:
                end = T
                start = max(0, end - data_segment)

            seg = data[:, start:end]    # shape (C, segment_length_in_samples)
            segments.append(seg)
            new_labels.append(label)    # same label for all segments of this trial

        print(f"Created {len(segments)} segments")
        if label != None:
            return segments, new_labels
        else:
            return segments

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
    
    
    def save_data(self):
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
        
        
        