from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import mne
from mne.preprocessing import ICA
from process_data.preprocessing import Preprocessing
import yaml
import os
from datetime import datetime
from einops import rearrange, reduce, repeat

"""
x_raw = self.data[index] 
if self.preprocess is not None: 
    x = self.preprocess(x_raw) 
else: 
    x = x_raw 
return x, self.label[index]
"""

class EEGdataset(Dataset):
    def __init__(self, X, y, preprocess = None):
        self.data = X
        self.label = y
        self.preprocess = preprocess

    def __getitem__(self, index):
        x_raw = self.data[index]
        if self.preprocess is not None:
            x = self.preprocess(x_raw)
        else:
            x = x_raw
        x = torch.from_numpy(x).float()
        y = torch.tensor(self.label[index]).long()
        return x, y

    def __len__(self):
        return len(self.data)
    
    
    
if __name__ == "__main__":
    #p = Preprocessing()
    #list_channel = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    #dataset = EEGdataset(data_path="eeg_data.pkl", label_path="label.pkl")
    #x, y = dataset[0]

    #print(x.shape)
    #with open("setting.yaml") as f:
    #    config = yaml.safe_load(f)
    #print(config)
    data = [
    ("Name", "Age", "City"),
    ("Alice", "25", "New York"),
    ("Bob", "30", "San Francisco"),
    ("Charlie", "35", "London")
    ]

    for name, age, city in data:
        print(f"{name:<10}{age:^5}{city:>15}")
    metrics = {"accuracy": 0.85, "precision": 0.70}
    print(*[f"{k}: {v:.4f}" for k, v in metrics.items()], sep="\n")

    time = str(datetime.now())
    print(time)

    