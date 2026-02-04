import torch
import mne
from mne.preprocessing import ICA
import numpy as np

class Preprocessing():
    def __init__(self, data = None, normalize = False, standardize = True):
        self.normal = normalize
        self.standard = standardize

        if self.standard and self.normal:
            raise TypeError("can't both normalize and standardize")
       
        if self.standard == True:
            print("data")
            print(data.shape)
            data = np.stack(data, axis=0)
            self.mean = data.mean(axis = (0,2))
            self.std = data.std(axis = (0,2))
            self.std[self.std == 0] = 1

        if self.normal == True:
            data = np.stack(data, axis=0)
            self.ma = data.max(axis = (0,2))
            self.mi = data.min(axis = (0,2))

    def __call__(self,X):
        assert not (self.normal and self.standard)
        if self.normal == True:
            X = self.normalize(X)
        if self.standard == True:
            X = self.standardize(X)
        
        return X

    def normalize(self, X):
        """normalize data with min max per channel over participants"""
        x = X - self.mi[:,None]
        return x/(self.ma[:,None] - self.mi[:,None])

    def standardize(self, X):
        """standardize per channel over participants"""     
        return (X - self.mean[:, None]) / self.std[:, None]
    
    
    