from scipy.signal import resample
import numpy as np

def resample_eeg(eeg, previous_freq, new_freq):
    """Resample data with new frequency"""
    B, C, T = eeg.shape
    new_t = int(round((new_freq*T)/previous_freq))
    resample_data = resample(x=eeg,num=new_t,axis=2)
    return resample_data


def standardize_channel(eeg):
    #size of eeg B,C,T
    mean = np.mean(eeg, axis=-1, keepdims=True)
    std = np.std(eeg, axis=-1, keepdims=True) + 1e-8

    
    eeg = (eeg - mean)/std

    return eeg