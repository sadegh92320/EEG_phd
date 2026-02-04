from torcheeg.datasets import M3CVDataset, TSUBenckmarkDataset, DEAPDataset, SEEDDataset, moabb
import os
import torch
import shutil
import random
import mne

import pandas as pd
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
import copy
import numpy as np
import torcheeg
import torch
from torcheeg.datasets import M3CVDataset, TSUBenckmarkDataset, DEAPDataset, SEEDDataset, moabb
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LIST, M3CV_CHANNEL_LIST, TSUBENCHMARK_CHANNEL_LIST
from torcheeg.datasets import CSVFolderDataset


data_root_path = "./io_root/"


PHYSIONETMI_CHANNEL_LIST = ['Fc5.', 'Fc3.', 'Fc1.', 
                            'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 
                            'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 
                            'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 
                            'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 
                            'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 
                            'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 
                            'Oz..', 'O2..', 'Iz..']
PHYSIONETMI_CHANNEL_LIST = [x.strip('.').upper() for x in PHYSIONETMI_CHANNEL_LIST]


use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
    


def get_physionet_dataset():
    channels_name = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']

    """
    In summary, the experimental runs were:

    1   Baseline, eyes open
    2   Baseline, eyes closed
    3   Task 1 (open and close left or right fist)                  -> 4 5
    4   Task 2 (imagine opening and closing left or right fist)     -> 0 1
    5   Task 3 (open and close both fists or both feet)             -> 6 7
    6   Task 4 (imagine opening and closing both fists or both feet)-> 2 3
    7   Task 1
    8   Task 2
    9   Task 3
    10  Task 4
    11  Task 1
    12  Task 2
    13  Task 3
    14  Task 4

    """
    session_id2task_id = {
        3:1, 4:2, 5:3, 6:4,
        7:1, 8:2, 9:3, 10:4,
        11:1, 12:2, 13:3, 14:4,
        
    }
    task2event_id = {
        0:dict([('T1', 4), ('T2', 5)]),
        1:dict([('T1', 0), ('T2', 1)]),
        2:dict([('T1', 6), ('T2', 7)]),
        3:dict([('T1', 2), ('T2', 3)])
    }

    
    if not os.path.exists(data_root_path+'io/PhysioNetMI'):
        src_path = "/Volumes/Elements/EEG_data/files-2"
        ls = []
        channels_name = None
        for subject in range(1,110):
            for task in [0,1,2,3]:
                for session in [3,7,11]:
                    session += task
                    file_path = src_path + "/S{:03d}".format(subject) + '/' + "S{:03d}R{:02d}.edf".format(subject,session)
                    raw = mne.io.read_raw_edf(file_path,preload=True)
                    
                    if channels_name is None:
                        channels_name = copy.deepcopy(raw.ch_names)
                    else:
                        assert channels_name == raw.ch_names
                        
                    event_id = task2event_id[session_id2task_id[session]-1]
                    # -- split epochs
                    epochs = mne.Epochs(raw, 
                            events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                            tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                            preload=True, 
                            decim=1,
                            baseline=None, 
                            reject_by_annotation=False)
                    
                    d = {
                        # "subject_id":[subject],
                        # "sess_id": [session],
                        # "task_id":[task],
                        "file_path":[file_path],
                        "labels":"".join([str(ev[-1]) for ev in epochs.events])
                    }
                    
                    ls.append(pd.DataFrame(d))
        table = pd.concat(ls, ignore_index=True)
        # print(table)
        print(channels_name)
        table.to_csv("/Volumes/Elements/EEG_data/files-2/physionetmi_meta.csv", index=False)

        def default_read_fn(file_path, task_id=None, session_id=None, subject_id=None, **kwargs):
            session_id = int(file_path.split('R')[-1].split('.')[0])
            # -- read raw file
            raw = mne.io.read_raw_edf(file_path,preload=True)
            
            event_id = task2event_id[session_id2task_id[session_id]-1]
            # -- split epochs
            epochs = mne.Epochs(raw, 
                    events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                    tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                    preload=True, 
                    decim=1,
                    baseline=None, 
                    reject_by_annotation=False)
            
            return epochs

        dataset = CSVFolderDataset(csv_path="/Volumes/Elements/EEG_data/files-2/physionetmi_meta.csv",
                                read_fn=default_read_fn,
                                io_path=data_root_path+'io/PhysioNetMI',
                            online_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.To2d()
                            ]),
                                #    label_transform=transforms.Select('label'),
                                num_worker=4)
    dataset = CSVFolderDataset(
                        io_path=data_root_path+'io/PhysioNetMI',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, PHYSIONETMI_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=160*4, p=1.0),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*6) * 1e3), #V-> 1000uV
                            transforms.To2d()
                        ]),
                        label_transform=transforms.Compose([
                            #   transforms.Select('labels'),
                            #   transforms.StringToInt()
                            transforms.Lambda(lambda x : 0)
                        ]))
    return dataset


# --------------- merge to Fold Dataset

def get_TSU_dataset():
    dataset = TSUBenckmarkDataset(
            root_path="/Volumes/Elements/EEG_data/TSUBenchmark",
            io_path=data_root_path+'io/tsu_benchmark',
            online_transform=transforms.Compose([
                transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, TSUBENCHMARK_CHANNEL_LIST)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                transforms.To2d(),
            ]),
            label_transform=transforms.Select('trial_id'))
    return dataset


def get_M3CV_dataset():
    dataset = M3CVDataset(
                        root_path="./aistudio/",
                        io_path=data_root_path+'io/m3cv',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, M3CV_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                            transforms.To2d(),
                        ]),
                        label_transform=transforms.Compose([
                            transforms.Select('subject_id'),
                            transforms.StringToInt()
                        ]))
    return dataset

def get_SEED_dataset():
    dataset = SEEDDataset(
                            root_path="/Volumes/Elements/EEG_data/SEED",
                            io_path="/Volumes/Elements/EEG_data/torcheeg_io/seed",
                          online_transform=transforms.Compose([
                              transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, SEED_CHANNEL_LIST)),
                              transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=250*4, p=1.0),
                              transforms.Lambda(lambda x: temporal_interpolation(x, 256*10)/1000),# 1000uV
                              transforms.To2d(),                          
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('emotion'),
                              transforms.Lambda(lambda x: x + 1)
                          ]))
    return dataset

#transforms.Lambda(
#    lambda x: (
#        lambda y: (y - y.mean(dim=-1, keepdim=True)) /
#                  (y.std(dim=-1, keepdim=True) + 1e-6)
#    )(temporal_interpolation(x, 256*4) / 1000)
#)

if __name__ == "__main__":
    pass
    #dataset = get_physionet_dataset()
    #for i, (x,y) in enumerate(dataset):
    #    x = x.squeeze(0)
    #    output = "MAE_pretraining/data_mae/data"
    #    os.makedirs(output, exist_ok=True)
    #    filename = os.path.join(output, f"physioNET{i}.npz")
    #    np.savez(filename, x=x)
        