from torch.utils.data import Dataset
import os
import h5py
import torch


class UpperLimbDataset(Dataset):
    def __init__(self, data_path, fold=0, classification_task="motorimagination", data_length=None, train=True, class_label=None, transform=None, chan_info=None):
        """
        Args:
            data_path: path for the .h5 file
            fold (int): The index (0 to 4) specifying which CV fold parameters to use.
            classification_task (str): which classification task to use 
            data_length (int, optional): data_length to use
            train (bool): If True, select trials for training (i.e. those NOT marked as test).
                          If False, select test trials.
            class_label (dict): maps the class names to integers
            transform (callable, optional): Any additional transformation to apply on the sample.
            chan_info (torch tensor, optional): Any additional channel information that will be passed into the model 
        """
        self.subjectName = os.path.splitext(os.path.basename(data_path))[0]
        self.fold = fold
        self.classification_task = classification_task
        self.data_length = data_length
        self.train = train
        self.class_label = class_label
        self.transform = transform
        self.chan_info = chan_info
        
        # load the dataset
        with h5py.File(data_path,'r') as f5:
            X_loaded = f5['X'][()]
            classes   = f5['df/class'][()].astype(str)
            # get the current fold indices
            if self.train:
                indices = f5[f"folds/{classification_task}/fold_{fold}/train"][()]
            else:
                indices = f5[f"folds/{classification_task}/fold_{fold}/test"][()]
            self.data = X_loaded[indices,:,:]
            self.labels = classes[indices]
        
        if self.transform:
            self.data = self.transform(self.data)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        label = self.labels[idx]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label]
