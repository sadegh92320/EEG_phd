import torch
import numpy as np
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import os
import random
from torch.utils.data import DataLoader


#Get the evaluation scheme for downstream tasks
#Get the number of participants
#Use the scheme to define the test set
#use the rest of the data for training and validation
#4 types of evaluation schemes: population, per-subject, subject-transfer, per-subject-transfer
#for population the output is a train and val loader with all the data
#for per-subject, the output is a list of train, val and test loader for each participant
#for subject-transfer, the output is a list of test loaders for each participant, where the test loader is trained on all the other participants
#for per-subject-transfer, the output is a list of train, val and test loader for each participant, where the test loader is
    
    


class DownstreamDataLoader:
    def __init__(self, data_path, config):
        random.seed(92)  # Set a fixed seed for reproducibility
        self.data_path = data_path
        self.config = config
        self.datasets = []
        self.train_loaders = []
        self.test_loaders = []
        self.val_loaders = []
        self.whole_train_loader = None
        self.whole_val_loader = None
        self.whole_test_loader = None
        self.participant_data = self.load_participant_data()  # Load all participant data at initialization
        self.val_data = os.path.join(self.data_path, "val.h5")
        self.train_data = os.path.join(self.data_path, "train.h5")
        self.num_participants = self.get_number_of_participants()


    def get_number_of_participants(self):
        """Get the number of participants in the dataset."""
        with h5py.File(self.train_data, 'r') as f:
            particpant = f['participant'][:]
            return len(np.unique(particpant))
        
    def load_participant_data(self):
        # Load the data from the path and return it as a list of (participant, segment) tuples
        participant_data = {}

        
       
        for filename in os.listdir(self.data_path): 
            train_val = {}
            try:
                part_nb = int(filename)  # Assuming each file is named with the participant number
            except ValueError:
                continue  # Skip files that don't match the expected format
            
            val_path = os.path.join(self.data_path, filename, "val")
            train_path = os.path.join(self.data_path, filename, "train")
            if not os.path.exists(val_path) or not os.path.exists(train_path):
                raise FileNotFoundError(f"Expected 'val' and 'train' directories for participant {part_nb} in {self.data_path}")
            for val_file in os.listdir(val_path):
                path = os.path.join(val_path, val_file)
                try:
                    train_val["val"].append(path)
                except KeyError:
                    train_val["val"] = [path]
            for train_file in os.listdir(train_path):
                path = os.path.join(train_path, train_file)
                try:
                    train_val["train"].append(path)
                except KeyError:
                    train_val["train"] = [path]
            participant_data[part_nb] = train_val
        
        return participant_data
    
    def get_data_for_population(self):
        """Get the data for population evaluation, which includes all the data from all participants for training and validation."""
        # For population evaluation, we can use all the data for training and testing
        train_data = []
        val_data = []
        test_data = []
        for part_nb, train_val in self.participant_data.items():
            train, val = self.split_train_val(train_val["train"], val_ratio=0.2)
            val_data.extend(val)
            train_data.extend(train)
            test_data.extend(train_val["val"])
        train_dataset = self.make_dataset(train_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
        val_dataset = self.make_dataset(val_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
        test_dataset = self.make_dataset(test_data, custom_dataset_class=None)  # You can define a custom dataset class if needed

        return train_dataset, val_dataset, test_dataset
    


    def get_full_subject_dataset(self, participant_number):
        """Get the full data for a specific participant, which includes all the data from that participant for training and validation."""
        with h5py.File(self.train_data, 'r') as f:
            particpant = f['participant'][:]
            indices = np.where(particpant == participant_number)[0]
            train_data = f['x'][indices]
            train_labels = f['y'][indices]
        with h5py.File(self.val_data, 'r') as f:
            particpant = f['participant'][:]
            indices = np.where(particpant == participant_number)[0]
            val_data = f['x'][indices]
            val_labels = f['y'][indices]
        x_data = np.concatenate([train_data, val_data], axis=0)
        y_data = np.concatenate([train_labels, val_labels], axis=0)
        dataset = self.make_dataset(x_data, y_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
        return dataset
    
    def get_loso_train_dataset(self, participant_number):
        """Get the training and validation data for leave-one-subject-out evaluation, which includes all the data from all participants except the test participant for training and validation."""
        with h5py.File(self.train_data, 'r') as f:
            particpant = f['participant'][:]
            indices = np.where(particpant != participant_number)[0]
            x = f['x'][indices]
            y = f['y'][indices]
            train_dataset = self.make_dataset(x, y, custom_dataset_class=None)

        with h5py.File(self.val_data, 'r') as f:
            particpant = f['participant'][:]
            indices = np.where(particpant != participant_number)[0]
            x = f['x'][indices]
            y = f['y'][indices]
            val_dataset = self.make_dataset(x, y, custom_dataset_class=None)

            # You can define a custom dataset class if needed
        
        return train_dataset, val_dataset
    
    def get_data_for_leave_one_participant_out(self):
        """Get the data for leave-one-subject-out evaluation, which includes all the data from all participants except the test participant for training and validation."""
        loso_data = []
        for part_nb in (self.num_participants):
            test_data = self.get_full_subject_dataset(part_nb)
            train_data, val_data = self.get_loso_train_dataset(part_nb)
            loso_data.append((train_data, val_data, test_data))
        return loso_data
       
    
    def per_subject(self, participant_number):
        """
            Get the training, validation and test data for a specific participant, 
            which includes all the data from that participant for training and validation, 
            and the test data is the data from that participant.
        """
        train_val = self.participant_data[participant_number]
        train_data = train_val["train"]
        train_data, val_data = self.split_train_val(train_data, val_ratio=0.2)
        test_data = train_val["val"]
        val_dataset = self.make_dataset(val_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
        train_dataset = self.make_dataset(train_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
        test_dataset = self.make_dataset(test_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
    
        return train_dataset, val_dataset, test_dataset            
       

    def get_subject_transfer(self, participant_number):
        """
            Get the test data for subject-transfer evaluation, 
            which includes all the data from all participants except 
            the test participant for training and validation, 
            and the test data is the data from the test participant.
        """
        # For subject-transfer evaluation, we can use the data from all participants except the test participant for training and validation
        test_data = []
        for part_nb, train_val in self.participant_data.items():
            if part_nb != participant_number:
                test_data.extend(train_val["train"])
                test_data.extend(train_val["val"])
        test_dataset = self.make_dataset(test_data, custom_dataset_class=None)  # You can define a custom dataset class if needed
       
        
        return test_dataset
            
    def get_per_subject_transfer(self):
        """
            Get the training, validation and test data for per-subject-transfer evaluation, 
            which includes all the data from all participants except the test participant for training and validation, 
            and the test data is the data from the test participant.
        """
        transfer = []
        for part_nb in self.participant_data.keys():
            train_participant, val_participant, test_participant = self.per_subject(participant_number=part_nb)
            test_loaders = self.get_subject_transfer(participant_number=part_nb)
            transfer.append((train_participant, val_participant, test_participant, test_loaders))
        return transfer
    
    def split_train_val(self, train_data, val_ratio=0.2):
        """Split the training data into training and validation sets based on the specified validation ratio."""
        n_val = int(len(train_data) * val_ratio)
        val_data = random.sample(train_data, n_val)
        train_data = [data for data in train_data if data not in val_data]
        return train_data, val_data
           

    def make_dataset(self, raw_data, custom_dataset_class):
        """Convert the raw data into a PyTorch Dataset using the specified custom dataset class."""
        # Convert the raw data into a PyTorch Dataset
        return custom_dataset_class(raw_data)
    
    
    
if __name__ == "__main__":
    import h5py
    # Open the file in read mode
    with h5py.File('downstream/data/upper_limb/train.h5', 'r') as f:
        # List the 'folders' (Groups) at the top level
        print("Keys in the file:", list(f.keys()))
        print("train data shape:", f['participant'])
        indices = np.where(f['participant'][:] != 5)[0]
        print("train data shape after filtering participant 5:", indices.shape)
        print("train data shape after filtering participant 5:", f['x'][indices].shape)