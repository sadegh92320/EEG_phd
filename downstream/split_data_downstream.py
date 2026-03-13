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

class DataSplitter:
    def __init__(self, evaluation_scheme, config):
        self.evaluation_scheme = evaluation_scheme
        self.config = config
        self.num_participants = self.config["num_participants"]
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def get_evaluation_scheme(self):
        return self.evaluation_scheme

    def split_data(self):
        if self.evaluation_scheme == "leave_one_participant_out":
            self.test_data = random.sample(range(self.num_participants), 1)
            self.train_data = [p for p in range(self.num_participants) if p not in self.test_data]
            self.train_data, self.val_data = self.train_val_split(self.train_data)
        elif self.evaluation_scheme == "population":
            self.train_data = range(self.num_participants)
            self.test_data = range(self.num_participants)
            self.train_data, self.val_data = self.train_val_split(self.train_data)
        elif self.evaluation_scheme == "per_subject":
            self.train_data = range(self.num_participants)
            self.test_data = range(self.num_participants)
            self.train_data, self.val_data = self.train_val_split(self.train_data)
        else:
            raise ValueError(f"Unknown evaluation scheme: {self.evaluation_scheme}")
    
    def train_val_split(self, data, val_ratio = 0.2):
        size_train = int(len(data) * (1-val_ratio))
        data = random.shuffle(data)
        train_data = data[:size_train]
        val_data = data[size_train:]
        return train_data, val_data
    


class DownstreamDataLoader:
    def __init__(self, data_path, config, train_val_test_splitter):
        self.data_path = data_path
        self.config = config
        self.datasets = []
        self.train_loaders = []
        self.test_loaders = []
        self.val_loaders = []
        self.train_val_test_splitter = train_val_test_splitter
        self.whole_train_loader = None
        self.whole_val_loader = None
        self.whole_test_loader = None




    def load_participant_data(self, participant_number):
        # Load the data from the path and return it as a list of (participant, segment) tuples
        raw_data = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".npz"):
                participant_number = int(filename.split("_")[0])  # Assuming filename format is "participant_segment.npz"
                if participant_number == participant_number:
                    data = np.load(os.path.join(self.data_path, filename))
                    x = data["x"]
                    y = data["y"]
                    raw_data.append((x, y))
        return raw_data
    
    def make_dataset(self, raw_data, custom_dataset_class, participant_number):
        # Convert the raw data into a PyTorch Dataset
        return custom_dataset_class(raw_data, participant_number)
    
    def create_datasets(self, participant_numbers, custom_dataset_class):
        for participant_number in participant_numbers:
            raw_data = self.load_participant_data(participant_number)
            dataset = self.make_dataset(raw_data, custom_dataset_class, participant_number)
            self.datasets.append(dataset)
        return self
    
    def make_loader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
    
    def indiviudal_loader(self):
        for dataset in self.datasets:
            batch_size = self.config["batch_size"]
            shuffle = True
            loader = self.make_loader(dataset, batch_size, shuffle)
            if dataset.participant_number in self.train_val_test_splitter.train_data:
                self.train_loaders.append(loader)
            elif dataset.participant_number in self.train_val_test_splitter.val_data:
                self.val_loaders.append(loader)
            elif dataset.participant_number in self.train_val_test_splitter.test_data:
                self.test_loaders.append(loader)
    
    def load_data_whole(self, raw_data):
        for dataset in self.datasets:
            batch_size = self.config["batch_size"]
            shuffle = True
            train = []
            val = []
            test = []
            if dataset.participant_number in self.train_val_test_splitter.train_data:
                train.append(dataset)
            elif dataset.participant_number in self.train_val_test_splitter.val_data:
                val.append(dataset)
            elif dataset.participant_number in self.train_val_test_splitter.test_data:
                test.append(dataset)

        self.whole_train_loader = self.make_loader(ConcatDataset(train), batch_size, shuffle)
        self.whole_val_loader = self.make_loader(ConcatDataset(val), batch_size, shuffle)
        self.whole_test_loader = self.make_loader(ConcatDataset(test), batch_size, shuffle)

    