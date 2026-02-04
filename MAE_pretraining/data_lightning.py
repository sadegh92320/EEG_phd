from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from dataset import EEGdataset
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from einops import rearrange, reduce, repeat
import os
import numpy as np
import torch
from process_data.preprocessing import Preprocessing
from torch.utils.data import DataLoader, random_split
import lightning as L  # or pytorch_lightning as pl

class EEGData(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.1,
                 num_workers=0, pin_memory=False, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.transform = Preprocessing
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._X = None
        self.preprocess = None 


    def _load_npz_folder(self):
        data, labels = [], []
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.lower().endswith(".npz"):
                continue
            path = os.path.join(self.data_dir, fname)
            npz = np.load(path)
            x = npz["x"]
           
            if isinstance(x, np.ndarray) and x.ndim == 0:
                x = np.array([x])
            print(x.shape)
            data.append(x)

       
        return np.array(data)

    def setup(self, stage=None):
        if self._X is None:
            self._X = self._load_npz_folder()  
            
        N = len(self._X)
        n_val = int(self.val_split * N)
        if n_val == 0:
            n_val = 1
        n_train = N - n_val
        

        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(N, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        print(len(val_idx))

       
        X_train = self._X[train_idx] 
        X_val = self._X[val_idx] 

        if self.preprocess is None:
            self.preprocess = self.transform(X_train)

        if stage in (None, "fit", "validate"):
            self.train_dataset = EEGdataset(X=X_train, y=X_train, preprocess=self.preprocess)
            self.val_dataset = EEGdataset(X=X_val, y=X_val, preprocess=self.preprocess)

        if stage in (None, "test"):
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
