import torch
import torchvision
import math
import random
import numpy as np
from torch.utils.data import random_split


def load_fn(x):
    npz = (np.load(x))
    x = npz["x"]
    x = torch.tensor(x)
    
    window_length = 4*256  
    data_length = x.shape[1]  

    # Calculate the maximum starting index for the windows
    max_start_index = data_length - window_length

    # Generate random indices
    if max_start_index>0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index+window_length]
    x = x.to(torch.float)
    return x

max_epochs = 200
max_lr = 5e-4
batch_size=32
devices=[0]



dataset = torchvision.datasets.DatasetFolder(root="MAE_pretraining/data_bis", loader=load_fn,  extensions=['.npz'])
val_ratio = 0.1
n = len(dataset)
n_val = int(n * val_ratio)
n_train = n - n_val

g = torch.Generator().manual_seed(42)
train_dataset, valid_dataset = random_split(dataset, [n_train, n_val], generator=g)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)