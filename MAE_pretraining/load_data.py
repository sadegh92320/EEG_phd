import torch
from torch.utils.data import DataLoader, ConcatDataset
from MAE_pretraining.pretrain_dataset import get_pretrain_dataset, InterleavedDistributedBatchSampler, SubInterleavedDistributedBatchSampler
import torch.distributed as dist

def collate_fn(batch):
    """Load data with padding"""
    # batch is a list of tensors 'x' of shape (C, T)
    channel_list = [x.shape[0] for x in batch]
    c_max = max(channel_list)
    pad_x = torch.nn.utils.rnn.pad_sequence(
        sequences=batch, 
        batch_first=True, 
        padding_value=0.0
    )
    
    size_batch = len(batch)
    mask = torch.zeros((size_batch, c_max), dtype=torch.bool)
    for i, c_len in enumerate(channel_list):
        mask[i, :c_len] = True

    return pad_x, mask


def get_batch_size(num_channel):
    """
    More channels = smaller batch size
    to maintain a constant VRAM footprint.
    Tuned for A100 80GB with mixed precision + Riemannian covariance.
    """
    if num_channel >= 64:
        return 32
    elif num_channel >= 32:
        return 64
    else:
        return 128


def get_dataloader(config, use_global_norm=False, clamp_channels=False):
    dataset_to_use = config["data_use"]

    train_sets = [get_pretrain_dataset(dataset, type="train", use_global_norm=use_global_norm, clamp_channels=clamp_channels) for dataset in dataset_to_use]
    valid_sets = [get_pretrain_dataset(dataset, type="val", use_global_norm=use_global_norm, clamp_channels=clamp_channels) for dataset in dataset_to_use]
    

    # FIX 1 & 2: Actually call the function to calculate batch sizes
    batch_train = [get_batch_size(ds.channel_num) for ds in train_sets]
    batch_valid = [get_batch_size(ds.channel_num) for ds in valid_sets]

    concat_ds_train = ConcatDataset(train_sets)
    concat_ds_valid = ConcatDataset(valid_sets)

    # FIX 3: Safely determine DDP rank/world_size (defaults to 1 GPU if not initialized)
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0


    samples_per_epoch_train = [30000] * len(train_sets)
    samples_per_epoch_val = len(valid_sets)


    # FIX 4: Pass the LIST of datasets (train_sets), not concat_ds_train
    train_sampler = SubInterleavedDistributedBatchSampler(
        datasets=train_sets, 
        batch_sizes=batch_train,
        num_replicas=world_size, 
        rank=rank,
        shuffle=True, 
        drop_last=False,
        samples_per_epoch = samples_per_epoch_train

    )
    
    valid_sampler = SubInterleavedDistributedBatchSampler(
        datasets=valid_sets, 
        batch_sizes=batch_valid,
        num_replicas=world_size, 
        rank=rank,
        shuffle=False, 
        drop_last=False,
        samples_per_epoch=None
    )

    data_loader_train = DataLoader(
        concat_ds_train,
        batch_sampler=train_sampler,
        num_workers=10,          
        pin_memory=True,
        persistent_workers=True,
    )

    data_loader_valid = DataLoader(
        concat_ds_valid,
        batch_sampler=valid_sampler,
        num_workers=10,           
        pin_memory=True,
        persistent_workers=True,
    )    
    
    return data_loader_train, data_loader_valid