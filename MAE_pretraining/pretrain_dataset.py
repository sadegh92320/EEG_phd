from torch.utils.data import Dataset
import torch
from MAE_pretraining.utils import resample_eeg, standardize_channel
import os
import yaml
import numpy as np
import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, BatchSampler
from pathlib import Path
import math



def get_pretrain_dataset(datasetName, type):
    dataset = None

    if datasetName == "im":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config= Path("MAE_pretraining/info_dataset/im_lab.yaml"),
                                  new_freq=200)

    if datasetName == "p300":
        dataset = PretrainDataset(dataset_name=datasetName, type=type,config=Path("MAE_pretraining/info_dataset/p300.yaml"),
                                  new_freq=200)

    if datasetName == "ssvep":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/ssvep.yaml"),
                                  new_freq=200)

    if datasetName == "hgd":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/hgd.yaml"),
                                  new_freq=200)

    if datasetName == "seed":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/seed.yaml"),
                                  new_freq=200)
        
    if datasetName == "seed2":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/seed2.yaml"),
                                  new_freq=200)

    if datasetName == "eeg_mi_bci":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/eeg_mi_bci.yaml"),
                                  new_freq=200)

    if datasetName == "bci_comp_iv2a":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config= Path("MAE_pretraining/info_dataset/bci_comp_2a.yaml"),
                                  new_freq=200)

    if datasetName == "bci_comp_iv2b":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config= Path("MAE_pretraining/info_dataset/bci_comp_2b.yaml"),
                                  new_freq=200)

    if datasetName == "auditory":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config= Path("MAE_pretraining/info_dataset/auditory.yaml"),
                                  new_freq=200)

    if datasetName == "online":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/online_bci_cla.yaml"),
                                  new_freq=200)

    if datasetName == "mi":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config=Path("MAE_pretraining/info_dataset/LMI_C.yaml"),
                                  new_freq=200)

    if datasetName == "mif":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/LMI_F.yaml",
                                  new_freq=200)
    if dataset == None:
        raise ValueError("Please enter a correct dataset name")
    
    return dataset
    

class SubInterleavedDistributedBatchSampler(Sampler):
    """
    For N sub-datasets, this sampler:
      1. Builds one DistributedSampler per sub-dataset
      2. Optionally samples only a random subset of local indices per epoch
      3. Groups them into batches
      4. Yields batches in round-robin order
      5. Maps local indices to global ConcatDataset indices via offsets

    You must call sampler.set_epoch(epoch) at the start of each epoch.
    """

    def __init__(
        self,
        datasets: list,
        batch_sizes: list,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        samples_per_epoch: list | None = None,
    ):
        assert len(datasets) == len(batch_sizes), "Every dataset needs a batch size."
        if samples_per_epoch is not None:
            assert len(samples_per_epoch) == len(datasets), "samples_per_epoch must match datasets length."

        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch

        self.epoch = 0

        lengths = [len(d) for d in self.datasets]
        self.cum_lengths = np.cumsum([0] + lengths).tolist()

        self.sub_samplers = []
        for ds in self.datasets:
            ds_sampler = DistributedSampler(
                ds,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=False,
            )
            self.sub_samplers.append(ds_sampler)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)

    def _make_batches_for_dataset(self, ds_idx: int):
        """
        Get this rank's local indices for one dataset, optionally subsample them,
        then chunk into batches.
        """
        local_indices = list(iter(self.sub_samplers[ds_idx]))

        if self.samples_per_epoch is not None:
            target_n = self.samples_per_epoch[ds_idx]

            # Never ask for more than the available local shard if sampling without replacement
            target_n = min(target_n, len(local_indices))

            if self.shuffle:
                g = torch.Generator()
                # Different subset each epoch, dataset, and rank
                g.manual_seed(self.seed + 1000 * self.epoch + 100 * ds_idx + self.rank)
                perm = torch.randperm(len(local_indices), generator=g).tolist()
                local_indices = [local_indices[i] for i in perm[:target_n]]
            else:
                local_indices = local_indices[:target_n]

        bs = self.batch_sizes[ds_idx]
        batches = []

        if self.drop_last:
            usable = (len(local_indices) // bs) * bs
            local_indices = local_indices[:usable]

        for start in range(0, len(local_indices), bs):
            batch = local_indices[start:start + bs]
            if len(batch) < bs and self.drop_last:
                continue
            if len(batch) > 0:
                batches.append(batch)

        return batches

    def __iter__(self):
        offsets = torch.tensor(self.cum_lengths, dtype=torch.int64, device="cpu")

        # Build per-dataset batches for this epoch
        all_batches = [self._make_batches_for_dataset(i) for i in range(len(self.datasets))]
        iters = [iter(batches) for batches in all_batches]
        finished = [False] * len(iters)

        while not all(finished):
            for idx, batch_it in enumerate(iters):
                if finished[idx]:
                    continue

                try:
                    local_idx_list = next(batch_it)
                except StopIteration:
                    finished[idx] = True
                    continue

                local_idx_tensor = torch.tensor(local_idx_list, dtype=torch.int64, device="cpu")
                global_idx_tensor = local_idx_tensor + offsets[idx]
                yield global_idx_tensor.tolist()

    def __len__(self):
        total_batches = 0

        for i, ds in enumerate(self.datasets):
            # Local shard size for this rank from DistributedSampler
            local_len = len(self.sub_samplers[i])

            if self.samples_per_epoch is not None:
                local_len = min(local_len, self.samples_per_epoch[i])

            bs = self.batch_sizes[i]

            if self.drop_last:
                total_batches += local_len // bs
            else:
                total_batches += math.ceil(local_len / bs)

        return total_batches

class InterleavedDistributedBatchSampler(Sampler):
    """
    For N sub-datasets (each with its own DistributedSampler),
    this BatchSampler will:

      1.  For each sub-dataset i:
            - Instantiate `DistributedSampler(dataset_i, ...)`
            - Wrap it in a `BatchSampler(...)` so that each child yields
              (list_of_local_indices_from_dataset_i) of length batch_size_i.
      2.  In __iter__(), it iterates “round-robin” over those N
          BatchSamplers, yielding one batch from ds0, then one from ds1, …,
          until all of them are exhausted.
      3.  Each yielded batch is mapped → global ConcatDataset index space
          via a simple offset.  That way, when DataLoader actually fetches
          `dataset[global_index]`, it ends up asking the right sub-dataset.

    You must call sampler.set_epoch(epoch) each time you start a new epoch.
    """
    def __init__(self,
                 datasets: list,
                 batch_sizes: list,
                 num_replicas: int,
                 rank: int,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 0):
        """
        Args:
          - datasets:    List[torch.utils.data.Dataset], e.g. [dataset_1, dataset_2, ...]
          - batch_sizes: Same length as `datasets`.  The batch size you want
                         for each sub-dataset.  (You can make them all the same,
                         or pick different per-dataset batch sizes if desired.)
          - num_replicas: world_size (for all DistributedSampler)
          - rank:        this process’s global rank (for all DistributedSampler)
          - shuffle:     whether each sub-sampler shuffles internally
          - drop_last:   whether to drop the last non-full batch in each sub-dataset
          - seed:        seed for shuffling in each DistributedSampler
        """
        assert len(datasets) == len(batch_sizes), "Every dataset needs a batch size."
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # 1.  Compute the cumulative sizes so we can map “local index → global index”
        lengths = [len(d) for d in self.datasets]
        self.cum_lengths = np.cumsum([0] + lengths).tolist()
        #    e.g. if lengths = [1000, 2000, 1500],
        #         cum_lengths = [0, 1000, 3000, 4500].
        #    Then a local index j in dataset_i  is at global index (cum_lengths[i] + j).

        # 2.  For each sub-dataset, build a DistributedSampler + BatchSampler
        self.sub_samplers = []
        self.sub_batch_samplers = []
        for i, ds in enumerate(self.datasets):
            ds_sampler = DistributedSampler(
                ds,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=self.seed
            )
            self.sub_samplers.append(ds_sampler)

            bs = self.batch_sizes[i]
            # Note: BatchSampler just groups each sampler’s output into lists of length bs
            batch_sampler = BatchSampler(
                sampler=ds_sampler,
                batch_size=bs,
                drop_last=self.drop_last
            )
            self.sub_batch_samplers.append(batch_sampler)

        # 3.  We'll hold all of the sub_bsamplers in a list, and keep pointers
        #     to their iterators so we can cycle through them.
        self._iters = [None] * len(self.sub_batch_samplers)
        self._finished = [False] * len(self.sub_batch_samplers)

    def set_epoch(self, epoch: int):
        """
        Must be called at the start of each epoch, so each DistributedSampler
        reseeds/shuffles itself.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)

    def __iter__(self):
        """
        Round‐robin over each sub‐BatchSampler, but use a Tensor addition
        instead of a Python list comprehension to compute global indices.
        """
        # Pre‐convert cumulative offsets to a CPU‐tensor once
        # (shape = number_of_subdatasets,)
        offsets = torch.tensor(self.cum_lengths, dtype=torch.int64, device="cpu")

        # (Re)create one iterator per sub‐BatchSampler
        self._iters = [iter(bs) for bs in self.sub_batch_samplers]
        self._finished = [False] * len(self._iters)

        # Loop until all sub‐iterators are exhausted
        while not all(self._finished):
            for idx, batch_it in enumerate(self._iters):
                if self._finished[idx]:
                    continue

                try:
                    # `local_idx_list` is a Python list of ints from this sub‐dataset
                    local_idx_list = next(batch_it)
                except StopIteration:
                    self._finished[idx] = True
                    continue

                # Convert that Python list into a small 1‐D tensor
                local_idx_tensor = torch.tensor(local_idx_list, dtype=torch.int64, device="cpu")
                # Add the precomputed offset for this sub‐dataset
                global_idx_tensor = local_idx_tensor + offsets[idx]
                # Convert back to a Python list for DataLoader
                global_idx_list = global_idx_tensor.tolist()

                yield global_idx_list
        # Once all are finished, we’re done for this epoch.

    def __len__(self):
        # Total number of mini-batches (across all sub-datasets) = sum of their lengths.
        return sum(len(bs) for bs in self.sub_batch_samplers)




class PretrainDataset(Dataset):
    def __init__(self, dataset_name, config, 
                 new_freq=None, resample=False, type = "train"):
        super().__init__()
        config_path = config
        
        self.dataset_name = dataset_name
        
        self.resample = resample 
        self.new_freq = new_freq

        # Load Global Channel Config
        with open(Path("MAE_pretraining/info_dataset/channel_info.yaml"), "r") as file:
            self.channel_config = yaml.safe_load(file)

        # Load Dataset-Specific Config
        if config_path is None:
             raise ValueError("config_path must be provided")
             
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.old_freq = self.config["frequency"] 
        
        self.channel_list = self.config.get("channel_list", [])
        self.channel_id = self.get_chan_idx()
        datafolder = Path(self.config["data_file"]) / type
        self.datafolder = datafolder
       

       
        self.file_paths = self._get_file_paths(datafolder)
        self.channel_num = len(self.channel_id)

    def _get_file_paths(self, folder_path):
        """Stores paths to files instead of loading them into RAM."""
        paths = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(".npz"):
                paths.append(os.path.join(folder_path, fname))
        return paths
    
    def get_chan_idx(self):
        channel_id = []
        channel_list = [ch.lower() for ch in self.channel_list]
        chan_map = {key.lower(): val for key,val in self.channel_config["channels_mapping"].items()}
        
        for ch in channel_list:
         
            idx = chan_map.get(ch)
            if idx is None:
                raise ValueError(f"Channel '{ch}' not found in general channel_info.yaml mapping.")
            channel_id.append(idx)
        return channel_id

    def __getitem__(self, index):
        
        path = self.file_paths[index]
        with np.load(path) as npz:
            eeg = npz["x"]
        
       
        if self.resample and (self.new_freq != self.old_freq):
            eeg = resample_eeg(eeg=eeg, previous_freq=self.old_freq, new_freq=self.new_freq)
            
       
        eeg = standardize_channel(eeg)
       
        eeg = np.clip(eeg, -500, 500)
        
        return torch.from_numpy(eeg).float(), torch.tensor(self.channel_id, dtype=torch.long)

    def __len__(self):
       
        return len(self.file_paths)