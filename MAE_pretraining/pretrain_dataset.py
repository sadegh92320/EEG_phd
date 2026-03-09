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



def get_pretrain_dataset(datasetName, type):
    dataset = None
    if datasetName == "p300":
        dataset = PretrainDataset(dataset_name=datasetName, type=type,config="/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/p300.yaml",
                                  new_freq=200)

    if datasetName == "ssvep":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/ssvep.yaml",
                                  new_freq=200)

    if datasetName == "hgd":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/hgd.yaml",
                                  new_freq=200)

    if datasetName == "seed":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/seed.yaml",
                                  new_freq=200)

    if datasetName == "eeg_mi_bci":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/eeg_mi_bci.yaml",
                                  new_freq=200)

    if datasetName == "bci_comp_iv2a":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
                                  new_freq=200)

    if datasetName == "bci_comp_iv2b":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="paper_1_code/MAE_pretraining/info_dataset/bci_comp_2b.yaml",
                                  new_freq=200)

    if datasetName == "auditory":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="/MAE_pretraining/info_dataset/auditory.yaml",
                                  new_freq=200)

    if datasetName == "online":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/online_bci_cla.yaml",
                                  new_freq=200)

    if datasetName == "mi":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/LMI_C.yaml",
                                  new_freq=200)

    if datasetName == "mif":
        dataset = PretrainDataset(dataset_name=datasetName, type=type, config="MAE_pretraining/info_dataset/LMI_F.yaml",
                                  new_freq=200)
    if dataset == None:
        raise ValueError("Please enter a correct dataset name")
    
    return dataset
    


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
    def __init__(self, dataset_name, config_path, 
                 new_freq=None, resample=False, type = "train"):
        super().__init__()
        
        self.dataset_name = dataset_name
        
        self.resample = resample 
        self.new_freq = new_freq

        # Load Global Channel Config
        with open("/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/channel_info.yaml", "r") as file:
            self.channel_config = yaml.safe_load(file)

        # Load Dataset-Specific Config
        if config_path is None:
             raise ValueError("config_path must be provided")
             
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.old_freq = self.config["frequency"] 
        
        self.channel_list = self.config.get("channel_list", [])
        self.channel_id = self.get_chan_idx()
        datafolder = os.path.join(self.config["data_file"], type)
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
        for ch in self.channel_list:
         
            idx = self.channel_config["channel_mapping"].get(ch)
            if idx is None:
                raise ValueError(f"Channel '{ch}' not found in general channel_info.yaml mapping.")
            channel_id.append(idx)
        return channel_id

    def __getitem__(self, index):
        
        path = self.file_paths[index]
        npz = np.load(path)
        eeg = npz["x"]
        
       
        if self.resample and (self.new_freq != self.old_freq):
            eeg = resample_eeg(eeg=eeg, previous_freq=self.old_freq, new_freq=self.new_freq)
            
       
        eeg = standardize_channel(eeg)
       
        eeg = np.clip(eeg, -500, 500)
        
        return torch.from_numpy(eeg).float(), self.channel_id

    def __len__(self):
       
        return len(self.file_paths)