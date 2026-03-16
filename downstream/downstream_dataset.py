from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset


class UpperLimbDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        fold=0,
        config_path=None,
        classification_task="motorimagination",
        data_length=None,
        normalize=True,
    ):
        self.classification_task = classification_task
        self.data_length = data_length
        self.data = np.array(X)
        self.class_label = np.array(y)
        self.normalize = normalize
        self.fold = fold

        with open(Path("MAE_pretraining/info_dataset/channel_info_rred.yaml"), "r") as file:
            self.channel_config = yaml.safe_load(file)

        if config_path is None:
            raise ValueError("config_path must be provided")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.channel_list = self.config.get("channel_list", [])
        self.channel_id = self.get_chan_idx()
        self.channel_num = len(self.channel_id)

    def get_chan_idx(self):
        channel_id = []
        channel_list = [ch.lower() for ch in self.channel_list]
        chan_map = {
            key.lower(): val
            for key, val in self.channel_config["channels_mapping"].items()
        }

        for ch in channel_list:
            idx = chan_map.get(ch)
            if idx is None:
                raise ValueError(
                    f"Channel '{ch}' not found in general channel_info.yaml mapping."
                )
            channel_id.append(idx)
        return channel_id

    def __len__(self):
        return len(self.class_label)

    def __getitem__(self, idx):
        trial_data = self.data[idx]
        label = self.class_label[idx]

        if self.data_length is not None:
            trial_data = trial_data[:, :self.data_length]

        trial_data = np.clip(trial_data, -500, 500)

        if self.normalize:
            mean = np.mean(trial_data, axis=1, keepdims=True)
            std = np.std(trial_data, axis=1, keepdims=True) + 1e-6
            trial_data = (trial_data - mean) / std

        return (
            torch.from_numpy(trial_data).float(),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(self.channel_id, dtype=torch.long),
        )