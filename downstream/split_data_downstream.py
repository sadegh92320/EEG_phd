import torch
import numpy as np
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import os
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


#Get the evaluation scheme for downstream tasks
#Get the number of participants
#Use the scheme to define the test set
#use the rest of the data for training and validation
#4 types of evaluation schemes: population, per-subject, subject-transfer, per-subject-transfer
#for population the output is a train and val loader with all the data
#for per-subject, the output is a list of train, val and test loader for each participant
#for subject-transfer, the output is a list of test loaders for each participant, where the test loader is trained on all the other participants
#for per-subject-transfer, the output is a list of train, val and test loader for each participant, where the test loader is
    
    


import os
import random
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class DownstreamDataLoader:
    def __init__(self, data_path, config, custom_dataset_class=None, norm_mode="default", base_sfreq=256, pre_split=False):
        random.seed(92)
        self.data_path = data_path
        self.config = config
        self.custom_dataset_class = custom_dataset_class
        self.norm_mode = norm_mode  # per-model normalization key (e.g. "steegformer", "biot")
        self.base_sfreq = base_sfreq  # baseline sampling rate of the stored data
        self.pre_split = pre_split  # if True, train/eval split is pre-defined (e.g. TUEV)

        self.train_data_path = os.path.join(self.data_path, "train.h5")
        self.val_data_path = os.path.join(self.data_path, "val.h5")

        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Missing file: {self.train_data_path}")
        if not os.path.exists(self.val_data_path):
            raise FileNotFoundError(f"Missing file: {self.val_data_path}")

        self.participant_ids = self.get_participant_ids()
        self.num_participants = len(self.participant_ids)

    def get_participant_ids(self):
        """Get sorted unique participant IDs from train.h5 and val.h5."""
        with h5py.File(self.train_data_path, "r") as f:
            train_participants = f["participant"][:]

        with h5py.File(self.val_data_path, "r") as f:
            val_participants = f["participant"][:]

        all_ids = np.unique(np.concatenate([train_participants, val_participants]))
        return sorted(all_ids.tolist())

    def _load_split_by_participant(self, h5_path, participant_number=None, include=True):
        """
        Load x, y from one HDF5 file.
        If participant_number is None -> load everything.
        If include=True -> keep only that participant.
        If include=False -> keep everyone except that participant.
        """
        with h5py.File(h5_path, "r") as f:
            participants = f["participant"][:]

            if participant_number is None:
                indices = np.arange(len(participants))
            else:
                if include:
                    indices = np.where(participants == participant_number)[0]
                else:
                    indices = np.where(participants != participant_number)[0]

            x = f["x"][indices]
            y = f["y"][indices]

        return x, y

    def get_data_for_population(self):
        """
        Population protocol:
        - pooled train from train.h5
        - validation split from pooled train
        - pooled test from val.h5

        If later you want paper-faithful per-subject population evaluation,
        you can additionally create one test dataset per subject.
        """
        x_train, y_train = self._load_split_by_participant(self.train_data_path)
        x_test, y_test = self._load_split_by_participant(self.val_data_path)

        full_train_dataset = self.make_dataset(x_train, y_train, custom_dataset_class=self.custom_dataset_class)

        full_idx = np.arange(len(full_train_dataset))
        train_idx, val_idx = train_test_split(
            full_idx,
            test_size=0.2,
            random_state=92,
            shuffle=True,
            stratify=y_train if (not np.issubdtype(np.array(y_train).dtype, np.floating) and len(np.unique(y_train)) > 1) else None,
        )

        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(full_train_dataset, val_idx)
        test_dataset = self.make_dataset(x_test, y_test, custom_dataset_class=self.custom_dataset_class)

        return train_dataset, val_dataset, test_dataset

    def get_full_subject_dataset(self, participant_number):
        """
        Full held-out subject dataset: both sessions (train.h5 + val.h5).
        """
        x_train, y_train = self._load_split_by_participant(
            self.train_data_path, participant_number=participant_number, include=True
        )
        x_val, y_val = self._load_split_by_participant(
            self.val_data_path, participant_number=participant_number, include=True
        )

        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)

        return self.make_dataset(x, y, custom_dataset_class=self.custom_dataset_class)

    def get_loso_train_dataset(self, participant_number):
        """
        LOSO training/validation:
        - train from train.h5 for all subjects except held-out one
        - val from val.h5 for all subjects except held-out one
        """
        x_train, y_train = self._load_split_by_participant(
            self.train_data_path, participant_number=participant_number, include=False
        )
        x_val, y_val = self._load_split_by_participant(
            self.val_data_path, participant_number=participant_number, include=False
        )

        train_dataset = self.make_dataset(x_train, y_train, custom_dataset_class=self.custom_dataset_class)
        val_dataset = self.make_dataset(x_val, y_val, custom_dataset_class=self.custom_dataset_class)

        return train_dataset, val_dataset

    def get_data_for_leave_one_participant_out(self):
        """
        Returns list of:
            (train_dataset, val_dataset, test_dataset)
        for each held-out participant.
        """
        loso_data = []

        for part_nb in self.participant_ids:
            test_dataset = self.get_full_subject_dataset(part_nb)
            train_dataset, val_dataset = self.get_loso_train_dataset(part_nb)
            loso_data.append((train_dataset, val_dataset, test_dataset))

        return loso_data

    def per_subject(self, participant_number):
        """
        Per-subject self protocol:
        - subject's train.h5 data -> split into train/val
        - subject's val.h5 data -> self test
        """
        x_train, y_train = self._load_split_by_participant(
            self.train_data_path, participant_number=participant_number, include=True
        )
        x_test, y_test = self._load_split_by_participant(
            self.val_data_path, participant_number=participant_number, include=True
        )

        full_train_dataset = self.make_dataset(x_train, y_train, custom_dataset_class=self.custom_dataset_class)

        full_idx = np.arange(len(full_train_dataset))
        train_idx, val_idx = train_test_split(
            full_idx,
            test_size=0.2,
            random_state=92,
            shuffle=True,
            stratify=y_train if (not np.issubdtype(np.array(y_train).dtype, np.floating) and len(np.unique(y_train)) > 1) else None,
        )

        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(full_train_dataset, val_idx)
        test_dataset = self.make_dataset(x_test, y_test, custom_dataset_class=self.custom_dataset_class)

        return train_dataset, val_dataset, test_dataset

    def get_subject_transfer(self, participant_number):
        """
        Per-subject transfer test set:
        all OTHER subjects from train.h5 + val.h5
        """
        x_train, y_train = self._load_split_by_participant(
            self.train_data_path, participant_number=participant_number, include=False
        )
        x_val, y_val = self._load_split_by_participant(
            self.val_data_path, participant_number=participant_number, include=False
        )

        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)

        return self.make_dataset(x, y, custom_dataset_class=self.custom_dataset_class)

    # ─────────────────────────────────────────────────────────────
    # Cross-subject split (ST-EEGFormer / CBraMod convention)
    # ─────────────────────────────────────────────────────────────

    def get_cross_subject_split(self, test_ratio=0.2, val_ratio=0.1, seed=42):
        """
        Cross-subject evaluation (ST-EEGFormer FACED protocol):
        - 80% of SUBJECTS → train (further split into train/val)
        - 20% of SUBJECTS → test
        - No subject appears in more than one split.

        This combines all data from train.h5 + val.h5 first
        (since the export splits trials, not subjects).

        Exception: if the dataset config has pre_split=True (e.g. TUEV),
        the train/eval split is pre-defined by the dataset authors.
        In that case we respect it: train.h5 → train+val, val.h5 → test.

        Returns: (train_dataset, val_dataset, test_dataset)
        """
        # ── Pre-split datasets (e.g. TUEV): respect the original split ──
        if self.pre_split:
            x_train_full, y_train_full = self._load_split_by_participant(self.train_data_path)
            x_test, y_test = self._load_split_by_participant(self.val_data_path)

            # Split train into train/val (by participant, not by trial)
            with h5py.File(self.train_data_path, "r") as f:
                train_parts = f["participant"][:]

            train_pids = np.unique(train_parts)
            rng = np.random.default_rng(seed)
            rng.shuffle(train_pids)
            n_val = max(1, int(round(len(train_pids) * val_ratio)))
            val_pids = set(train_pids[:n_val].tolist())
            actual_train_pids = set(train_pids[n_val:].tolist())

            train_idx = np.array([i for i, p in enumerate(train_parts) if p in actual_train_pids])
            val_idx = np.array([i for i, p in enumerate(train_parts) if p in val_pids])

            print(f"  Pre-split dataset: {len(actual_train_pids)} train, {len(val_pids)} val subjects (from train.h5)")
            with h5py.File(self.val_data_path, "r") as f:
                test_parts = f["participant"][:]
            n_test_subj = len(np.unique(test_parts))
            print(f"  Test: {n_test_subj} subjects (from val.h5)")
            print(f"  Samples: {len(train_idx)} train, {len(val_idx)} val, {len(x_test)} test")

            train_ds = self.make_dataset(x_train_full[train_idx], y_train_full[train_idx], custom_dataset_class=self.custom_dataset_class)
            val_ds = self.make_dataset(x_train_full[val_idx], y_train_full[val_idx], custom_dataset_class=self.custom_dataset_class)
            test_ds = self.make_dataset(x_test, y_test, custom_dataset_class=self.custom_dataset_class)

            return train_ds, val_ds, test_ds

        # ── Standard cross-subject: pool and reshuffle ──
        # Load ALL data (combine trial-level splits back together)
        x_all, y_all, part_all = self._load_all_data()

        all_pids = np.unique(part_all)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_pids)

        n_test = max(1, int(round(len(all_pids) * test_ratio)))
        n_val = max(1, int(round(len(all_pids) * val_ratio)))

        test_pids = set(all_pids[:n_test].tolist())
        val_pids = set(all_pids[n_test:n_test + n_val].tolist())
        train_pids = set(all_pids[n_test + n_val:].tolist())

        train_idx = np.array([i for i, p in enumerate(part_all) if p in train_pids])
        val_idx = np.array([i for i, p in enumerate(part_all) if p in val_pids])
        test_idx = np.array([i for i, p in enumerate(part_all) if p in test_pids])

        print(f"  Cross-subject split: {len(train_pids)} train, {len(val_pids)} val, {len(test_pids)} test subjects")
        print(f"  Samples: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

        train_ds = self.make_dataset(x_all[train_idx], y_all[train_idx], custom_dataset_class=self.custom_dataset_class)
        val_ds = self.make_dataset(x_all[val_idx], y_all[val_idx], custom_dataset_class=self.custom_dataset_class)
        test_ds = self.make_dataset(x_all[test_idx], y_all[test_idx], custom_dataset_class=self.custom_dataset_class)

        return train_ds, val_ds, test_ds

    def _load_all_data(self):
        """Load and merge all data from train.h5 + val.h5 into (x, y, participant) arrays."""
        xs, ys, ps = [], [], []
        for path in [self.train_data_path, self.val_data_path]:
            with h5py.File(path, "r") as f:
                xs.append(f["x"][:])
                ys.append(f["y"][:])
                ps.append(f["participant"][:])
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)

    def get_per_subject_transfer(self):
        """
        Returns list of:
            (train_dataset, val_dataset, self_test_dataset, transfer_test_dataset)
        """
        transfer = []

        for part_nb in self.participant_ids:
            
            train_dataset, val_dataset, self_test_dataset = self.per_subject(part_nb)
            transfer_test_dataset = self.get_subject_transfer(part_nb)

            transfer.append(
                (train_dataset, val_dataset, self_test_dataset, transfer_test_dataset)
            )

        return transfer

    def make_dataset(self, x, y, custom_dataset_class):
        """Convert x, y arrays into a dataset."""
        return custom_dataset_class(x, y, config_path=self.config, norm_mode=self.norm_mode, base_sfreq=self.base_sfreq)

    


    
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