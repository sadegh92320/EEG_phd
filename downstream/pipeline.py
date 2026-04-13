import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from export_data.export_stew import StewImport
from dataset import EEGdataset
import torch.nn as nn
from downstream.training_model import TrainerDownstream
import yaml
from process_data.mne_constructor import MNEMethods
from collections import Counter
import os
import numpy as np
import optuna
from process_data.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from MAE_pretraining.old_idea.bert_riemaniann_loss import RiemannLossBert
from MAE_pretraining.bert_parallel_approx_riemann import ApproxAdaptiveRiemannBert
from MAE_pretraining.pretrain_gnn import GNNEncoderDecoder
from MAE_pretraining.old_idea.bert_parallel_adaptive_riemann import AdaptiveRiemannBert
from MAE_pretraining.pretraining import EncoderDecoder

import random
import torchvision
from torch.utils.data import random_split
import lightning as L
from lightning.pytorch import Trainer
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from downstream.downstream_model import Downstream
#from downstream.models.conv_model import SimpleEEGfrom
from pytorch_lightning.loggers import WandbLogger
import wandb
from MAE_pretraining.load_data import get_dataloader
from downstream.split_data_downstream import DownstreamDataLoader
from MAE_pretraining.old_idea.bert_adaptive_ema_only import AdaptiveRiemannEMABert
from MAE_pretraining.old_idea.bert_ema_graph import AdaptiveRiemannEMAGraphBert
from MAE_pretraining.old_idea.bert_riemann_gnn import AdaptiveRiemannGNNBert


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class Pipeline:
    """Experiment pipeline from pretraining to downstream task"""
    def __init__(self, dataimporter = None, dataset = None, trainer = None, config = None, preprocess = False, is_split = False, pretraining = True):
        self.dataimporter = dataimporter
        self.trainer = trainer 
        self.eeg_dataset = dataset
        self.config = config
        self.preprocess = preprocess
        self.is_split = is_split
        self.downtream_loader = None
        self.data = None
        self.label = None
        self.pretraining = pretraining
        self.model = None
        self.checkpoint_path = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
       
    #Change in the future be careful of different size samples 
    def load_data(self, data_path):
        """Load downstream data"""
        data_eeg = sorted(os.listdir(data_path))
        data = []
        label = []
        for data_file in data_eeg:
            file = os.path.join(data_path, data_file)
            if not data_file.lower().endswith(".npz"):
                continue
           
            npz = np.load(file)
            x = npz["x"]   
            y = npz["y"]   
            if self.is_split:
                x, y = self.split(x, y)  
            else:
                y = [y]
                x = [x]
            
            data.extend(x)
            label.extend(y)

        return np.array(data), np.array(label)


    # To change in the future try to make the model adaptable
    def load_fn(self, x):
        """Make eeg sample match temporal size of 1024"""
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

   #This is not used anymore keep it for archive only
    def import_data_pretraining(self):
        """Import dataloader for pretraining"""
        dataset = torchvision.datasets.DatasetFolder(root=self.config["pretrain_data"], loader=self.load_fn,  extensions=['.npz'])
        val_ratio = 0.1
        n = len(dataset)
        n_val = int(n * val_ratio)
        n_train = n - n_val

        g = torch.Generator().manual_seed(42)
        train_dataset, valid_dataset = random_split(dataset, [n_train, n_val], generator=g)

        g_train = torch.Generator().manual_seed(42)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=10, shuffle=True, generator=g_train, worker_init_fn=seed_worker)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, num_workers=10, shuffle=False, worker_init_fn=seed_worker)

        return train_loader, valid_loader
    

    def import_data_pretrain(self, use_global_norm=False, clamp_channels=False):
        """Import the validation and train dataloader"""
        train_loader, valid_loader = get_dataloader(self.config, use_global_norm=use_global_norm, clamp_channels=clamp_channels)
        return train_loader, valid_loader


    def load_encoder(self, pretrain=True, log_mode='pade',
                     use_corr_masking=True, resume_ckpt=None, use_global_norm=False,
                     clamp_channels=False):
        """
        Pretrain the MAE and return the checkpoint path for downstream loading.

        Contributions:
            1. Riemannian spatial attention bias (Padé [1,1] log map)
            2. Geometric Temporal Value Injection for temporal heads

        Ablation table:
            1. baseline (parallel attn, no Riemannian) → log_mode='baseline'
            2. approx (S-I)                            → log_mode='approx', use_corr_masking=False
            3. pade                                    → log_mode='pade',   use_corr_masking=False
            4. pade + correlation masking               → log_mode='pade',   use_corr_masking=True

        Args:
            pretrain:         if True, train from scratch; else load existing ckpt
            log_mode:         'pade', 'approx', or 'baseline'
            use_corr_masking: if True, use correlation-aware channel masking;
                              if False, use standard random BERT masking
            resume_ckpt:      path to checkpoint to resume from (None = from scratch)
            use_global_norm:  if True, use global normalization (preserves channel
                              variance ratios); if False, use z-standardization
            clamp_channels:   if True, clamp channels with variance > 10× median
                              (only applies when use_global_norm=True). Use if
                              training is unstable due to noisy electrodes.
        """
        assert log_mode in ('pade', 'approx', 'baseline'), \
            f"log_mode must be 'pade', 'approx', or 'baseline', got '{log_mode}'"

        CKPT_DIR = self.config["lighting_CKPT_DIR"]
        os.makedirs(CKPT_DIR, exist_ok=True)

        if pretrain:
            train_loader, valid_loader = self.import_data_pretrain(use_global_norm=use_global_norm, clamp_channels=clamp_channels)

            # ── Select model variant ──
            # All variants use the same parallel attention architecture.
            # The baseline freezes head_scales=0 and beta=0 so both
            # Riemannian bias and temporal value injection branches have
            # zero contribution — same architecture, same param count.
            if log_mode == 'baseline':
                print("[Ablation] Parallel attention with Riemannian + cross-channel mixing disabled")
                model = ApproxAdaptiveRiemannBert(use_corr_masking=use_corr_masking)
                for layer in model.encoder:
                    layer.attn.riemannian_bias.head_scales.requires_grad = False
                    layer.attn.riemannian_bias.head_scales.zero_()
                    layer.attn.beta.requires_grad = False
                    layer.attn.beta.zero_()
            else:
                masking_str = "corr-masking" if use_corr_masking else "random-masking"
                print(f"[Ablation] Riemannian log_mode='{log_mode}', {masking_str}")
                model = ApproxAdaptiveRiemannBert(use_corr_masking=use_corr_masking)
                # Override log_mode in every encoder layer if needed
                if log_mode == 'approx':
                    for layer in model.encoder:
                        layer.attn.riemannian_bias.adaptive_log.log_mode = 'approx'

            # ── Run name for wandb (easy to compare in dashboard) ──
            norm_tag = "gnorm" if use_global_norm else "zstd"
            if log_mode == 'baseline':
                run_name = f"baseline-vit-{norm_tag}"
            else:
                run_name = f"riemann-{log_mode}-{norm_tag}"
                if use_corr_masking:
                    run_name += "-corrmask"

            ckpt_callback = ModelCheckpoint(
                    dirpath=CKPT_DIR,
                    monitor="val_mse",
                    mode="min",
                    save_top_k=5,
                    save_last=True,
                    filename="epoch{epoch}-" + run_name + "-{val_mse:.4f}",
                )

            wandb.login(key="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")
            wandb_logger = WandbLogger(
                project="eeg_foundation_model",
                name=run_name,
                log_model="all"
            )
            wandb_logger.experiment.config.update({
                "enc_dim": 512,
                "dec_dim": 384,
                "depth_e": 8,
                "depth_d": 4,
                "mask_prob": 0.7,
                "patch_size": 16,
                "log_mode": log_mode,
                "use_corr_masking": use_corr_masking,
                "use_global_norm": use_global_norm,
                "clamp_channels": clamp_channels,
            })

            callbacks = [TQDMProgressBar(refresh_rate=20), ckpt_callback]

            trainer = Trainer(
                callbacks=callbacks,
                log_every_n_steps=5,
                logger=wandb_logger,
                max_epochs=40,
                precision="16-mixed",
                gradient_clip_val=1.0,
            )

            trainer.fit(model, train_dataloaders=train_loader,
                        val_dataloaders=valid_loader,
                        ckpt_path=resume_ckpt,
                        )

            self.checkpoint_path = ckpt_callback.best_model_path
            print(f"Best checkpoint: {self.checkpoint_path}")
        else:
            import glob
            candidates = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch*-baseline-*.ckpt")))
            last_ckpt = os.path.join(CKPT_DIR, "last.ckpt")
            if candidates:
                self.checkpoint_path = candidates[-1]
            elif os.path.exists(last_ckpt):
                self.checkpoint_path = last_ckpt
            else:
                print("WARNING: No checkpoint found. Downstream will use random weights.")
                self.checkpoint_path = None
    
    def import_data(self):
        """get the raw data, process them, save them and input them in the dataset"""
        if self.preprocess == True:
            self.dataimporter().remove_artifacts().partition_data().save_data()
            data, label = self.load_data(self.dataimporter.data_dir)
        else:
            path_data = os.path.join(self.config["output_data_path"], self.config["experiment_folder"])
            data, label = self.load_data(path_data)  
        self.data = data
        self.label = label           
        return self
    
    def load_downstream(self, pretrain=True):
        """Load the downstream model using the pretrained checkpoint."""
        self.load_encoder(pretrain=pretrain)
        print("done loading encoder")
        self.model = Downstream(
            checkpoint_path=self.checkpoint_path,
            enc_dim=512,       # must match EncoderDecoder default
            depth_e=8,         # must match EncoderDecoder default
            patch_size=16,     # must match EncoderDecoder default
            num_classes=self.config["num_classes"],
        )
        
    def get_data_downstream(self, evaluation_scheme):
        if evaluation_scheme == "population":
            data = self.downtream_loader.get_data_for_population()
        elif evaluation_scheme == "LOSO":
            data = self.downtream_loader.get_data_for_leave_one_participant_out()
        elif evaluation_scheme == "per_subject_transfer":
            data = self.downtream_loader.get_per_subject_transfer()
        else:
            raise ValueError(f"Unknown evaluation scheme: {evaluation_scheme}")
    
    def make_model(self):
            return Downstream(
                checkpoint_path=self.checkpoint_path,
                enc_dim=512,       # must match EncoderDecoder default
                depth_e=8,
                patch_size=16,
                num_classes=self.config["num_classes"],
            )
    
    def make_conv_model(self):
        pass

    def loop_over_model(self):
        """Go through all the created models to test their performance against our own one"""
        pass

    def get_random_baseline_performance(self, evaluation_scheme):
        """Get the performance of a random baseline for the given evaluation scheme"""
        self.load_downstream(pretrain=False)  # we don't need the pretrained encoder for a random baseline
        if evaluation_scheme == "population":
            data = self.downtream_loader.get_data_for_population()
            self.trainer = self.trainer("cnnmodule", self.make_model, "adam", torch.nn.CrossEntropyLoss(), batch_size = 32, config = self.config, data = self.data, label = self.label)
        elif evaluation_scheme == "LOSO":
            data = self.downtream_loader.get_data_for_leave_one_participant_out()
        elif evaluation_scheme == "per_subject_transfer":
            data = self.downtream_loader.get_per_subject_transfer()
        else:
            raise ValueError(f"Unknown evaluation scheme: {evaluation_scheme}")
        
        y_true = []
        y_pred = []
        for x, y in data:
            y_true.append(y)
            y_pred.append(random.randint(0, self.config["num_classes"]-1))
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Random baseline accuracy for {evaluation_scheme}: {accuracy:.4f}")
        return accuracy

    def get_model_performance(self, evaluation_scheme):
        """Train the model"""
        labels = []  # all targets in the dataset
        self.load_downstream()
        for y in self.label:
            labels.append(int(y))

        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels)          # [35, 28, 27] example
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_counts)

        print("class_counts:", class_counts)
        print("class_weights:", class_weights)
        self.trainer = self.trainer("cnnmodule", self.make_model, "adam", torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device)), batch_size = 32, config = self.config, data = self.data, label = self.label)
        self.trainer.train_whole_data()
        return self
    
    def split(self, data, label, segment_length=3, overlap=0, sampling_rate=250):
        """
        Split the eeg in several segment
        data:  shape (C, T) after transpose
        label: single label for this trial (scalar or 0-d/1-d)
        """
        # make sure it's (C, T)
        data = data.transpose() if data.shape[0] > data.shape[1] else data
        C, T = data.shape
        print(f"data shape (C, T): {data.shape}")

        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length  # segment length in samples

        if step <= 0:
            raise ValueError(
                f"step computed as {step}. Check segment_length={segment_length} and overlap={overlap}."
            )

        # handle short signals: if T < data_segment, we still create 1 segment
        if T <= data_segment:
            number_segment = 0
        else:
            number_segment = (T - data_segment) // step

        segments = []
        new_labels = []

        for i in range(number_segment + 1):
            start = i * step
            end = start + data_segment

            # safety in case of boundary issues
            if end > T:
                end = T
                start = max(0, end - data_segment)

            seg = data[:, start:end]    # shape (C, segment_length_in_samples)
            segments.append(seg)
            new_labels.append(label)    # same label for all segments of this trial

        print(f"Created {len(segments)} segments")
        return segments, new_labels

    


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)

    with open("MAE_pretraining/setting_pretraining.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config=config)
    pipeline.load_encoder()
    

