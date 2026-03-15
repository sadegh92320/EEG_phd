import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
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
from MAE_pretraining.load_data import get_dataloader
from downstream.split_data_downstream import DownstreamDataLoader


class Pipeline:
    """Experiment pipeline from pretraining to downstream task"""
    def __init__(self, dataimporter = None, dataset = None, trainer = None, config = None, preprocess = False, is_split = False, pretraining = True):
        self.dataimporter = dataimporter
        self.trainer = trainer 
        self.eeg_dataset = dataset
        self.config = config
        self.preprocess = preprocess
        self.is_split = is_split
        self.downtream_loader = DownstreamDataLoader(data_path=self.config["output_data_path"], config=self.config)
        self.data = None
        self.label = None
        self.pretraining = pretraining
        self.encoder = None
        self.model = None
        self.temporal_embedding = None
        self.channel_embedding = None
        self.class_token = None
        self.patch = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
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


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=10, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, num_workers=10, shuffle=False)

        return train_loader, valid_loader
    

    def import_data_pretrain(self):
        """Import the validation and train dataloader"""
        train_loader, valid_loader = get_dataloader(self.config)
        return train_loader, valid_loader


    def load_encoder(self):
        """
        Load the encoder of the pretrained model and optionally pretrain the model if 
        not done before.
        """
        CKPT_DIR = self.config["lighting_CKPT_DIR"]
        os.makedirs(CKPT_DIR, exist_ok=True)
        CKPT_PATH = os.path.join(CKPT_DIR, "best_test.ckpt")

        if self.pretraining:
            train_loader, valid_loader = self.import_data_pretrain()
            model = EncoderDecoder()

            ckpt = ModelCheckpoint(
                    dirpath=CKPT_DIR,
                    monitor="val_mse",
                    mode="min",
                    save_top_k=5,
                    save_last=True,
                    filename="epoch{epoch}-val_mse{val_mse:.4f}",
                )

            wandb_logger = WandbLogger(
                project="eeg-foundation-model",
                name="mae-baseline-run1",
                log_model="all"
            )
            wandb_logger.experiment.config.update({
                "enc_dim": 512,
                "dec_dim": 384,
                "depth_e": 8,
                "depth_d": 4,
                "mask_prob": 0.75,
                "patch_size": 16
            })

            trainer = Trainer(
                callbacks=[TQDMProgressBar(refresh_rate=20), ckpt],
                log_every_n_steps=5,
                logger=wandb_logger,
                max_epochs=400,
                precision="16-mixed",
                gradient_clip_val=1.0,
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        # Load trained weights
        model = EncoderDecoder.load_from_checkpoint(CKPT_PATH)

        self.encoder = model.encoder
        self.temporal_embedding = model.temporal_embedding_e
        self.channel_embedding = model.channel_embedding_e
        self.class_token = model.class_token
        self.patch = model.patch.patch
    
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
    
    def load_downstream(self):
        """Load the downstream model using the encoder"""
        self.load_encoder()
        print("done loading encoder")
        self.model = Downstream(encoder=self.encoder, temporal_embedding=self.temporal_embedding, path_eeg=self.patch,\
                                channel_embedding=self.channel_embedding, class_token=self.class_token, \
                                 enc_dim=1024, num_classes=self.config["num_classes"])
        
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
                encoder=self.encoder,
                temporal_embedding=self.temporal_embedding,
                path_eeg=self.patch,
                channel_embedding=self.channel_embedding,
                class_token=self.class_token,
                enc_dim=768,
                num_classes=self.config["num_classes"]
            )
    
    def make_conv_model(self):
        pass

    def loop_over_model(self):
        """Go through all the created models to test their performance against our own one"""
        pass

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
    with open("MAE_pretraining\setting_pretraining.yaml") as f:
        config = yaml.safe_load(f)
    
    pipeline = Pipeline(config=config)
    pipeline.load_encoder()
    

