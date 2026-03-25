import torch
import random
import os
import numpy as np
import lightning as L
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.training_model import TrainerDownstream, EarlyStopper
from downstream.models.deep_learning_model.eeg_net import EEGNet


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_config_FACED():
    config = {
            "num_classes": 9,
            "metric": "bacc",  # balanced accuracy (macro recall) — standard for FACED
            "model_path": "downstream/saved_models",
            "result_output": "downstream/results",
        }
    return config


# ── Config ──


# ── Model ──
def get_eeg_net():
    model = EEGNet(
        no_spatial_filters=2,
        no_channels=32,
        no_temporal_filters=8,
        temporal_length_1=128,       # half of fs (256/2) — captures 2Hz+ temporal features
        temporal_length_2=32,        # scaled up proportionally for 256 Hz
        window_length=7680,          # 30s * 256 Hz
        num_class=9,
    )

    return model

# ── Data ──

def get_loader(data_path):
    loader = DownstreamDataLoader(
        data_path=data_path,
        config="downstream/info_dataset/faced.yaml",
        custom_dataset_class=Downstream_Dataset,
    )
    return loader


loss_fn = torch.nn.CrossEntropyLoss()

# =============================================
# Protocol 1: Population
# =============================================

def get_pop_perf(loader, model, config):
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    trainer = TrainerDownstream(
        model_name="EEGNet",
        model=model,
        optimizer="adam",
        loss_fn=loss_fn,
        batch_size=64,
        config=config,
        early_stopper=EarlyStopper,
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        training_mode="full",
    )
    trainer.run_population(name_project="eegnet_faced")


# =============================================
# Protocol 2 & 3: Per-Subject Self + Transfer
# =============================================
def get_perf_per_subj(loader, model, config):
    for pid in loader.participant_ids:
        train_sub, val_sub, test_sub = loader.per_subject(pid)
        transfer_test = loader.get_subject_transfer(pid)

        trainer = TrainerDownstream(
            model_name="EEGNet", model=model, optimizer="adam",
            loss_fn=loss_fn, batch_size=64, config=config,
            early_stopper=EarlyStopper, training_mode="full",
        )
        trainer.run_per_subject(
            name_project="eegnet_faced",
            participant_number=pid,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,       
            test_data_pop=transfer_test,   
        )


# =============================================
# Protocol 4: LOO Zero-Shot (LOSO)
# =============================================
def get_LOSO_zero_shot_perf(loader, config, model):
    for pid in loader.participant_ids:
        train_pop, val_pop = loader.get_loso_train_dataset(pid)
        test_sub = loader.get_full_subject_dataset(pid)

        trainer = TrainerDownstream(
            model_name="EEGNet", model=model, optimizer="adam",
            loss_fn=loss_fn, batch_size=64, config=config,
            early_stopper=EarlyStopper, training_mode="full",
        )
        trainer.run_LOSO(
            participant_number=pid,
            name_project="eegnet_faced",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            test_data_sub=test_sub,
        )


# =============================================
# Protocol 5: LOO Fine-Tune
# =============================================
def get_LOO_fine_tune_perf(loader, model, config):
    for pid in loader.participant_ids:
        train_pop, val_pop = loader.get_loso_train_dataset(pid)
        train_sub, val_sub, test_sub = loader.per_subject(pid)

        trainer = TrainerDownstream(
            model_name="EEGNet", model=model, optimizer="adam",
            loss_fn=loss_fn, batch_size=64, config=config,
            early_stopper=EarlyStopper, training_mode="full",
        )
        trainer.run_LOSO_fine_tune(
            participant_number=pid,
            name_project="eegnet_faced",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
        )


# =============================================
# Protocol 6: LOO Drop
# =============================================
def get_perf_LOO_drop(loader, model, config):
    for pid in loader.participant_ids:
        train_pop, val_pop = loader.get_loso_train_dataset(pid)
        train_sub, val_sub, test_sub = loader.per_subject(pid)

        trainer = TrainerDownstream(
            model_name="EEGNet", model=model, optimizer="adam",
            loss_fn=loss_fn, batch_size=64, config=config,
            early_stopper=EarlyStopper, training_mode="full",
        )
        trainer.run_LOSO_drop(
            participant_number=pid,
            name_project="eegnet_faced",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
        )


# =============================================
# Cross-Subject (ST-EEGFormer FACED protocol)
# =============================================
def get_cross_subject_perf(loader, model, config):
    """80% subjects train, 20% subjects test — single split, not full LOSO."""
    train_ds, val_ds, test_ds = loader.get_cross_subject_split(
        test_ratio=0.2, val_ratio=0.1, seed=42,
    )

    trainer = TrainerDownstream(
        model_name="EEGNet", model=model, optimizer="adam",
        loss_fn=loss_fn, batch_size=64, config=config,
        early_stopper=EarlyStopper,
        train_data=train_ds, val_data=val_ds, test_data=test_ds,
        training_mode="full",
    )
    trainer.run_population(name_project="eegnet_faced_cross_subject")


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)

    config = get_config_FACED()
    model = get_eeg_net()
    data_path = "downstream/data/faced"
    loader = get_loader(data_path)
    get_cross_subject_perf(loader=loader, model=model, config=config)