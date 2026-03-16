from MAE_pretraining.pretraining import EncoderDecoder

from downstream.split_data_downstream import DownstreamDataLoader
import torch
import os
from torch.utils.data import DataLoader
from downstream.training_model import TrainerDownstream
from downstream.downstream_model import Downstream
import yaml
from downstream.downstream_dataset import Downstream_Dataset
import wandb

def get_random_baseline_performance(evaluation_scheme):
        """Get the performance of a random baseline for the given evaluation scheme"""
        with open("setting.yaml", "r") as file:
            config = yaml.safe_load(file)

        loader = DownstreamDataLoader(data_path="/Users/sadeghemami/paper_1_code/downstream/data/bci_comp_2a", config="MAE_pretraining/info_dataset/bci_comp_2a.yaml", custom_dataset_class=Downstream_Dataset)
        encoder = EncoderDecoder()
        model = Downstream(
                encoder=encoder.encoder,
                temporal_embedding=encoder.temporal_embedding_e,
                path_eeg=encoder.patch,
                channel_embedding=encoder.channel_embedding_e,
                class_token=encoder.class_token,
                enc_dim=512,
                num_classes=4,
            )
        if evaluation_scheme == "population":
                train_data, val_data, test_data = loader.get_data_for_population()
                trainer = TrainerDownstream("randombaseline", model, "adam", torch.nn.CrossEntropyLoss(), batch_size = 32, config=config,train_data=train_data, val_data=val_data, test_data=test_data)
                trainer.run_population(name_project="random_baseline_performance")
        elif evaluation_scheme == "LOSO":
            pass
        elif evaluation_scheme == "per_subject_transfer":
            pass
        elif evaluation_scheme == "per_subject":
            pass
        else:
            raise ValueError(f"Invalid evaluation scheme: {evaluation_scheme}")
        
def get_baseline_MAE(evaluation_scheme):
        """Get the performance of a random baseline for the given evaluation scheme"""
        with open("setting.yaml", "r") as file:
            config = yaml.safe_load(file)

        loader = DownstreamDataLoader(data_path="/Users/sadeghemami/paper_1_code/downstream/data/bci_comp_2a", config="MAE_pretraining/info_dataset/bci_comp_2a.yaml", custom_dataset_class=UpperLimbDataset)
        encoder = EncoderDecoder()
        encoder.load_state_dict(torch.load("checkpoint/best.cpkt", map_location="cpu"))
        model = Downstream(
                encoder=encoder.encoder,
                temporal_embedding=encoder.temporal_embedding_e,
                path_eeg=encoder.patch,
                channel_embedding=encoder.channel_embedding_e,
                class_token=encoder.class_token,
                enc_dim=512,
                num_classes=4,
            )
        if evaluation_scheme == "population":
                train_data, val_data, test_data = loader.get_data_for_population()
                trainer = TrainerDownstream("randombaseline", model, "adam", torch.nn.CrossEntropyLoss(), batch_size = 32, config=config,train_data=train_data, val_data=val_data, test_data=test_data)
                trainer.run_population()
        elif evaluation_scheme == "LOSO":
            pass
        elif evaluation_scheme == "per_subject_transfer":
            pass
        elif evaluation_scheme == "per_subject":
            pass
        else:
            raise ValueError(f"Invalid evaluation scheme: {evaluation_scheme}")
           

if __name__ == "__main__":
    with open("setting.yaml", "r") as file:
        config = yaml.safe_load(file)

    get_random_baseline_performance(evaluation_scheme="population")
    