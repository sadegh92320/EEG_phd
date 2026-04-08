"""
Benchmark evaluation for deep learning baselines (trained from scratch).

Supported models:
    - EEGNet       (Compact CNN with depthwise/separable convolutions)
    - Conformer    (CNN + Transformer hybrid)
    - DeepConvNet  (Deep convolutional network)
    - CTNet        (CNN-Transformer EEG classifier)

Usage:
    python -m downstream.get_deep_learning_model_perf \
        --model eegnet \
        --dataset faced \
        --protocol population
"""

import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn

from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.training_model import TrainerDownstream, EarlyStopper
from downstream.get_benchmark_foundation_model import summarize_results


# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────
# Dataset-specific configs
# ────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "faced": {
        "num_classes": 9,
        "metric": "bacc",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/faced",
        "config_yaml": "downstream/info_dataset/faced.yaml",
        "num_channels": 32,
        "data_length": 2560,  # 10s * 256Hz
        "sampling_rate": 256,
    },
    "bci_comp_2a": {
        "num_classes": 4,
        "metric": "accuracy",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/bci_comp_2a",
        "config_yaml": "MAE_pretraining/info_dataset/bci_comp_2a.yaml",
        "num_channels": 22,
        "data_length": 1000,  # 4s * 250Hz
        "sampling_rate": 250,
    },
    "upper_limb": {
        "num_classes": 6,
        "metric": "acc1",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/upper_limb",
        "config_yaml": "downstream/info_dataset/upperlimb.yaml",
        "num_channels": 32,
        "data_length": 768,
        "sampling_rate": 128,
    },
    "mumtaz": {
        "num_classes": 2,
        "metric": "bacc",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/mumtaz",
        "config_yaml": "downstream/info_dataset/mumtaz.yaml",
        "num_channels": 19,
        "data_length": 1280,  # 5s * 256Hz
        "sampling_rate": 256,
    },
    "error": {
        "num_classes": 2,
        "metric": "bacc",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/error",
        "config_yaml": "downstream/info_dataset/error.yaml",
        "num_channels": 64,
        "data_length": 142,  # ~0.284s * 500Hz
        "sampling_rate": 500,
    },
    "physio_P300": {
        "num_classes": 2,
        "metric": "bacc",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/physio_P300",
        "config_yaml": "downstream/info_dataset/physio_P300.yaml",
        "num_channels": 64,
        "data_length": 539,  # ~2.1s * 256Hz
        "sampling_rate": 256,
    },
    "binocular": {
        "num_classes": 40,
        "metric": "accuracy",
        "model_path": "downstream/saved_models",
        "result_output": "downstream/results",
        "data_path": "downstream/data/binocular",
        "config_yaml": "downstream/info_dataset/binocular.yaml",
        "num_channels": 64,
        "data_length": 250,  # 1s * 250Hz (sliding window segments)
        "sampling_rate": 250,
    },
}


# ────────────────────────────────────────────────────────────────
# Model builders
# ────────────────────────────────────────────────────────────────

def build_eegnet(num_classes, num_channels, data_length, sampling_rate, **kwargs):
    """EEGNet: Compact CNN with depthwise and separable convolutions."""
    from downstream.models.deep_learning_model.eeg_net import EEGNet

    model = EEGNet(
        no_spatial_filters=2,
        no_channels=num_channels,
        no_temporal_filters=8,
        temporal_length_1=sampling_rate // 2,   # half of fs — captures 2Hz+ temporal features
        temporal_length_2=sampling_rate // 8,   # scaled proportionally
        window_length=data_length,
        num_class=num_classes,
    )
    return model


def build_conformer(num_classes, num_channels, data_length, sampling_rate, **kwargs):
    """Conformer: CNN + Transformer hybrid for EEG classification."""
    from downstream.models.deep_learning_model.conformer import Conformer

    model = Conformer(
        num_channel=num_channels,
        data_length=data_length,
        emb_size=40,
        depth=6,
        n_classes=num_classes,
    )
    return model


def build_deepconvnet(num_classes, num_channels, data_length, sampling_rate, **kwargs):
    """DeepConvNet: Deep convolutional network for EEG decoding."""
    from downstream.models.deep_learning_model.deepconvnet import DeepConvNet

    model = DeepConvNet(
        number_channel=num_channels,
        nb_classes=num_classes,
        data_length=data_length,
        sampling_rate=sampling_rate,
    )
    return model


def build_ctnet(num_classes, num_channels, data_length, sampling_rate, **kwargs):
    """CTNet: CNN-Transformer EEG classifier."""
    from downstream.models.deep_learning_model.ctnet import EEGTransformer

    model = EEGTransformer(
        number_class=num_classes,
        number_channel=num_channels,
        data_length=data_length,
        sampling_rate=sampling_rate,
    )
    return model


MODEL_BUILDERS = {
    "eegnet": build_eegnet,
    "conformer": build_conformer,
    "deepconvnet": build_deepconvnet,
    "ctnet": build_ctnet,
}


# ────────────────────────────────────────────────────────────────
# Evaluation protocols
# ────────────────────────────────────────────────────────────────

loss_fn = torch.nn.CrossEntropyLoss()


def run_population(model, model_name, loader, config):
    """Protocol 1: Population-level evaluation (within-subject 80/10/10)."""
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    trainer = TrainerDownstream(
        model_name=model_name,
        model=model,
        optimizer="adamw",
        loss_fn=loss_fn,
        batch_size=64,
        config=config,
        early_stopper=EarlyStopper,
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        training_mode="classic_nn",
    )
    trainer.run_population(name_project=f"{model_name}_population")


def run_per_subject(model, model_name, loader, config):
    """Protocol 2 & 3: Per-subject self + transfer evaluation."""
    all_metrics = []
    for pid in loader.participant_ids:
        train_sub, val_sub, test_sub = loader.per_subject(pid)
        transfer_test = loader.get_subject_transfer(pid)

        trainer = TrainerDownstream(
            model_name=model_name,
            model=model,
            optimizer="adamw",
            loss_fn=loss_fn,
            batch_size=64,
            config=config,
            early_stopper=EarlyStopper,
            training_mode="classic_nn",
        )
        metrics = trainer.run_per_subject(
            name_project=f"{model_name}",
            participant_number=pid,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
            test_data_pop=transfer_test,
        )
        all_metrics.append(metrics)

    summarize_results(
        all_metrics, loader.participant_ids, model_name,
        config.get("dataset_name", "unknown"), "per_subject",
        result_dir=config.get("result_output", "downstream/results"),
    )


def run_loso_zero_shot(model, model_name, loader, config):
    """Protocol 4: Leave-one-subject-out zero-shot evaluation."""
    all_metrics = []
    for pid in loader.participant_ids:
        train_pop, val_pop = loader.get_loso_train_dataset(pid)
        test_sub = loader.get_full_subject_dataset(pid)

        trainer = TrainerDownstream(
            model_name=model_name,
            model=model,
            optimizer="adamw",
            loss_fn=loss_fn,
            batch_size=64,
            config=config,
            early_stopper=EarlyStopper,
            training_mode="classic_nn",
        )
        metrics = trainer.run_LOSO(
            participant_number=pid,
            name_project=f"{model_name}",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            test_data_sub=test_sub,
        )
        all_metrics.append(metrics)

    summarize_results(
        all_metrics, loader.participant_ids, model_name,
        config.get("dataset_name", "unknown"), "loso",
        result_dir=config.get("result_output", "downstream/results"),
    )


def run_loso_fine_tune(model, model_name, loader, config):
    """Protocol 5: Leave-one-subject-out + per-subject fine-tuning."""
    all_metrics = []
    for pid in loader.participant_ids:
        train_pop, val_pop = loader.get_loso_train_dataset(pid)
        train_sub, val_sub, test_sub = loader.per_subject(pid)

        trainer = TrainerDownstream(
            model_name=model_name,
            model=model,
            optimizer="adamw",
            loss_fn=loss_fn,
            batch_size=64,
            config=config,
            early_stopper=EarlyStopper,
            training_mode="classic_nn",
        )
        metrics = trainer.run_LOSO_fine_tune(
            participant_number=pid,
            name_project=f"{model_name}",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
        )
        all_metrics.append(metrics)

    summarize_results(
        all_metrics, loader.participant_ids, model_name,
        config.get("dataset_name", "unknown"), "loso_ft",
        result_dir=config.get("result_output", "downstream/results"),
    )


def run_cross_subject(model, model_name, loader, config):
    """
    Cross-subject evaluation:
    70% subjects train, 10% val, 20% test — single split.
    """
    train_ds, val_ds, test_ds = loader.get_cross_subject_split(
        test_ratio=0.2, val_ratio=0.1, seed=42,
    )

    trainer = TrainerDownstream(
        model_name=model_name,
        model=model,
        optimizer="adamw",
        loss_fn=loss_fn,
        batch_size=64,
        config=config,
        early_stopper=EarlyStopper,
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        training_mode="classic_nn",
    )
    trainer.run_population(name_project=f"{model_name}_cross_subject")


PROTOCOL_RUNNERS = {
    "population": run_population,
    "per_subject": run_per_subject,
    "loso": run_loso_zero_shot,
    "loso_ft": run_loso_fine_tune,
    "cross_subject": run_cross_subject,
}


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deep learning baselines on downstream EEG tasks."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_BUILDERS.keys()),
        help="Which deep learning model to run.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Which downstream dataset to evaluate on.",
    )
    parser.add_argument(
        "--protocol", type=str, default="population",
        choices=list(PROTOCOL_RUNNERS.keys()),
        help="Evaluation protocol to run.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override default data path for the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Seed ──
    seed_everything(args.seed)

    # ── Dataset config ──
    ds_cfg = DATASET_CONFIGS[args.dataset].copy()
    if args.data_path is not None:
        ds_cfg["data_path"] = args.data_path

    config = {
        "num_classes": ds_cfg["num_classes"],
        "metric": ds_cfg["metric"],
        "model_path": ds_cfg["model_path"],
        "result_output": ds_cfg["result_output"],
        "dataset_name": args.dataset,
    }

    # ── Data loader (no per-model normalization for DL baselines) ──
    loader = DownstreamDataLoader(
        data_path=ds_cfg["data_path"],
        config=ds_cfg["config_yaml"],
        custom_dataset_class=Downstream_Dataset,
    )

    # ── Build model ──
    builder = MODEL_BUILDERS[args.model]
    model = builder(
        num_classes=ds_cfg["num_classes"],
        num_channels=ds_cfg["num_channels"],
        data_length=ds_cfg["data_length"],
        sampling_rate=ds_cfg["sampling_rate"],
    )

    print(f"\n{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Protocol: {args.protocol}")
    print(f"  Mode:     classic_nn (trained from scratch)")
    print(f"{'='*60}\n")

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {total_params - trainable_params:,}\n")

    # ── Run evaluation ──
    runner = PROTOCOL_RUNNERS[args.protocol]
    runner(model, args.model, loader, config)


if __name__ == "__main__":
    main()
