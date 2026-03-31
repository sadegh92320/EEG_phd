"""
Benchmark evaluation for external foundation models (pretrained baselines).

Supported models:
    - STEEGFormer  (ST-EEGFormer, MAE-based, encoder-only downstream)
    - LaBraM       (BEiT-v2 style, NeuralTransformer)
    - BIOT         (Linear-attention transformer, STFT-based)
    - CBraMod      (Criss-cross spatial-temporal transformer)
    - EEGPT        (JEPA/MAE encoder + reconstructor/predictor)

Usage:
    python -m downstream.get_benchmark_foundation_model \
        --model steegformer \
        --dataset faced \
        --checkpoint /path/to/weights.pth \
        --protocol population
"""

import argparse
import os
import random
import torch
import numpy as np
import yaml
from functools import partial
import torch.nn as nn

from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.training_model import TrainerDownstream, EarlyStopper
from downstream.downstream_model import (
    Downstream, DownstreamGNN, DownstreamRiemannLoss,
    DownstreamRiemannTransformerPara, DownstreamRiemannTransformerSeq,
    DownstreamRiemannEMA,
)

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
        "data_length": 2560,  # 30s * 256Hz (baseline rate, resampled per-model in dataset)
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
        "data_length": 1000,  # 4s * 250Hz (baseline rate, resampled per-model in dataset)
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
}


# ────────────────────────────────────────────────────────────────
# Model builders
# ────────────────────────────────────────────────────────────────

def buil_baseline(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    model = Downstream(checkpoint_path=checkpoint_path, num_classes=num_classes)
    return model

def build_gnn_embedding(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    model = DownstreamGNN(checkpoint_path=checkpoint_path, num_classes=num_classes)
    return model

def build_riemann_loss(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    model = DownstreamRiemannLoss(checkpoint_path=checkpoint_path, num_classes=num_classes)
    return model

def build_riemann_transformer_para(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """Adaptive Riemannian parallel transformer (approx log map, learned SPD reference)."""
    model = DownstreamRiemannTransformerPara(num_classes=num_classes, checkpoint_path=checkpoint_path)
    return model

def build_riemann_transformer_seq(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    model = DownstreamRiemannTransformerSeq(num_classes=num_classes, checkpoint_path=checkpoint_path)
    return model

def build_riemann_ema(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """Adaptive Riemannian parallel transformer + EMA population covariance as reference."""
    model = DownstreamRiemannEMA(num_classes=num_classes, checkpoint_path=checkpoint_path)
    return model

def build_steegformer(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """ST-EEGFormer: encoder-only from pretrained MAE."""
    from downstream.models.foundation_models.STEEGformer import steegformer_small_downstream
    training_mode = kwargs.get("training_mode", "linear_probe")

    model = steegformer_small_downstream(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        aggregation="class",
    )

    # Model is frozen by default inside steegformer_small_downstream.
    # For full fine-tuning, unfreeze everything.
    if training_mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    return model


def _resampled_length(data_length, base_sfreq, model_name):
    """
    Compute the data_length the model will actually see after
    Downstream_Dataset resamples from base_sfreq to the model's native rate.
    """
    from downstream.downstream_dataset import MODEL_PREPROCESS_CONFIG
    target_sfreq = MODEL_PREPROCESS_CONFIG.get(model_name, MODEL_PREPROCESS_CONFIG["default"])["sfreq"]
    if target_sfreq == base_sfreq:
        return data_length
    return int(round(data_length * target_sfreq / base_sfreq))


def build_labram(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """
    LaBraM: BEiT-v2 style neural transformer.
    forward(x, input_chans) where x is (B, N_electrodes, n_patches, patch_size)
    and input_chans is channel indices for positional embedding.
    Head is already a single Linear(200, num_classes) — proper linear probe.
    """
    from downstream.models.foundation_models.labram import NeuralTransformer

    base_sfreq = kwargs.get("base_sfreq", 256)
    training_mode = kwargs.get("training_mode", "linear_probe")
    data_length = _resampled_length(data_length, base_sfreq, "labram")

    # LaBraM-base config: 200-dim embed, 200-size patches, depth=12
    model = NeuralTransformer(
        EEG_size=data_length,
        patch_size=200,
        in_chans=1,  # raw EEG (uses TemporalConv)
        out_chans=8,
        num_classes=num_classes,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=1e-4,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    if checkpoint_path is not None:
        from downstream.models.foundation_models.labram import load_state_dict
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        load_state_dict(model, state_dict)
        print(f"  [LaBraM] Loaded pretrained weights from {checkpoint_path}")

    # Reset classification head for the target task
    model.reset_classifier(num_classes)

    if training_mode == "linear_probe":
        # Freeze encoder, only train head (already Linear(200, num_classes))
        for name, p in model.named_parameters():
            if "head" not in name:
                p.requires_grad = False
    # else: all params trainable for fine-tuning

    return model


def build_biot(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """
    BIOT: Linear-attention transformer with STFT-based patch embedding.
    forward(x) where x is (B, C, T).
    """
    from downstream.models.foundation_models.biot import BIOTClassifier, ClassificationHead

    training_mode = kwargs.get("training_mode", "linear_probe")

    model = BIOTClassifier(
        input_eeg_channel=num_channels,
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=num_classes,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Load encoder (biot) weights only
        biot_state = {}
        for k, v in state_dict.items():
            if k.startswith("biot."):
                biot_state[k[5:]] = v  # strip "biot." prefix
            elif not k.startswith("classifier") and not k.startswith("chan_conv"):
                biot_state[k] = v

        missing, unexpected = model.biot.load_state_dict(biot_state, strict=False)
        if missing:
            print(f"  [BIOT] Missing keys: {missing}")
        if unexpected:
            print(f"  [BIOT] Unexpected keys: {unexpected}")
        print(f"  [BIOT] Loaded pretrained weights from {checkpoint_path}")

    # Freeze encoder, only train classifier + channel conv
    for p in model.biot.parameters():
        p.requires_grad = False

    if training_mode == "linear_probe":
        # Replace paper's ELU+Linear head with standardized linear probe
        # (original head preserved in ClassificationHead for fine-tuning)
        model.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )
    # else: keep paper's ClassificationHead (ELU + Linear)

    if training_mode == "full":
        for p in model.biot.parameters():
            p.requires_grad = True

    return model


def build_cbramod(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """
    CBraMod: Criss-cross spatial-temporal transformer.
    forward(x) where x is (B, C, T) — internally windows into patches of 200.
    """
    from downstream.models.foundation_models.cbramod import CBraModClassifier

    base_sfreq = kwargs.get("base_sfreq", 256)
    training_mode = kwargs.get("training_mode", "linear_probe")
    data_length = _resampled_length(data_length, base_sfreq, "cbramod")

    # data_length must work with patch_size=200 (it unfolds with step=200)
    n_patches = data_length // 200
    model = CBraModClassifier(
        num_class=num_classes,
        num_channel=num_channels,
        data_length=n_patches * 200,
        pretrained_dir=checkpoint_path,
    )

    from einops.layers.torch import Rearrange

    # Freeze backbone by default (linear probe)
    for p in model.backbone.parameters():
        p.requires_grad = False

    if training_mode == "linear_probe":
        # avgpooling_patch_reps: pool over channels & patches → Linear(200, C)
        # Matches original repo's avgpooling_patch_reps classifier
        model.classifier = nn.Sequential(
            Rearrange('b c s d -> b d c s'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(200, num_classes),
        )
    elif training_mode == "full":
        # all_patch_reps: flatten → 3-layer MLP (paper's original head)
        # Matches original repo's all_patch_reps classifier
        flat_features = num_channels * n_patches * 200
        model.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(flat_features, n_patches * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(n_patches * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, num_classes),
        )
        for p in model.backbone.parameters():
            p.requires_grad = True

    return model


def build_eegpt(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """
    EEGPT: JEPA-based encoder with LitEEGPTModel wrapper.
    forward(x, chans_id) where x is (B, C, T).

    EEGPT's published linear probe is a 2-layer design:
        Linear(2048, 16) → flatten → Linear(16*data_length/64, num_class)
    Total ~34K params — close enough to the other models' single-linear probes.
    We keep the paper's probe structure for both modes since modifying it
    would require changing the forward() method.
    """
    import yaml
    from downstream.models.foundation_models.eegpt import LitEEGPTModel

    base_sfreq = kwargs.get("base_sfreq", 256)
    training_mode = kwargs.get("training_mode", "linear_probe")
    config_yaml = kwargs.get("config_yaml", None)
    data_length = _resampled_length(data_length, base_sfreq, "eegpt")

    # Read dataset's channel list from its YAML config
    dataset_channel_list = None
    if config_yaml is not None:
        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        dataset_channel_list = cfg.get("channel_list", None)

    model = LitEEGPTModel(
        chans_num=num_channels,
        num_class=num_classes,
        data_length=data_length,
        load_path=checkpoint_path if checkpoint_path else "../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
        dataset_channel_list=dataset_channel_list,
    )

    # Freeze encoder, only train linear probes + channel conv
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    # EEGPT's probe (Linear(2048,16) → Linear(~256, C)) is already small (~34K params).
    # Paper's forward() has probe logic baked in, so we keep it as-is for linear_probe.
    # For fine-tuning, unfreeze the encoder.
    if training_mode == "full":
        for p in model.target_encoder.parameters():
            p.requires_grad = True

    return model


# Registry
MODEL_BUILDERS = {
    "steegformer": build_steegformer,
    "labram": build_labram,
    "biot": build_biot,
    "cbramod": build_cbramod,
    "eegpt": build_eegpt,
    "baseline": buil_baseline,
    "encoder_gnn": build_gnn_embedding,
    "riemann_loss": build_riemann_loss,
    "riemann_para": build_riemann_transformer_para,
    "riemann_adaptive": build_riemann_transformer_para,  # alias — same model
    "riemann_ema": build_riemann_ema,
    "riemann_seq": build_riemann_transformer_seq,
}


# ────────────────────────────────────────────────────────────────
# Evaluation protocols
# ────────────────────────────────────────────────────────────────

loss_fn = torch.nn.CrossEntropyLoss()


def run_population(model, model_name, loader, config):
    """Protocol 1: Population-level evaluation."""
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
        training_mode=config.get("training_mode", "linear_probe"),
    )
    trainer.run_population(name_project=f"{model_name}_population")


def run_per_subject(model, model_name, loader, config):
    """Protocol 2 & 3: Per-subject self + transfer evaluation."""
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
            training_mode=config.get("training_mode", "linear_probe"),
        )
        trainer.run_per_subject(
            name_project=f"{model_name}",
            participant_number=pid,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
            test_data_pop=transfer_test,
        )


def run_loso_zero_shot(model, model_name, loader, config):
    """Protocol 4: Leave-one-subject-out zero-shot evaluation."""
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
            training_mode=config.get("training_mode", "linear_probe"),
        )
        trainer.run_LOSO(
            participant_number=pid,
            name_project=f"{model_name}",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            test_data_sub=test_sub,
        )


def run_loso_fine_tune(model, model_name, loader, config):
    """Protocol 5: Leave-one-subject-out + per-subject fine-tuning."""
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
            training_mode=config.get("training_mode", "linear_probe"),
        )
        trainer.run_LOSO_fine_tune(
            participant_number=pid,
            name_project=f"{model_name}",
            train_data_pop=train_pop,
            val_data_pop=val_pop,
            train_data_sub=train_sub,
            val_data_sub=val_sub,
            test_data_sub=test_sub,
        )


def run_cross_subject(model, model_name, loader, config):
    """
    Cross-subject zero-shot evaluation (ST-EEGFormer FACED protocol):
    80% subjects train, 20% subjects test — single split, NOT full LOSO loop.
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
        training_mode=config.get("training_mode", "linear_probe"),
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
        description="Evaluate external foundation model baselines on downstream EEG tasks."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_BUILDERS.keys()),
        help="Which foundation model baseline to run.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Which downstream dataset to evaluate on.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained weights (.pth / .ckpt). Required for most models.",
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
    parser.add_argument(
        "--training_mode", type=str, default="linear_probe",
        choices=["linear_probe", "full"],
        help="Training mode: 'linear_probe' uses standardized single-linear head, "
             "'full' uses each paper's original classifier head.",
    )

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
        "training_mode": args.training_mode,
    }

    # ── Data loader (with per-model normalization) ──
    loader = DownstreamDataLoader(
        data_path=ds_cfg["data_path"],
        config=ds_cfg["config_yaml"],
        custom_dataset_class=Downstream_Dataset,
        norm_mode=args.model,  # maps to MODEL_PREPROCESS_CONFIG in downstream_dataset.py
        base_sfreq=ds_cfg["sampling_rate"],  # baseline rate of stored data (250 for 2a, 256 for FACED)
    )

    # ── Build model ──
    builder = MODEL_BUILDERS[args.model]
    model = builder(
        num_classes=ds_cfg["num_classes"],
        checkpoint_path=args.checkpoint,
        num_channels=ds_cfg["num_channels"],
        data_length=ds_cfg["data_length"],
        base_sfreq=ds_cfg["sampling_rate"],
        training_mode=args.training_mode,
        config_yaml=ds_cfg["config_yaml"],
    )

    print(f"\n{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Protocol: {args.protocol}")
    print(f"  Checkpoint: {args.checkpoint or 'None (random init)'}")
    print(f"{'='*60}\n")

    # Count trainable vs frozen params
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
