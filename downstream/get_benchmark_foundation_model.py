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
from downstream.training_model import TrainerDownstream, EarlyStopper, FIXED_HP
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
        "data_length": 7680,  # 30s * 256Hz (baseline rate, resampled per-model in dataset)
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
    """Adaptive Riemannian parallel transformer (Padé log map, learned SPD reference)."""
    log_mode = kwargs.get("log_mode", "pade")
    use_frechet = kwargs.get("use_frechet", False)
    model = DownstreamRiemannTransformerPara(
        num_classes=num_classes, checkpoint_path=checkpoint_path,
        log_mode=log_mode, use_frechet=use_frechet,
    )
    return model

def build_riemann_transformer_seq(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    model = DownstreamRiemannTransformerSeq(num_classes=num_classes, checkpoint_path=checkpoint_path)
    return model

def build_riemann_ema(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """Adaptive Riemannian parallel transformer + EMA population covariance as reference."""
    model = DownstreamRiemannEMA(num_classes=num_classes, checkpoint_path=checkpoint_path)
    return model

def _partial_unfreeze(model, block_container, num_total_blocks, finetune_layers, head_modules):
    """
    STEEGformer-style partial unfreezing: freeze everything, then unfreeze
    the last `finetune_layers` transformer blocks + head modules.

    Args:
        model: The full model.
        block_container: nn.ModuleList of transformer blocks.
        num_total_blocks: Total number of blocks.
        finetune_layers: How many blocks to unfreeze from the end.
        head_modules: List of nn.Module instances to always unfreeze (classifier head, norms).
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Unfreeze last N blocks
    start_idx = max(0, num_total_blocks - finetune_layers)
    for i in range(start_idx, num_total_blocks):
        for p in block_container[i].parameters():
            p.requires_grad = True

    # 3. Unfreeze head modules
    for mod in head_modules:
        for p in mod.parameters():
            p.requires_grad = True

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  [Partial unfreeze] Last {finetune_layers}/{num_total_blocks} blocks + head: "
          f"{trainable} trainable, {frozen} frozen params")


def build_steegformer(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """ST-EEGFormer: encoder-only from pretrained MAE."""
    from downstream.models.foundation_models.STEEGformer import steegformer_small_downstream
    training_mode = kwargs.get("training_mode", "linear_probe")

    model = steegformer_small_downstream(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        aggregation="mean",
    )

    # Model is frozen by default inside steegformer_small_downstream.
    if training_mode == "full":
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 8)
        _partial_unfreeze(
            model, model.blocks, num_total_blocks=len(model.blocks),
            finetune_layers=finetune_layers,
            head_modules=[model.head, model.norm],
        )
    elif training_mode == "lora":
        # Backbone stays frozen; inject LoRA into attention layers, unfreeze head
        from downstream.lora import inject_lora
        inject_lora(model, rank=8, alpha=16.0)
        for p in model.head.parameters():
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
    elif training_mode == "full":
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 12)
        head_mods = [model.head]
        if model.fc_norm is not None:
            head_mods.append(model.fc_norm)
        _partial_unfreeze(
            model, model.blocks, num_total_blocks=len(model.blocks),
            finetune_layers=finetune_layers,
            head_modules=head_mods,
        )
    elif training_mode == "lora":
        # Freeze everything, inject LoRA, unfreeze head
        for p in model.parameters():
            p.requires_grad = False
        from downstream.lora import inject_lora
        inject_lora(model, rank=8, alpha=16.0)
        for p in model.head.parameters():
            p.requires_grad = True

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
        n_channels=18,  # Must match chan_conv output (18) and pretrained checkpoint
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

    # Keep paper's original ClassificationHead (ELU + Linear) for both LP and FT.
    # chan_conv is always trainable (projects arbitrary channels → 18 BIOT channels).
    for p in model.chan_conv.parameters():
        p.requires_grad = True

    if training_mode == "full":
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 4)
        # BIOT uses linear_attention_transformer: blocks are in
        # model.biot.transformer.layers (SequentialSequence).layers (nn.ModuleList)
        block_container = model.biot.transformer.layers.layers
        _partial_unfreeze(
            model, block_container, len(block_container), finetune_layers,
            head_modules=[model.classifier, model.chan_conv],
        )
    elif training_mode == "lora":
        # Backbone frozen by default above; inject LoRA, unfreeze classifier
        from downstream.lora import inject_lora
        inject_lora(model.biot, rank=8, alpha=16.0)
        for p in model.classifier.parameters():
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

    # Freeze backbone by default (linear probe)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Paper's original feed_forward: flatten (b, C*S*200) → 3-layer MLP
    # This matches the original CBraModClassifier in the STEEGFormer repo exactly.
    flat_dim = num_channels * n_patches * 200  # = C * S * 200

    if training_mode == "linear_probe":
        # Paper uses same feed_forward MLP for both LP and FT
        model.classifier = nn.Sequential(
            nn.Linear(flat_dim, n_patches * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(n_patches * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, num_classes),
        )
    elif training_mode == "full":
        # Same MLP head as paper
        model.classifier = nn.Sequential(
            nn.Linear(flat_dim, n_patches * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(n_patches * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, num_classes),
        )
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 12)
        _partial_unfreeze(
            model, model.backbone.encoder.layers, len(model.backbone.encoder.layers),
            finetune_layers, head_modules=[model.classifier],
        )
    elif training_mode == "lora":
        from downstream.lora import inject_lora
        model.classifier = nn.Sequential(
            nn.Linear(flat_dim, n_patches * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(n_patches * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, num_classes),
        )
        inject_lora(model.backbone, rank=8, alpha=16.0)
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def build_eegpt(num_classes, checkpoint_path, num_channels, data_length, **kwargs):
    """
    EEGPT: JEPA-based encoder with LitEEGPTModel wrapper.
    forward(x, chans_id) where x is (B, C, T).

    EEGPT's encoder output is (B, N_temporal, 4, 512) — channels are already
    absorbed via summary tokens, so the probe is channel-independent.

    linear_probe mode: standardized single-linear head (mean-pool + Linear(2048, C))
    full mode: paper's original 2-layer probe (Linear(2048,16) → Linear(16*N, C))
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

    # Freeze encoder
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    # Always use paper's original 2-layer probe (LinearWithConstraint) for both LP & FT.
    # chan_conv is always trainable (projects arbitrary channels → EEGPT channels).
    model._use_standard_probe = False
    for p in model.chan_conv.parameters():
        p.requires_grad = True

    if training_mode == "linear_probe":
        # Encoder frozen (done above), train chan_conv + probe1 + probe2
        pass  # probe1/probe2 are already trainable by default
    elif training_mode == "full":
        # Partially unfreeze encoder + train probe
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 12)
        _partial_unfreeze(
            model, model.target_encoder.blocks, len(model.target_encoder.blocks),
            finetune_layers,
            head_modules=[model._original_probe1, model._original_probe2, model.chan_conv],
        )
    elif training_mode == "lora":
        from downstream.lora import inject_lora
        inject_lora(model.target_encoder, rank=8, alpha=16.0)
        # probe1/probe2 + chan_conv stay trainable

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
# Result aggregation helpers
# ────────────────────────────────────────────────────────────────

SCALAR_METRICS = ["accuracy", "recall", "precision", "f1_score", "roc_auc",
                  "kappa", "acc1", "acc2", "bacc"]

MODEL_COLORS = {
    "steegformer": "#4C78A8", "labram": "#F58518", "cbramod": "#E45756",
    "biot": "#72B7B2", "eegpt": "#54A24B", "eegnet": "#B07AA1",
    "conformer": "#FF9DA7", "deepconvnet": "#9D755D", "ctnet": "#BAB0AC",
    "riemann_adaptive": "#4E79A7", "riemann_para": "#F28E2B",
    "riemann_ema": "#76B7B2", "riemann_seq": "#59A14F",
    "riemann_loss": "#EDC948", "baseline": "#AF7AA1",
    "encoder_gnn": "#E15759",
}


def summarize_results(all_metrics, participant_ids, model_name, dataset_name,
                      protocol_name, result_dir="downstream/results"):
    """
    Aggregate per-participant metrics, print a summary table with ALL metrics,
    save per-model results to JSON, and generate a combined box plot with all
    models that have been evaluated so far.

    Args:
        all_metrics: list of dicts, one per participant (from run_LOSO / run_per_subject).
        participant_ids: list of participant identifiers (same order as all_metrics).
        model_name: str, e.g. "steegformer".
        dataset_name: str, e.g. "faced".
        protocol_name: str, e.g. "loso" or "per_subject".
        result_dir: str, directory to save outputs.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(result_dir, exist_ok=True)

    # ── Collect scalar metrics into arrays ──
    metric_arrays = {}
    for key in SCALAR_METRICS:
        vals = []
        for m in all_metrics:
            if key in m:
                v = m[key]
                vals.append(v.item() if hasattr(v, "item") else float(v))
        if vals:
            metric_arrays[key] = np.array(vals)

    n_participants = len(all_metrics)

    # ── Print per-participant table (display subset of metrics) ──
    display_keys = [k for k in ["bacc", "acc1", "f1_score", "precision", "recall", "kappa"]
                    if k in metric_arrays]

    header = (f"\n{'='*70}\n"
              f"  {protocol_name.upper()} Results — {model_name} on {dataset_name} "
              f"({n_participants} participants)\n{'='*70}")
    print(header)

    col_w = 10
    hdr_line = f"  {'Participant':<14}" + "".join(f"{k:>{col_w}}" for k in display_keys)
    print(hdr_line)
    print(f"  {'─' * (14 + col_w * len(display_keys))}")

    for i, pid in enumerate(participant_ids):
        row = f"  {str(pid):<14}"
        for k in display_keys:
            v = metric_arrays[k][i] if i < len(metric_arrays[k]) else float("nan")
            if k in ("bacc", "acc1", "accuracy"):
                row += f"{v * 100:>{col_w}.1f}%"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)

    print(f"  {'─' * (14 + col_w * len(display_keys))}")
    mean_row = f"  {'Mean±Std':<14}"
    for k in display_keys:
        arr = metric_arrays[k]
        if k in ("bacc", "acc1", "accuracy"):
            mean_row += f"{arr.mean() * 100:.1f}±{arr.std() * 100:.1f}%".rjust(col_w)
        else:
            mean_row += f"{arr.mean():.3f}±{arr.std():.3f}".rjust(col_w)
    print(mean_row)

    # ── Print ALL metrics summary ──
    print(f"\n  {'─'*50}")
    print(f"  Summary of ALL metrics (Mean ± Std):")
    print(f"  {'─'*50}")
    for k in SCALAR_METRICS:
        if k not in metric_arrays:
            continue
        arr = metric_arrays[k]
        if k in ("bacc", "acc1", "acc2", "accuracy"):
            print(f"    {k:<14} {arr.mean()*100:6.2f} ± {arr.std()*100:5.2f}%")
        else:
            print(f"    {k:<14} {arr.mean():6.4f} ± {arr.std():6.4f}")
    print(f"{'='*70}\n")

    # ── Save per-model JSON (for combined plotting) ──
    json_path = os.path.join(result_dir,
                             f"{model_name}_{dataset_name}_{protocol_name}.json")
    json_data = {
        "model": model_name,
        "dataset": dataset_name,
        "protocol": protocol_name,
        "n_participants": n_participants,
        "participant_ids": [int(p) if isinstance(p, (int, np.integer)) else str(p)
                           for p in participant_ids],
        "metrics": {k: v.tolist() for k, v in metric_arrays.items()},
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Results saved to: {json_path}")

    # ── Generate combined box plot (all models evaluated so far) ──
    _plot_combined_boxplot(dataset_name, protocol_name, result_dir)

    # ── Log summary to wandb ──
    try:
        import wandb
        summary = {}
        for k, arr in metric_arrays.items():
            summary[f"summary/{k}_mean"] = float(arr.mean())
            summary[f"summary/{k}_std"] = float(arr.std())
            summary[f"summary/{k}_median"] = float(np.median(arr))
        summary["summary/n_participants"] = n_participants
        summary["summary/protocol"] = protocol_name
        summary["summary/model"] = model_name
        summary["summary/dataset"] = dataset_name

        wandb.init(
            project=f"{protocol_name}_summary",
            name=f"{model_name}_{dataset_name}_{protocol_name}_summary",
            reinit=True,
            config=summary,
        )
        wandb.log(summary)

        plot_path = os.path.join(result_dir,
                                 f"{dataset_name}_{protocol_name}_all_models_boxplot.png")
        if os.path.exists(plot_path):
            wandb.log({"boxplot_all_models": wandb.Image(plot_path)})
        wandb.finish()
    except Exception as e:
        print(f"  [Warning] Could not log summary to wandb: {e}")

    return metric_arrays


def _plot_combined_boxplot(dataset_name, protocol_name, result_dir):
    """
    Read all saved JSON result files for a given dataset + protocol and
    draw a single box plot with every model side by side.
    """
    import json
    import glob as glob_mod
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = os.path.join(result_dir, f"*_{dataset_name}_{protocol_name}.json")
    json_files = sorted(glob_mod.glob(pattern))
    if not json_files:
        return

    # Load all model results
    model_data = {}  # model_name -> np.array of primary metric values
    primary_metric = None
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        mname = data["model"]
        metrics = data["metrics"]
        # Pick primary metric (prefer bacc, then acc1, then accuracy)
        for pm in ("bacc", "acc1", "accuracy"):
            if pm in metrics:
                primary_metric = pm
                model_data[mname] = np.array(metrics[pm])
                break

    if not model_data or primary_metric is None:
        return

    # Sort models: foundation models first, then DL baselines, alphabetical within each
    foundation = ["steegformer", "labram", "cbramod", "biot", "eegpt"]
    dl_baselines = ["eegnet", "conformer", "deepconvnet", "ctnet"]
    own_models = ["riemann_adaptive", "riemann_para", "riemann_ema",
                  "riemann_seq", "riemann_loss", "baseline", "encoder_gnn"]

    def sort_key(name):
        if name in foundation:
            return (0, foundation.index(name))
        if name in dl_baselines:
            return (1, dl_baselines.index(name))
        if name in own_models:
            return (2, own_models.index(name))
        return (3, 0)

    model_names = sorted(model_data.keys(), key=sort_key)
    n_models = len(model_names)

    # ── Draw combined box plot ──
    fig_w = max(6, n_models * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    positions = list(range(1, n_models + 1))
    all_vals = [model_data[m] * 100 for m in model_names]
    colors = [MODEL_COLORS.get(m, "#888888") for m in model_names]

    bp = ax.boxplot(all_vals, positions=positions, vert=True, patch_artist=True,
                    widths=0.55,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#555", linewidth=1),
                    capprops=dict(color="#555", linewidth=1),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="#E45756", alpha=0.6))

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(all_vals, colors)):
        # Color boxes
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.7)

        # Overlay individual points (jittered)
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=color, alpha=0.5, s=20, zorder=3,
                   edgecolors="white", linewidth=0.4)

        # Mean diamond
        ax.scatter([i + 1], [vals.mean()], marker="D", color="white",
                   edgecolors="#333", s=35, zorder=4, linewidth=0.8)

        # Mean±std label above whisker
        whisker_top = bp["caps"][i * 2 + 1].get_ydata()[0]
        ax.text(i + 1, whisker_top + 1.5, f"{vals.mean():.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")

    # Labels
    metric_label = {"bacc": "Balanced Accuracy", "acc1": "Top-1 Accuracy",
                    "accuracy": "Accuracy"}.get(primary_metric, primary_metric)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=11)
    ax.set_title(f"{dataset_name.upper()} — {protocol_name.upper()}\n"
                 f"{metric_label} per participant ({n_models} models)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_names],
                       fontsize=9, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.3, n_models + 0.7)
    fig.tight_layout()

    plot_path = os.path.join(result_dir,
                             f"{dataset_name}_{protocol_name}_all_models_boxplot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Combined box plot saved to: {plot_path}")


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
            training_mode=config.get("training_mode", "linear_probe"),
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
            training_mode=config.get("training_mode", "linear_probe"),
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
            training_mode=config.get("training_mode", "linear_probe"),
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
        choices=["linear_probe", "full", "lora"],
        help="Training mode: 'linear_probe' (frozen backbone + linear head), "
             "'full' (unfreeze everything), 'lora' (frozen backbone + low-rank adapters + head).",
    )
    parser.add_argument(
        "--log_mode", type=str, default="pade",
        choices=["pade", "approx", "baseline"],
        help="Riemannian log map mode for riemann_para/riemann_adaptive models. "
             "Use 'approx' for S-I ablation, 'pade' for full Padé (default).",
    )
    parser.add_argument(
        "--norm", type=str, default=None,
        choices=["global_mad", "z_standardize"],
        help="Override downstream normalization. Use 'global_mad' for MAD-pretrained "
             "checkpoints, 'z_standardize' for z-std-pretrained checkpoints. "
             "If not set, uses the default from MODEL_PREPROCESS_CONFIG for the model.",
    )
    parser.add_argument(
        "--use_frechet", action="store_true", default=False,
        help="Enable Fréchet whitening for riemann_para/riemann_adaptive models. "
             "The R_inv_sqrt buffer is loaded from the checkpoint automatically.",
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
        "dataset_name": args.dataset,
    }

    # ── Data loader (with per-model normalization) ──
    # If --norm is provided, override the normalization method for this model
    # while keeping the model's native sampling rate.
    norm_mode_key = args.model
    if args.norm is not None:
        from downstream.downstream_dataset import MODEL_PREPROCESS_CONFIG
        base_cfg = MODEL_PREPROCESS_CONFIG.get(args.model, MODEL_PREPROCESS_CONFIG["default"]).copy()
        base_cfg["norm"] = {"method": args.norm}
        override_key = f"_override_{args.model}"
        MODEL_PREPROCESS_CONFIG[override_key] = base_cfg
        norm_mode_key = override_key
        print(f"  [Norm override] {args.model} → {args.norm} (sfreq={base_cfg['sfreq']})")

    loader = DownstreamDataLoader(
        data_path=ds_cfg["data_path"],
        config=ds_cfg["config_yaml"],
        custom_dataset_class=Downstream_Dataset,
        norm_mode=norm_mode_key,
        base_sfreq=ds_cfg["sampling_rate"],
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
        log_mode=args.log_mode,
        use_frechet=args.use_frechet,
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
