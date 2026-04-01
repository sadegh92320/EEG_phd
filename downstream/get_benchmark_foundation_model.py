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
        aggregation="class",
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

    from einops.layers.torch import Rearrange, Reduce

    # Freeze backbone by default (linear probe)
    for p in model.backbone.parameters():
        p.requires_grad = False

    if training_mode == "linear_probe":
        # Mean-pool over channels AND patches → Linear(200, num_classes)
        # (b, C, S, 200) → mean over C and S → (b, 200) → Linear → (b, num_classes)
        # ~402 params for binary — minimal head that purely tests backbone quality.
        model.classifier = nn.Sequential(
            Reduce('b c s d -> b d', 'mean'),
            nn.Linear(200, num_classes),
        )
    elif training_mode == "full":
        # Pool over patches first to avoid parameter explosion on long/many-channel data.
        # (b, C, S, 200) → mean over S → (b, C, 200) → flatten → MLP
        in_features = num_channels * 200
        model.classifier = nn.Sequential(
            Reduce('b c s d -> b c d', 'mean'),
            Rearrange('b c d -> b (c d)'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 12)
        _partial_unfreeze(
            model, model.backbone.encoder.layers, len(model.backbone.encoder.layers),
            finetune_layers, head_modules=[model.classifier],
        )
    elif training_mode == "lora":
        # Frozen backbone + LoRA adapters + same LP head
        from downstream.lora import inject_lora
        model.classifier = nn.Sequential(
            Reduce('b c s d -> b d', 'mean'),
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

    if training_mode == "linear_probe":
        # Standardized single-linear probe (matches other models)
        model._use_standard_probe = True
        # Freeze the original 2-layer probe so only the standard one trains
        for p in model._original_probe1.parameters():
            p.requires_grad = False
        for p in model._original_probe2.parameters():
            p.requires_grad = False
    elif training_mode == "full":
        # Paper's original 2-layer probe + partially unfrozen encoder
        model._use_standard_probe = False
        finetune_layers = FIXED_HP["full"].get("finetune_layers", 12)
        _partial_unfreeze(
            model, model.target_encoder.blocks, len(model.target_encoder.blocks),
            finetune_layers,
            head_modules=[model._original_probe1, model._original_probe2],
        )
        # Freeze the standard probe so only the original one trains
        for p in model.probe_norm.parameters():
            p.requires_grad = False
        model.probe_linear.weight.requires_grad = False
        model.probe_linear.bias.requires_grad = False
    elif training_mode == "lora":
        # Frozen encoder + LoRA adapters + standard LP head
        model._use_standard_probe = True
        for p in model._original_probe1.parameters():
            p.requires_grad = False
        for p in model._original_probe2.parameters():
            p.requires_grad = False
        from downstream.lora import inject_lora
        inject_lora(model.target_encoder, rank=8, alpha=16.0)
        # Standard probe head stays trainable (probe_norm + probe_linear)

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


def summarize_results(all_metrics, participant_ids, model_name, dataset_name,
                      protocol_name, result_dir="downstream/results"):
    """
    Aggregate per-participant metrics, print a summary table, save a box plot,
    and log the summary to wandb.

    Args:
        all_metrics: list of dicts, one per participant (from run_LOSO / run_per_subject).
        participant_ids: list of participant identifiers (same order as all_metrics).
        model_name: str, e.g. "steegformer".
        dataset_name: str, e.g. "faced".
        protocol_name: str, e.g. "loso" or "per_subject".
        result_dir: str, directory to save outputs.
    """
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

    # ── Print per-participant table ──
    # Pick display metrics (subset that's most useful)
    display_keys = [k for k in ["bacc", "acc1", "f1_score", "precision", "recall", "kappa"]
                    if k in metric_arrays]

    header = f"\n{'='*70}\n  {protocol_name.upper()} Results — {model_name} on {dataset_name} ({n_participants} participants)\n{'='*70}"
    print(header)

    col_w = 10
    hdr_line = f"  {'Participant':<14}" + "".join(f"{k:>{col_w}}" for k in display_keys)
    print(hdr_line)
    print(f"  {'─' * (14 + col_w * len(display_keys))}")

    for i, pid in enumerate(participant_ids):
        row = f"  {str(pid):<14}"
        for k in display_keys:
            v = metric_arrays[k][i] if i < len(metric_arrays[k]) else float("nan")
            if k in ("bacc", "acc1"):
                row += f"{v * 100:>{col_w}.1f}%"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)

    print(f"  {'─' * (14 + col_w * len(display_keys))}")
    mean_row = f"  {'Mean±Std':<14}"
    for k in display_keys:
        arr = metric_arrays[k]
        if k in ("bacc", "acc1"):
            mean_row += f"{arr.mean() * 100:.1f}±{arr.std() * 100:.1f}%".rjust(col_w)
        else:
            mean_row += f"{arr.mean():.3f}±{arr.std():.3f}".rjust(col_w)
    print(mean_row)
    print(f"{'='*70}\n")

    # ── Box plot of balanced accuracy ──
    primary_metric = "bacc" if "bacc" in metric_arrays else ("acc1" if "acc1" in metric_arrays else None)
    if primary_metric is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        vals = metric_arrays[primary_metric] * 100  # percent

        bp = ax.boxplot(vals, vert=True, patch_artist=True, widths=0.5,
                        boxprops=dict(facecolor="#4C78A8", alpha=0.7),
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="#555"),
                        capprops=dict(color="#555"),
                        flierprops=dict(marker="o", markersize=5, markerfacecolor="#E45756", alpha=0.7))

        # Overlay individual points (jittered)
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.ones(len(vals)) + jitter, vals, color="#4C78A8", alpha=0.5,
                   s=25, zorder=3, edgecolors="white", linewidth=0.5)

        # Mean marker
        ax.scatter([1], [vals.mean()], marker="D", color="white", edgecolors="#333",
                   s=40, zorder=4, linewidth=0.8)

        metric_label = "Balanced Accuracy" if primary_metric == "bacc" else "Top-1 Accuracy"
        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_title(f"{model_name} — {dataset_name} — {protocol_name.upper()}\n"
                     f"Mean: {vals.mean():.1f}% ± {vals.std():.1f}%  (n={len(vals)})",
                     fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([model_name])
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(result_dir,
                                 f"{model_name}_{dataset_name}_{protocol_name}_boxplot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Box plot saved to: {plot_path}")

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
        # Log the box plot image if it exists
        if primary_metric is not None and os.path.exists(plot_path):
            wandb.log({"boxplot": wandb.Image(plot_path)})
        wandb.finish()
    except Exception as e:
        print(f"  [Warning] Could not log summary to wandb: {e}")

    return metric_arrays


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
