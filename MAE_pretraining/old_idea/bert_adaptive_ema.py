"""
Adaptive Riemannian BERT with EMA Geometric Graph as Reference + SPD Loss

Builds on bert_parallel_adaptive_riemann.py with two additions:

1. EMA Geometric Graph: A persistent 144×144 buffer that accumulates
   channel-channel covariance structure across the entire training corpus
   via exponential moving average. Used as the REFERENCE POINT for the
   adaptive log map: S - G_ema instead of S - I. The Riemannian bias
   measures per-sample deviation from population average.

2. SPD Spectral Loss: Auxiliary regularizer that penalizes drift in channel
   covariance structure through the encoder layers (anchored to pre-masking
   input covariance). Uses eigvalsh log-spectral distance. Applied with
   warmup to avoid interfering with early reconstruction learning.

Three levels of spatial information:
    - nn.Embedding: channel identity ("I am Cz")
    - Per-sample Riemannian bias: instance-level geometry relative to population
    - EMA graph: population-level geometry (tangent space reference)
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from MAE_pretraining.old_idea.data_lightning import EEGData
import lightning as L
from lightning.pytorch import Trainer
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
import math
import random
import os
import torch.nn.init as init
from MAE_pretraining.transformer_variants import (
    AdaptiveRiemannianParallelTransformer,
    EMAGeometricGraph,
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PatchEEG(nn.Module):
    """Module that segments the eeg signal in patches and align them with encoder dimension"""
    def __init__(self, embed_dim=768, patch_size=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(patch_size, embed_dim)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=patch_size)

    def patch_eeg(self, x):
        B, C, T = x.shape
        x = x.unsqueeze(2)
        x = self.unfold(x)
        x = rearrange(x, "b (c p) s -> b c p s", c=C, p=self.patch_size)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.patch_eeg(x)
        return self.fc(x)


class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim):
        super(ChannelPositionalEmbed, self).__init__()
        self.channel_transformation = nn.Embedding(144, embedding_dim)
        init.zeros_(self.channel_transformation.weight)

    def forward(self, channel_indices):
        return self.channel_transformation(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def get_class_token(self):
        return self.pe[0, 0, :]

    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        pe_embeddings = self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)
        return pe_embeddings


class AdaptiveRiemannEMABert(pl.LightningModule):
    """
    BERT-style masked pretraining with:
    - Adaptive Riemannian parallel transformer (approx log map)
    - EMA geometric graph (population-level channel connectivity)
    - SPD spectral loss (channel covariance preservation regularizer)
    """
    def __init__(self, config=None, num_channels=64,
                 max_embedding=2000, enc_dim=512, depth_e=8,
                 mask_prob=0.5, patch_size=16, norm_pix_loss=False,
                 ema_momentum=0.99, spd_alpha=0.01, spd_warmup_epochs=3,
                 ema_update_every=10, spd_loss_layers=None):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.num_channels = num_channels
        self.spd_alpha = spd_alpha
        self.spd_warmup_epochs = spd_warmup_epochs
        self.ema_update_every = ema_update_every
        self._step_counter = 0

        # Which encoder layers to compute SPD loss on.
        # Default: first, middle, last — covers the full depth with 3 measurements.
        if spd_loss_layers is None:
            spd_loss_layers = {0, depth_e // 2, depth_e - 1}
        self.spd_loss_layers = set(spd_loss_layers)

        # Adaptive Riemannian parallel transformer layers
        self.encoder = nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='approx'
            ) for _ in range(depth_e)
        ])

        # EMA geometric graph — population-level channel connectivity
        # num_heads=4 because spatial heads = total_heads // 2 = 8 // 2 = 4
        self.ema_graph = EMAGeometricGraph(
            total_channels=144, momentum=ema_momentum, num_heads=4
        )

        self.mask_prob = mask_prob
        self.patch_size = patch_size
        self.patch = PatchEEG(patch_size=patch_size, embed_dim=enc_dim)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, enc_dim)))
        self.fc = nn.Linear(enc_dim, patch_size)
        self.norm_enc = nn.LayerNorm(enc_dim)

        # Temporal and channel embeddings
        self.channel_embedding_e = ChannelPositionalEmbed(embedding_dim=enc_dim)
        self.temporal_embedding_e = TemporalPositionalEncoding(d_model=enc_dim, max_len=max_embedding)

        self.criterion = nn.MSELoss()
        self.class_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        self.apply(self._init_weights)
        w = self.patch.fc.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.channel_embedding_e.channel_transformation.weight, std=0.02)

    def mask_bert(self, x):
        device = x.device
        B, L, D = x.shape
        mask = torch.rand(B, L, device=device) < self.mask_prob
        mask_token = self.mask_token.expand_as(x)
        mask_float = mask.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_token * mask_float
        return x, mask

    def _compute_channel_covariance(self, x, C, N):
        """Compute channel covariance (B, C, C) from flattened token sequence."""
        x_re = rearrange(x, 'b (n c) d -> b n c d', c=C)
        x_pool = x_re.mean(dim=1)  # (B, C, D) — pool over time
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            x_pool_f32 = x_pool.float()
            S = torch.bmm(x_pool_f32, x_pool_f32.transpose(-1, -2)) / self.enc_dim
            S = S + 1e-5 * torch.eye(C, device=S.device, dtype=torch.float32).unsqueeze(0)
        return S

    def _spd_spectral_distance(self, S_l, log_eig_anchor):
        """Log-spectral distance between layer covariance and precomputed anchor."""
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            eig_l = torch.linalg.eigvalsh(S_l.float()).clamp(min=1e-7)
        return (torch.log(eig_l) - log_eig_anchor).pow(2).sum(dim=-1).sqrt().mean()

    def encoder_forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = (torch.tensor(channel_list, dtype=torch.long, device=device)
                        if not isinstance(channel_list, torch.Tensor)
                        else channel_list.to(device))

        # Patch the EEG: (B, C, T) → (B, N, C, D) → (B, N*C, D)
        x = self.patch(x)
        N = x.shape[1]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(B, L, -1)

        # Channel embeddings
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        chan_embedding = self.channel_embedding_e(chan_id)
        x += chan_embedding

        # Temporal embeddings
        seq_idx = torch.arange(0, N, device=device, dtype=torch.long)
        eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
        tp = self.temporal_embedding_e(eeg_seq_indices)
        x += tp

        # Determine current SPD alpha (warmup schedule)
        current_epoch = self.current_epoch if self.training else self.spd_warmup_epochs * 2
        if current_epoch < self.spd_warmup_epochs:
            alpha = 0.0
        else:
            ramp = min(1.0, (current_epoch - self.spd_warmup_epochs) / max(self.spd_warmup_epochs, 1))
            alpha = self.spd_alpha * ramp
        spd_active = alpha > 0

        # ── Compute anchor covariance S_0 BEFORE masking ──
        # Needed for both EMA update and SPD loss (when active)
        channel_idx = channel_list[0]  # (C,) global channel indices
        self._step_counter += 1

        if spd_active or (self.training and self._step_counter % self.ema_update_every == 0):
            S_0 = self._compute_channel_covariance(x, C, N)

            # EMA update (every K steps)
            if self.training and (self._step_counter % self.ema_update_every == 0):
                self.ema_graph.update(S_0, channel_idx)

            # Precompute anchor log-eigenvalues (only when SPD loss is active)
            if spd_active:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False), \
                         torch.amp.autocast('cpu', enabled=False):
                        log_eig_0 = torch.log(
                            torch.linalg.eigvalsh(S_0.float()).clamp(min=1e-7)
                        )

        # Get population-level EMA reference for tangent space projection
        ema_ref = self.ema_graph.get_ref(channel_idx)  # (C, C)

        # BERT-style masking
        x, mask = self.mask_bert(x)

        # ── Pass through encoder with EMA reference ──
        spd_losses = []
        for i, transformer in enumerate(self.encoder):
            x = transformer(x, C, channel_idx=channel_idx, ema_ref=ema_ref)
            # Only compute SPD loss at selected layers when alpha > 0
            if spd_active and i in self.spd_loss_layers:
                S_l = self._compute_channel_covariance(x, C, N)
                d = self._spd_spectral_distance(S_l, log_eig_0)
                spd_losses.append(d)

        spd_loss = torch.stack(spd_losses).mean() if spd_losses else torch.tensor(0.0, device=x.device)

        x = self.norm_enc(x)
        x = self.fc(x)

        return x, mask, spd_loss, alpha

    def forward(self, eeg, channel_list):
        pred, mask, spd_loss, alpha = self.encoder_forward(eeg, channel_list)
        target, pad = self.patchify_1d(eeg, self.patch_size)
        B, Seq, Ch, P = target.shape
        target = target.view(B, Seq * Ch, P)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, unbiased=False, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)

        total_loss = loss + alpha * spd_loss
        return total_loss, pred, mask, loss, spd_loss

    def patchify_1d(self, x, patch_size: int):
        B, C, T = x.shape
        pad = (patch_size - (T % patch_size)) % patch_size
        if pad > 0:
            x = F.pad(x, (0, pad))
            T = T + pad
        N = T // patch_size
        x = x.view(B, C, N, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x, pad

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1000.0
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                      gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=1.0,
                           gradient_clip_algorithm='norm')

    def training_step(self, batch, batch_idx):
        data, channel_list = batch
        total_loss, pred, mask, recon_loss, spd_loss = self(data, channel_list)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("reconstruction_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("spd_loss", spd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log per-layer Riemannian attention head scales (mean, std, and individual)
        for i, layer in enumerate(self.encoder):
            scales = layer.attn.riemannian_bias.head_scales.detach()
            self.log(f"head_scale_mean/layer_{i}", scales.mean(), on_step=False, on_epoch=True)
            self.log(f"head_scale_std/layer_{i}", scales.std(), on_step=False, on_epoch=True)
            for h in range(scales.numel()):
                self.log(f"head_scale/layer_{i}_head_{h}", scales[h], on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, channel_list = batch
        total_loss, pred, mask, mse, spd_loss = self(data, channel_list)

        rmse = torch.sqrt(mse + 1e-8)
        pred_std = pred.std()

        self.log_dict(
            {"total_loss": total_loss, "val_mse": mse, "val_rmse": rmse,
             "val_pred_std": pred_std, "spd_loss": spd_loss},
            prog_bar=True, on_step=False, on_epoch=True
        )


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)
