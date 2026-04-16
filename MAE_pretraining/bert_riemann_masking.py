"""
Riemannian Transformer with Correlation-Aware Channel Masking

Extends bert_parallel_approx_riemann.py with a geometry-informed masking
strategy: instead of random token masking, we compute the per-sample channel
correlation matrix from raw EEG and mask entire channels together with their
most correlated partners. This prevents the model from reconstructing masked
channels by simply interpolating from a correlated neighbor, forcing it to
learn deeper temporal and spectral representations.

Uses Padé [1,1] approximant for tangent-space projection.
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
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
from MAE_pretraining.transformer_variants import AdaptiveRiemannianParallelTransformer


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


channel_list = ["Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
channel_list_2 = ["Fp2","AF3","AF4","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz",]
CHANNEL_DICT = dict(zip(range(len(channel_list)), channel_list))


class PatchEEG(nn.Module):
    """Module that segments the eeg signal in patches and align them with encoder dimension"""
    def __init__(self, embed_dim = 768, patch_size = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(patch_size, embed_dim)
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=(1,patch_size), stride=patch_size)
    def patch_eeg(self, x):
        B, C, T = x.shape

        #Add one dimension B,C,1,T
        x = x.unsqueeze(2)
        #size is (B, C*16, seq)
        #Retrieves patch of size patch_size for every channel and stack them
        x = self.unfold(x)
        x = rearrange(x, "b (c p) s -> b c p s", c = C, p = self.patch_size)
        #Shape of B seq C p with p being patch size
        x = x.permute(0, 3, 1, 2)
        return x
    def forward(self, x):
        #Patch the eeg and project it to the encoder dimension
        x = self.patch_eeg(x)
        return self.fc(x)



class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim):
        super(ChannelPositionalEmbed, self).__init__()
        self.channel_transformation = nn.Embedding(144, embedding_dim)
        init.zeros_(self.channel_transformation.weight)
    def forward(self, channel_indices):
        channel_embeddings = self.channel_transformation(channel_indices)
        return channel_embeddings




class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position.float() * div_term)
        pe[0,:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def get_class_token(self):
        return self.pe[0,0,:]

    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        pe_embeddings = self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)
        return pe_embeddings


class ApproxAdaptiveRiemannBert(pl.LightningModule):
    """
    BERT-style masked pretraining with adaptive Riemannian parallel transformer.

    Uses first-order approximation log(M) ≈ M - I instead of eigendecomposition.
    This avoids the B*N eigendecompositions per layer that the full version requires,
    while the adaptive reference R ensures M stays close to I for accuracy.

    The only remaining eigendecomposition is for R^{-1/2} — a single C×C matrix
    per layer, independent of batch size and sequence length.
    """
    def __init__(self, config=None, num_channels=64,
                 max_embedding=2000, enc_dim=512, depth_e=8,
                 mask_prob=0.5, patch_size=8, norm_pix_loss=False,
                 value_bias_layers=4,
                 learn_mu_reference=True):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.num_channels = num_channels

        # Adaptive Riemannian parallel transformer layers
        # log_mode='pade' → Padé [1,1] approximant: log(S) ≈ 2(S-I)(I+S)^{-1}
        # C3: learnable tangent-space centering via per-layer μ^(l)
        self.encoder = nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='pade',
                use_value_bias=(i < value_bias_layers),
                learn_mu_reference=learn_mu_reference,
            ) for i in range(depth_e)
        ])

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
    
    def mask_corr_chan(self, x, corr, num_chan, seed_prob=0.2,
                       patch_mask_prob=0.15, min_channels_for_corr=8):
        """
        Correlation-aware channel masking with second-round patch masking.

        When C >= min_channels_for_corr:
            Round 1: Mask entire correlated channel groups
            Round 2: Random patch masking on surviving visible tokens
        When C < min_channels_for_corr (e.g. BCI 2b with 3 channels):
            Fall back to standard random patch masking (too few channels
            for meaningful correlation-based masking)

        Args:
            x: (B, L, D) embedded tokens where L = N * C
            corr: (B, C, C) per-sample channel correlation matrix
            num_chan: C
            seed_prob: fraction of channels to use as seeds (round 1)
            patch_mask_prob: probability of masking visible tokens (round 2)
            min_channels_for_corr: minimum C to use correlation masking

        Returns:
            x_masked: (B, L, D) with masked tokens replaced by mask_token
            mask: (B, L) boolean — True = masked (for loss computation)
        """
        B, L, D = x.shape
        N = L // num_chan
        device = x.device

        # ── Fallback: too few channels → standard random patch masking ──
        if num_chan < min_channels_for_corr:
            return self.mask_bert(x)

        # ══════════════════════════════════════════════════════════════
        # Round 1: Correlation-aware channel masking
        # ══════════════════════════════════════════════════════════════

        # Step 1: Pick random seed channels per sample
        num_seed = max(1, math.ceil(seed_prob * num_chan))
        shuffle_id = torch.argsort(torch.rand(B, num_chan, device=device), dim=-1)
        seed_channels = shuffle_id[:, :num_seed]  # (B, num_seed)

        # Step 2: For each seed channel, find its most correlated partner
        corr_no_diag = corr.clone()
        corr_no_diag.diagonal(dim1=-2, dim2=-1).zero_()
        best_match = corr_no_diag.argmax(dim=-1)  # (B, C)
        corr_partners = best_match.gather(dim=-1, index=seed_channels)  # (B, num_seed)

        # Step 3: Second round of correlation expansion —
        # each partner's best match also gets masked (partner's partner)
        partners_partners = best_match.gather(dim=-1, index=corr_partners)  # (B, num_seed)

        # Step 4: Union of seeds + partners + partners' partners → channel mask
        channel_mask = torch.zeros(B, num_chan, device=device, dtype=torch.bool)
        channel_mask.scatter_(dim=-1, index=seed_channels, value=True)
        channel_mask.scatter_(dim=-1, index=corr_partners, value=True)
        channel_mask.scatter_(dim=-1, index=partners_partners, value=True)

        # Step 5: Expand channel mask to all time steps → (B, L)
        token_mask = channel_mask.unsqueeze(1).expand(-1, N, -1).reshape(B, L)

        # Step 6: Apply channel mask
        mask_float = token_mask.unsqueeze(-1).float()  # (B, L, 1)
        mask_token = self.mask_token.expand_as(x)
        x = x * (1 - mask_float) + mask_token * mask_float

        # ══════════════════════════════════════════════════════════════
        # Round 2: Random patch masking on visible (surviving) tokens
        # ══════════════════════════════════════════════════════════════

        # Generate random noise; already-masked tokens get 1.0 (never re-masked)
        patch_noise = torch.rand(B, L, device=device)
        patch_noise[token_mask] = 1.0
        extra_mask = patch_noise < patch_mask_prob  # (B, L)

        # Apply second-round mask
        extra_float = extra_mask.unsqueeze(-1).float()
        x = x * (1 - extra_float) + mask_token * extra_float

        # Combined mask for loss: reconstruct everything that was masked in either round
        combined_mask = token_mask | extra_mask

        return x, combined_mask



    
    
    def encoder_forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device) if not isinstance(channel_list, torch.Tensor) else channel_list.to(device)

        # Compute per-sample channel correlation from raw EEG
        x_centered = x - x.mean(dim=-1, keepdim=True)       # (B, C, T)
        x_normed = x_centered / (x_centered.std(dim=-1, keepdim=True) + 1e-8)
        corr = torch.bmm(x_normed, x_normed.transpose(-1, -2)) / T  # (B, C, C)
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

        # BERT-style masking
        #x, mask = self.mask_bert(x)
        x, mask = self.mask_corr_chan(x, corr, C)

        # Extract channel indices for the adaptive Riemannian reference
        # Within a batch, all samples share the same channel set (same dataset)
        channel_idx = channel_list[0]  # (C,) global channel indices

        # Pass through adaptive Riemannian transformer layers (approximation mode)
        for transformer in self.encoder:
            x = transformer(x, C, channel_idx=channel_idx)

        x = self.norm_enc(x)
        x = self.fc(x)

        return x, mask

    def forward(self, eeg, channel_list):
        pred, mask = self.encoder_forward(eeg, channel_list)
        target, pad = self.patchify_1d(eeg, self.patch_size)
        B, Seq, Ch, P = target.shape
        target = target.view(B, Seq * Ch, P)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, unbiased=False, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, pred, mask

    def patchify_1d(self, x, patch_size: int):
        """Segment the target eeg signal in patch and pad if necessary"""
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

    def training_step(self, batch, batch_idx):
        data, channel_list = batch
        loss, pred, mask = self(data, channel_list)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log the learned head scales across layers for analysis
        for i, layer in enumerate(self.encoder):
            scales = layer.attn.riemannian_bias.head_scales.detach()
            self.log(f"head_scale_mean/layer_{i}", scales.mean(), on_step=False, on_epoch=True)
            self.log(f"head_scale_std/layer_{i}", scales.std(), on_step=False, on_epoch=True)

            # C3: Learnable tangent-space reference μ^(l) — should grow toward
            # the log-Euclidean Fréchet mean of layer-l's covariance distribution
            mu = layer.attn.riemannian_bias.mu_log
            if mu is not None:
                self.log(f"mu_frobenius/layer_{i}", mu.detach().norm(),
                         on_step=False, on_epoch=True)
                self.log(f"mu_max_abs/layer_{i}", mu.detach().abs().max(),
                         on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, channel_list = batch
        mse, pred, mask = self(data, channel_list)

        rmse = torch.sqrt(mse + 1e-8)
        pred_std = pred.std()

        self.log_dict(
            {"val_mse": mse, "val_rmse": rmse, "val_pred_std": pred_std},
            prog_bar=True, on_step=False, on_epoch=True
        )


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)
