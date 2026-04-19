"""
Parallel Riemannian Transformer with Adaptive Log Map (First-Order Approximation)

Identical to bert_parallel_adaptive_riemann.py except:
- Uses log(M) ≈ M - I instead of full eigendecomposition
- Avoids all torch.linalg.eigh calls in the forward pass for the tangent
  space projection (only the reference R^{-1/2} still needs one eigh,
  but R is a single C×C matrix shared across the batch)
- The approximation becomes more accurate as the adaptive reference R
  converges to the data distribution, making M = R^{-1/2} S R^{-1/2}
  closer to I

Use this variant to:
1. Compare accuracy vs the full eigendecomposition version
2. Measure wall-clock speedup
3. If accuracy matches, use this as the flagship efficient version
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
from MAE_pretraining.transformer_variants import (
    AdaptiveRiemannianParallelTransformer,
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

    Uses first-order pade approximation instead of eigendecomposition.
    This avoids the B*N eigendecompositions per layer that the full version requires,
    while the adaptive reference R ensures M stays close to I for accuracy.

    """
    def __init__(self, config=None, num_channels=64,
                 max_embedding=2000, enc_dim=512, depth_e=8,
                 mask_prob=0.5, patch_size=16, norm_pix_loss=False,
                 use_corr_masking=True,
                 value_bias_layers=4,
                 learn_mu_reference=True,
                 use_luna_temporal=False, luna_num_slots=16,
                 luna_start_layer=2, luna_spd_beta_init=0.0,
                 use_rope=False, rope_freq_min=0.5, rope_freq_max=50.0,
                 rope_learnable=True,
                 spectral_loss_weight=0.0,
                 mask_strategy='random', mask_block_size=4):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.num_channels = num_channels
        self.use_corr_masking = use_corr_masking
        self.use_luna_temporal = use_luna_temporal
        self.use_rope = use_rope

        # Adaptive Riemannian parallel transformer layers
        # log_mode='pade' → Padé [1,1] approximant: log(S) ≈ 2(S-I)(I+S)^{-1}
        # C1: Riemannian spatial attention bias (score bias α_h · log(Σ))
        # C2: Luna temporal compression on layers >= luna_start_layer
        #     (skip early layers where local temporal structure matters most)
        self.encoder = nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='pade',
                use_value_bias=(i < value_bias_layers),
                learn_mu_reference=learn_mu_reference,
                use_luna_temporal=(use_luna_temporal and i >= luna_start_layer),
                luna_num_slots=luna_num_slots,
                luna_spd_beta_init=luna_spd_beta_init,
                use_rope=use_rope,
                rope_freq_min=rope_freq_min,
                rope_freq_max=rope_freq_max,
                rope_learnable=rope_learnable,
            ) for i in range(depth_e)
        ])

        self.mask_prob = mask_prob
        self.mask_strategy = mask_strategy
        self.mask_block_size = mask_block_size
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
        self.spectral_loss_weight = spectral_loss_weight
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
        """Standard random token masking (BERT-style)."""
        device = x.device
        B, L, D = x.shape
        mask = torch.rand(B, L, device=device) < self.mask_prob
        mask_token = self.mask_token.expand_as(x)
        mask_float = mask.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_token * mask_float
        return x, mask

    def mask_temporal_block(self, x, num_chan):
        """
        Block masking: mask contiguous temporal windows across ALL channels.

        Instead of random per-token masking (which allows local interpolation),
        this masks contiguous blocks of temporal patches. A masked block removes
        entire time windows, forcing the model to reconstruct from:
          - Distant temporal context (long-range temporal features)
          - Cross-channel spatial information (Riemannian spatial branch)

        Token layout: [t0_c0, t0_c1, ..., t0_cC, t1_c0, ..., tN_cC]
        A temporal block masks all channels at times [t, t+1, ..., t+block_size-1].

        Args:
            x:        (B, L, D) where L = N * C
            num_chan:  C — number of channels

        Returns:
            x:    (B, L, D) with blocked tokens replaced by mask_token
            mask: (B, L) boolean — True = masked
        """
        device = x.device
        B, L, D = x.shape
        C = num_chan
        N = L // C  # number of temporal positions
        block_size = self.mask_block_size

        # Number of blocks that fit in the temporal dimension
        n_blocks = N // block_size
        # If N isn't divisible, last few tokens handled separately
        n_remainder = N % block_size

        # Decide which blocks to mask: each block is masked with mask_prob
        # Shape: (B, n_blocks)
        block_mask = torch.rand(B, n_blocks, device=device) < self.mask_prob

        # Expand block mask to per-timestep mask: (B, N)
        time_mask = block_mask.unsqueeze(-1).expand(-1, -1, block_size).reshape(B, n_blocks * block_size)
        if n_remainder > 0:
            # Handle remainder: mask with same probability
            remainder_mask = torch.rand(B, 1, device=device) < self.mask_prob
            remainder_mask = remainder_mask.expand(-1, n_remainder)
            time_mask = torch.cat([time_mask, remainder_mask], dim=-1)  # (B, N)

        # Expand to all channels: when time t is masked, ALL channels at t are masked
        # Token layout: (B, N, C) → (B, N*C)
        token_mask = time_mask.unsqueeze(-1).expand(-1, -1, C).reshape(B, L)

        # Apply mask
        mask_token = self.mask_token.expand_as(x)
        mask_float = token_mask.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_token * mask_float

        return x, token_mask

    def mask_corr_channels(self, x, corr, num_chan, seed_prob=0.2,
                           patch_mask_prob=0.15, min_channels=8):
        """
        Correlation-aware channel masking with two-round strategy.

        Motivation: EEG electrodes have high spatial correlation. With random
        masking, the model can reconstruct a masked patch by copying from a
        neighboring electrode at the same timestep — a shortcut that produces
        low loss without learning deep temporal/spectral features.

        This method masks entire correlated channel groups, eliminating the
        shortcut. It uses the same covariance structure that feeds the
        Riemannian attention branch — a unified geometric framework.

        Round 1 — Channel masking:
            Pick random seed channels, mask them + their most correlated
            partner + the partner's most correlated partner (correlation chain).
            All time patches of masked channels are replaced with [MASK].

        Round 2 — Patch masking on survivors:
            Among the remaining visible tokens, randomly mask an additional
            patch_mask_prob fraction. This adds temporal reconstruction pressure
            so the model also learns within-channel temporal dynamics.

        Fallback:
            When num_chan < min_channels (e.g., BCI 2b with 3 channels),
            correlation masking is meaningless — falls back to standard
            random patch masking.

        Args:
            x:               (B, L, D) embedded tokens, L = N * C
            corr:            (B, C, C) per-sample Pearson correlation matrix
            num_chan:         C — number of EEG channels in this batch
            seed_prob:       fraction of C to use as random seeds
            patch_mask_prob: probability of masking visible tokens in round 2
            min_channels:    minimum C to use correlation masking

        Returns:
            x:    (B, L, D) with masked tokens replaced by mask_token
            mask: (B, L) boolean — True = masked (used by loss function)
        """
        B, L, D = x.shape
        N = L // num_chan
        device = x.device

        # ── Fallback for small channel counts ──
        if num_chan < min_channels:
            return self.mask_bert(x)

        # ══════════════════════════════════════════════════════════════
        # ROUND 1: Correlation-aware channel masking
        # ══════════════════════════════════════════════════════════════

        # 1a. Pick random seed channels per sample
        num_seed = max(1, math.ceil(seed_prob * num_chan))
        shuffle = torch.argsort(torch.rand(B, num_chan, device=device), dim=-1)
        seeds = shuffle[:, :num_seed]  # (B, num_seed)

        # 1b. Zero diagonal so a channel can't match itself
        corr_clean = corr.clone()
        corr_clean.diagonal(dim1=-2, dim2=-1).zero_()

        # 1c. For every channel, find its most correlated partner
        best_match = corr_clean.argmax(dim=-1)  # (B, C)

        # 1d. Build the correlation chain:
        #     seed → partner → partner's partner
        partners = best_match.gather(dim=-1, index=seeds)           # (B, num_seed)
        partners_partners = best_match.gather(dim=-1, index=partners)  # (B, num_seed)

        # 1e. Union all into a boolean channel mask
        chan_mask = torch.zeros(B, num_chan, device=device, dtype=torch.bool)
        chan_mask.scatter_(dim=-1, index=seeds, value=True)
        chan_mask.scatter_(dim=-1, index=partners, value=True)
        chan_mask.scatter_(dim=-1, index=partners_partners, value=True)

        # 1f. Expand channel mask to all time patches
        #     Token layout: [t0_c0, t0_c1, …, t0_cC, t1_c0, …]
        token_mask = chan_mask.unsqueeze(1).expand(-1, N, -1).reshape(B, L)

        # 1g. Apply round-1 mask
        mask_float = token_mask.unsqueeze(-1).float()
        mask_tok = self.mask_token.expand_as(x)
        x = x * (1 - mask_float) + mask_tok * mask_float

        # ══════════════════════════════════════════════════════════════
        # ROUND 2: Random patch masking on visible (surviving) tokens
        # ══════════════════════════════════════════════════════════════

        noise = torch.rand(B, L, device=device)
        noise[token_mask] = 1.0                    # already masked → never re-mask
        extra_mask = noise < patch_mask_prob        # (B, L)

        extra_float = extra_mask.unsqueeze(-1).float()
        x = x * (1 - extra_float) + mask_tok * extra_float

        # Combined mask for loss: union of both rounds
        combined_mask = token_mask | extra_mask

        return x, combined_mask

    def mask_block_corr(self, x, corr, num_chan, seed_prob=0.2):
        """
        Combined block + channel masking — eliminates BOTH shortcut axes.

        Design rationale (co-designed with EEG-RoPE):
            Why corr masking failed before (C1 only): the temporal branch was
            useless (no PE → uniform attention), so the model had only ONE
            functional branch to handle reconstruction. Not enough signal.

            With EEG-RoPE the temporal branch becomes functional. Now when
            correlated channels are masked, the temporal branch compensates
            via long-range context. And when temporal blocks are masked,
            the spatial branch compensates via Riemannian cross-channel
            structure. The two branches form a complementary pair.

            Double-masked tokens (both axes masked) are reconstructed via
            a two-hop path through the shared residual stream: layer 1
            enriches singly-masked intermediaries with cross-branch info,
            layer 2+ reads that enriched context. The shared QKV is the
            communication channel that makes this possible.

        Token regions and reconstruction pathways:
            Easy       (1-X)(1-Y): both branches have visible context
            Block-only    X(1-Y): temporal branch → long-range RoPE attention
            Chan-only  (1-X)Y   : spatial branch → Riemannian cross-channel
            Cross      X·Y      : two-hop via shared residual (layer 2+)

            Target: keep cross-region < 10% of tokens to avoid dominating
            the loss with the hardest reconstruction targets.

        Axis 1 (temporal): Block masking removes contiguous temporal windows.
            Block size = 4 patches (0.5s at 128Hz/16) — chosen to exceed
            the EEG-RoPE learned period (~0.25-0.33s at 3-4Hz).

        Axis 2 (spatial): Channel masking removes entire channels.
            For C >= 8: correlation chain (seed → partner → partner's partner)
            For C < 8: mask exactly 1 random channel per sample.
            This preserves the spatial learning signal (remaining channels
            provide cross-channel covariance for Riemannian bias) while
            forcing the model to actually learn that structure.

        Args:
            x:         (B, L, D) where L = N * C
            corr:      (B, C, C) per-sample correlation matrix
            num_chan:   C
            seed_prob: fraction of C to seed for correlation chains (C >= 8)

        Returns:
            x:    (B, L, D) with masked tokens
            mask: (B, L) boolean — True = masked
        """
        device = x.device
        B, L, D = x.shape
        C = num_chan
        N = L // C
        block_size = self.mask_block_size

        # ══════════════════════════════════════════════════════════════
        # ROUND 1: Temporal block masking
        # Target ~25-30% of temporal positions → keeps cross-region small.
        # With mask_prob=0.5: block_prob = 0.25-0.30
        # With mask_prob=0.75: block_prob = 0.375 (capped at 0.5 for safety)
        # ══════════════════════════════════════════════════════════════
        block_prob = min(self.mask_prob * 0.5, 0.5)

        n_blocks = N // block_size
        n_remainder = N % block_size

        block_mask = torch.rand(B, n_blocks, device=device) < block_prob
        time_mask = block_mask.unsqueeze(-1).expand(-1, -1, block_size).reshape(B, n_blocks * block_size)
        if n_remainder > 0:
            remainder_mask = torch.rand(B, 1, device=device) < block_prob
            remainder_mask = remainder_mask.expand(-1, n_remainder)
            time_mask = torch.cat([time_mask, remainder_mask], dim=-1)

        # Expand to all channels: (B, N) → (B, N*C) = (B, L)
        token_mask_r1 = time_mask.unsqueeze(-1).expand(-1, -1, C).reshape(B, L)

        # Apply round 1
        mask_tok = self.mask_token.expand_as(x)
        mask_float = token_mask_r1.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_tok * mask_float

        # ══════════════════════════════════════════════════════════════
        # ROUND 2: Channel masking — strategy depends on channel count
        # ══════════════════════════════════════════════════════════════
        if C >= 8:
            # Many channels: use correlation chain (seed → partner → partner's partner)
            # This masks ~40-50% of channels — correlated groups together.
            num_seed = max(1, math.ceil(seed_prob * C))
            shuffle = torch.argsort(torch.rand(B, C, device=device), dim=-1)
            seeds = shuffle[:, :num_seed]

            corr_clean = corr.clone()
            corr_clean.diagonal(dim1=-2, dim2=-1).zero_()
            best_match = corr_clean.argmax(dim=-1)

            partners = best_match.gather(dim=-1, index=seeds)
            partners_partners = best_match.gather(dim=-1, index=partners)

            chan_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
            chan_mask.scatter_(dim=-1, index=seeds, value=True)
            chan_mask.scatter_(dim=-1, index=partners, value=True)
            chan_mask.scatter_(dim=-1, index=partners_partners, value=True)
        else:
            # Few channels (C=3 for BCI 2b: C3, Cz, C4):
            # Mask exactly 1 random channel per sample.
            # Why not random patches? Because the spatial axis has only 3 tokens —
            # masking must be structured to create a meaningful cross-channel task.
            # The model must learn: C3↔C4 anti-correlation during lateralized
            # motor imagery, Cz as baseline — exactly the structure that matters
            # for downstream BCI classification.
            # Fraction masked: 1/C (33% for C=3) — conservative, keeps
            # cross-region at block_prob/3 ≈ 8-10%.
            chan_idx = torch.randint(0, C, (B,), device=device)  # (B,)
            chan_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
            chan_mask.scatter_(1, chan_idx.unsqueeze(1), True)

        # Expand channel mask to all timesteps: (B, C) → (B, N, C) → (B, L)
        token_mask_r2 = chan_mask.unsqueeze(1).expand(-1, N, -1).reshape(B, L)

        # Apply round 2 (only to tokens NOT already masked in round 1)
        new_masks = token_mask_r2 & ~token_mask_r1
        new_float = new_masks.unsqueeze(-1).float()
        x = x * (1 - new_float) + mask_tok * new_float

        combined_mask = token_mask_r1 | new_masks
        return x, combined_mask

    def encoder_forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device) if not isinstance(channel_list, torch.Tensor) else channel_list.to(device)

        # ── Compute per-sample channel correlation from RAW EEG ──
        # Needed for correlation-aware masking or combined block+corr masking.
        # The correlation matrix is the same geometric object that
        # the Riemannian attention branch operates on (covariance → SPD).
        if self.use_corr_masking or self.mask_strategy == 'block_corr':
            with torch.no_grad():
                x_centered = x - x.mean(dim=-1, keepdim=True)          # (B, C, T)
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
        # When RoPE is enabled, skip additive temporal PE — position is injected
        # via rotation inside the temporal attention (no magnitude imbalance).
        if not self.use_rope:
            seq_idx = torch.arange(0, N, device=device, dtype=torch.long)
            eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
            tp = self.temporal_embedding_e(eeg_seq_indices)
            x += tp

        # ── Masking ──
        # Strategies: 'random' (BERT), 'block' (temporal blocks),
        #             'corr' (correlation chains), 'block_corr' (combined)
        if self.mask_strategy == 'block_corr':
            x, mask = self.mask_block_corr(x, corr, C)
        elif self.mask_strategy == 'block':
            x, mask = self.mask_temporal_block(x, C)
        elif self.use_corr_masking or self.mask_strategy == 'corr':
            x, mask = self.mask_corr_channels(x, corr, C)
        else:
            x, mask = self.mask_bert(x)

        # Extract channel indices for the adaptive Riemannian reference
        channel_idx = channel_list[0]  # (C,) global channel indices

        # Pass through adaptive Riemannian transformer layers.
        # mask is passed so masked channels are zeroed before covariance
        # computation (prevents mask token from contaminating the geometry).
        for transformer in self.encoder:
            x = transformer(x, C, channel_idx=channel_idx, mask=mask)

        x = self.norm_enc(x)
        x = self.fc(x)

        return x, mask

    def _encode_features(self, eeg, channel_list):
        """Run encoder WITHOUT masking — returns normalized features.
        Used for online diagnostic metrics during validation."""
        B, C, T = eeg.shape
        device = eeg.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device) \
            if not isinstance(channel_list, torch.Tensor) else channel_list.to(device)

        x = self.patch(eeg)
        N = x.shape[1]
        L = N * C
        x = x.reshape(B, L, -1)

        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        x = x + self.channel_embedding_e(chan_id)

        if not self.use_rope:
            seq_idx = torch.arange(N, device=device)
            eeg_seq = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
            x = x + self.temporal_embedding_e(eeg_seq)

        channel_idx = channel_list[0]
        for layer in self.encoder:
            x = layer(x, C, channel_idx=channel_idx)

        return self.norm_enc(x)

    def forward(self, eeg, channel_list):
        pred, mask = self.encoder_forward(eeg, channel_list)
        target, pad = self.patchify_1d(eeg, self.patch_size)
        B, Seq, Ch, P = target.shape
        target = target.view(B, Seq * Ch, P)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, unbiased=False, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # ── Time-domain reconstruction loss ──
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss_time = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)

        # ── Spectral auxiliary loss ──
        # Forces the encoder to represent frequency content explicitly.
        # Per-patch FFT: 16 samples → 9 frequency bins (0, 8, 16, ..., 64 Hz).
        # Log-magnitude for better dynamic range (small spectral differences
        # at low power levels matter for EEG band features).
        if self.spectral_loss_weight > 0:
            pred_spec = torch.fft.rfft(pred, dim=-1).abs()    # (B, L, P//2+1)
            tgt_spec  = torch.fft.rfft(target, dim=-1).abs()  # (B, L, P//2+1)
            # Log1p for numerical stability and to compress dynamic range
            spec_loss_per_patch = (
                (torch.log1p(pred_spec) - torch.log1p(tgt_spec)) ** 2
            ).mean(dim=-1)
            loss_spec = (spec_loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss_time + self.spectral_loss_weight * loss_spec
            # Store for logging in training_step
            self._last_loss_time = loss_time.detach()
            self._last_loss_spec = loss_spec.detach()
        else:
            loss = loss_time

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                      gradient_clip_algorithm=None):
        """Clip gradients to prevent residual stream blow-up.
        Without this, x @ x^T in the covariance computation can overflow
        float32 → NaN in the Padé log-map → training crash."""
        self.clip_gradients(optimizer, gradient_clip_val=1.0,
                           gradient_clip_algorithm='norm')

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

        # ── NaN guard: stop training immediately if loss or predictions explode ──
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n{'='*60}")
            print(f"[NaN DETECTED] loss={loss.item()} at epoch={self.current_epoch}, "
                  f"batch={batch_idx}")
            print(f"  pred has NaN: {torch.isnan(pred).any().item()}, "
                  f"Inf: {torch.isinf(pred).any().item()}")
            print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"{'='*60}\n")
            raise ValueError(
                f"NaN/Inf detected in training loss at epoch {self.current_epoch}, "
                f"batch {batch_idx}. Check logs above for diagnostics."
            )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log mask ratio — useful to monitor correlation masking coverage
        mask_ratio = mask.float().mean()
        self.log("mask_ratio", mask_ratio, on_step=False, on_epoch=True, prog_bar=True)

        # Log spectral vs time loss components (when spectral loss is active)
        if self.spectral_loss_weight > 0 and hasattr(self, '_last_loss_time'):
            self.log("loss_time", self._last_loss_time, on_step=False, on_epoch=True)
            self.log("loss_spec", self._last_loss_spec, on_step=False, on_epoch=True)

        # Log the learned head scales across layers for analysis
        for i, layer in enumerate(self.encoder):
            # Spatial Riemannian bias head scales (Contribution 1: score bias α_h)
            scales = layer.attn.riemannian_bias.head_scales.detach()
            self.log(f"head_scale_mean/layer_{i}", scales.mean(), on_step=False, on_epoch=True)
            self.log(f"head_scale_std/layer_{i}", scales.std(), on_step=False, on_epoch=True)

            # Value bias β_h (Contribution 1: geometric value mixing, early layers only)
            if hasattr(layer.attn, 'value_beta'):
                vbeta = layer.attn.value_beta.detach()
                self.log(f"value_beta_mean/layer_{i}", vbeta.mean(), on_step=False, on_epoch=True)
                self.log(f"value_beta_max/layer_{i}", vbeta.abs().max(), on_step=False, on_epoch=True)

            # C3: Learnable tangent-space reference μ^(l).
            # Should converge toward the log-Euclidean Fréchet mean of the
            # covariance distribution — the diagnostic showed ‖mean log S‖_F
            # ranging ~9–28 across layers; μ should grow toward similar magnitudes
            # during training, shifting the effective reference away from identity.
            mu = layer.attn.riemannian_bias.mu_log
            if mu is not None:
                self.log(f"mu_frobenius/layer_{i}", mu.detach().norm(),
                         on_step=False, on_epoch=True)
                self.log(f"mu_max_abs/layer_{i}", mu.detach().abs().max(),
                         on_step=False, on_epoch=True)

            # C2: Luna temporal compression diagnostics
            if hasattr(layer.attn, 'luna_temporal'):
                luna = layer.attn.luna_temporal
                # SPD distance weight β — should grow from 0 if geometry helps
                self.log(f"luna_spd_beta/layer_{i}", luna.spd_beta.detach(),
                         on_step=False, on_epoch=True)
                # Prototype factor norms — track whether prototypes are specializing
                # mu_proto_factors: (l, C_total, r) — compute ‖U_q‖_F per slot
                U = luna.mu_proto_factors.detach()
                proto_norms = U.reshape(luna.num_slots, -1).norm(dim=-1)
                self.log(f"luna_proto_norm_mean/layer_{i}", proto_norms.mean(),
                         on_step=False, on_epoch=True)
                self.log(f"luna_proto_norm_std/layer_{i}", proto_norms.std(),
                         on_step=False, on_epoch=True)

            # ── RoPE frequency logging ──
            if hasattr(layer.attn, 'temporal_rope'):
                rope = layer.attn.temporal_rope
                # Convert to Hz for interpretability (dt ≈ patch_size/128)
                dt = self.patch_size / 128.0
                freqs_hz = rope.omega.detach() / (2.0 * 3.14159 * dt)
                self.log(f"rope_freq_mean_hz/layer_{i}", freqs_hz.mean(),
                         on_step=False, on_epoch=True)
                self.log(f"rope_freq_min_hz/layer_{i}", freqs_hz.min(),
                         on_step=False, on_epoch=True)
                self.log(f"rope_freq_max_hz/layer_{i}", freqs_hz.max(),
                         on_step=False, on_epoch=True)
                self.log(f"rope_freq_std_hz/layer_{i}", freqs_hz.std(),
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

        # ════════════════════════════════════════════════════════════
        # DIAGNOSTIC METRICS — early warning signals for feature quality
        # These let you judge in 5 epochs whether the model is learning
        # the right features, without waiting for downstream evaluation.
        # ════════════════════════════════════════════════════════════

        # ── 1. Per-frequency-band reconstruction quality ──
        # Decompose MSE into low-freq (0-16Hz) and high-freq (16-64Hz).
        # If the model only learns amplitude, low-freq MSE drops fast
        # but high-freq stays high. Good features → both improve.
        target_raw, _ = self.patchify_1d(data, self.patch_size)
        B, Seq, Ch, P = target_raw.shape
        target_flat = target_raw.view(B, Seq * Ch, P)
        if self.norm_pix_loss:
            mean = target_flat.mean(dim=-1, keepdim=True)
            var = target_flat.var(dim=-1, unbiased=False, keepdim=True)
            target_flat = (target_flat - mean) / (var + 1.e-6)**.5

        with torch.no_grad():
            pred_fft = torch.fft.rfft(pred, dim=-1).abs()    # (B, L, P//2+1)
            tgt_fft = torch.fft.rfft(target_flat, dim=-1).abs()
            spec_mse = (pred_fft - tgt_fft) ** 2              # (B, L, P//2+1)

            # Split: bins 0-1 = 0-16Hz (low), bins 2+ = 16-64Hz (high)
            # At 128Hz / patch_size=16: freq_resolution = 8Hz per bin
            low_mse = (spec_mse[..., :2] * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / 2
            high_mse = (spec_mse[..., 2:] * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / (P//2 - 1)

            self.log("val_mse_freq_low", low_mse, on_step=False, on_epoch=True)
            self.log("val_mse_freq_high", high_mse, on_step=False, on_epoch=True)

        # ── 2. Temporal attention structure ──
        # Entropy of temporal attention weights. Low entropy = peaked/local
        # (model learned temporal structure). High entropy = uniform
        # (temporal branch not learning). Track per-layer.
        # Only compute every 5 batches to limit overhead.
        if batch_idx % 5 == 0:
            with torch.no_grad():
                # Run a forward pass through encoder to capture attention
                x_diag = self.patch(data)
                N_diag = x_diag.shape[1]
                C_diag = x_diag.shape[2]
                L_diag = N_diag * C_diag
                x_diag = x_diag.reshape(B, L_diag, -1)

                # Add channel embeddings
                ch_list = channel_list
                if not isinstance(ch_list, torch.Tensor):
                    ch_list = torch.tensor(ch_list, dtype=torch.long, device=data.device)
                if ch_list.dim() == 1:
                    ch_list = ch_list.unsqueeze(0).expand(B, -1)
                chan_id = ch_list.unsqueeze(1).repeat(1, N_diag, 1).view(B, L_diag)
                x_diag = x_diag + self.channel_embedding_e(chan_id)

                if not self.use_rope:
                    seq_idx = torch.arange(N_diag, device=data.device)
                    eeg_seq = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C_diag).view(B, L_diag)
                    x_diag = x_diag + self.temporal_embedding_e(eeg_seq)

                # Forward through first and last encoder layer, capture temporal scores
                channel_idx = ch_list[0]
                for li, layer in enumerate(self.encoder):
                    if li in (0, len(self.encoder) - 1):
                        # Compute temporal attention scores for this layer
                        attn = layer.attn
                        x_n = layer.norm1(x_diag)
                        H = attn.num_heads
                        H2 = attn.heads_per_branch
                        d = attn.dim_head

                        qkv = attn.qkv(x_n).reshape(B, L_diag, 3, H, d).permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]
                        q_t = q[:, :H2]
                        k_t = k[:, :H2]

                        q_t = q_t.reshape(B, H2, N_diag, C_diag, d)
                        k_t = k_t.reshape(B, H2, N_diag, C_diag, d)
                        # Average across channels for efficiency
                        q_t_avg = q_t.mean(dim=3)  # (B, H2, N, d)
                        k_t_avg = k_t.mean(dim=3)

                        if hasattr(attn, 'temporal_rope') and attn.use_rope:
                            # RoPE expects (BC, H, N, d) — here B acts as the BC dim
                            q_t_avg, k_t_avg = attn.temporal_rope(q_t_avg, k_t_avg)

                        scores = (q_t_avg @ k_t_avg.transpose(-2, -1)) / (d ** 0.5)
                        attn_w = scores.softmax(dim=-1)  # (B, H2, N, N)

                        # Entropy: -sum(p * log(p))
                        entropy = -(attn_w * (attn_w + 1e-10).log()).sum(dim=-1).mean()
                        max_entropy = math.log(N_diag)  # uniform distribution entropy
                        norm_entropy = entropy / max_entropy  # 0=peaked, 1=uniform

                        # Locality: fraction of attention on nearest 3 positions
                        diag_mask = torch.zeros(N_diag, N_diag, device=data.device)
                        for offset in range(-1, 2):
                            diag_mask += torch.diag(torch.ones(N_diag - abs(offset),
                                                    device=data.device), offset)
                        locality = (attn_w * diag_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()

                        tag = "first" if li == 0 else "last"
                        self.log(f"val_attn_entropy_{tag}", norm_entropy,
                                 on_step=False, on_epoch=True)
                        self.log(f"val_attn_locality_{tag}", locality,
                                 on_step=False, on_epoch=True)

                    x_diag = layer(x_diag, C_diag, channel_idx=channel_idx)

        # ── 3. Feature sensitivity to temporal shuffle ──
        # Online version of the temporal diagnostic.
        # Run every 10 batches: shuffle temporal order, compare features.
        # Cosine < 0.95 = model uses temporal structure (good).
        if batch_idx % 10 == 0:
            with torch.no_grad():
                # Get features from normal input
                feat_normal = self._encode_features(data, channel_list)
                # Shuffle temporal order
                B_s, C_s, T_s = data.shape
                perm = torch.randperm(T_s // self.patch_size, device=data.device)
                # Reshape to patches, shuffle, reshape back
                data_patched = data.reshape(B_s, C_s, -1, self.patch_size)
                data_shuffled = data_patched[:, :, perm, :].reshape(B_s, C_s, T_s)
                feat_shuffled = self._encode_features(data_shuffled, channel_list)

                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    feat_normal.reshape(B_s, -1),
                    feat_shuffled.reshape(B_s, -1),
                    dim=-1
                ).mean()
                self.log("val_shuffle_cosine", cos_sim, on_step=False, on_epoch=True)


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)
