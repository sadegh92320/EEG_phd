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
    FilterBankBias,
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


def hilbert_transform_last_dim(x):
    """
    Compute the Hilbert transform along the last dimension via FFT.

    Input:  (..., T) real tensor
    Output: (..., T) real tensor — imaginary part of the analytic signal.

    The analytic signal is z(t) = x(t) + i·H[x(t)] where |z| is the
    instantaneous amplitude envelope and angle(z) is the instantaneous
    phase. Predicting (x, H[x]) forces the encoder to represent phase
    explicitly because H[x] cannot be recovered from x by amplitude-only
    interpolation.

    Implementation: standard FFT-domain filter (keep DC + Nyquist, double
    positive frequencies, zero negatives) — matches scipy.signal.hilbert.
    """
    N = x.shape[-1]
    # Cast to float for FFT even if input is in autocast dtype — ensures
    # correct analytic-signal construction without precision drift.
    x_f = x.float()
    Xf = torch.fft.fft(x_f, dim=-1)
    h = torch.zeros(N, device=x.device, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    analytic = torch.fft.ifft(Xf * h, dim=-1)
    return analytic.imag.to(x.dtype)


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
                 use_rope=False, rope_freq_min=0.5, rope_freq_max=50.0,
                 rope_learnable=True,
                 spectral_loss_weight=0.0,
                 mask_strategy='random', mask_block_size=4,
                 use_filter_bank=False, fb_num_bands=5,
                 fb_sample_rate=128.0, fb_kernel_size=65,
                 fb_band_edges=None, fb_learnable_cutoffs=False,
                 fb_beta_init=0.0, fb_l1_weight=0.0,
                 use_hilbert_target=False,
                 disable_bias=False,
                 use_temporal_bias=False, max_temporal_patches=128):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.num_channels = num_channels
        self.use_corr_masking = use_corr_masking
        self.use_rope = use_rope
        self.use_filter_bank = use_filter_bank
        self.fb_num_bands = fb_num_bands
        # FB sparsity penalty on per-layer (H_spatial × K) β tensor.
        # 0.0 → off (probe contract). ~1e-5 encourages band specialization
        # in fresh pretraining without crushing the bias to zero.
        self.fb_l1_weight = float(fb_l1_weight)
        # Cheap Hilbert analytic-signal target: predict (x, H[x]) per patch
        # instead of just x. Forces phase-aware representation via explicit
        # target supervision, at negligible compute cost. The output head
        # doubles (patch_size → 2*patch_size) and the target gains a Hilbert
        # component computed via FFT.
        self.use_hilbert_target = use_hilbert_target
        self.patch_size = patch_size

        # Adaptive Riemannian parallel transformer layers
        # log_mode='pade' → Padé [1,1] approximant: log(S) ≈ 2(S-I)(I+S)^{-1}
        # C1: Riemannian spatial attention bias (score bias α_h · log(Σ))
        #
        # Layer-wise bias schedule (compute / signal trade-off):
        #  - Temporal bias only on the FIRST HALF of layers (i < depth_e // 2).
        #    Temporal Σ adds a Padé log-map per (B*C) sample of size N×N which
        #    is the dominant added cost; restricting to early layers ~halves the
        #    extra cost while keeping the inductive bias where the signal is
        #    most "raw" (later layers are mixed enough that the bias contributes
        #    little — same intuition as ALiBi/RoPE only-on-low-layers studies).
        #  - Spatial bias DISABLED on the LAST layer (i == depth_e - 1):
        #    empirically the head-scales of the last layer collapse toward 0 and
        #    its μ_log barely moves — the layer is doing pure value mixing for
        #    reconstruction, not spatial structuring. Skipping its log-map saves
        #    one C×C eigh-equivalent per batch with no expected cost to features.
        self.encoder = nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='pade',
                use_value_bias=(i < value_bias_layers),
                learn_mu_reference=learn_mu_reference,
                use_rope=use_rope,
                rope_freq_min=rope_freq_min,
                rope_freq_max=rope_freq_max,
                rope_learnable=rope_learnable,
                use_filter_bank=use_filter_bank,
                fb_num_bands=fb_num_bands,
                fb_beta_init=fb_beta_init,
                # Disable spatial bias on the last layer (μ effectively unused there).
                disable_bias=(disable_bias or (i == depth_e - 1)),
                # Temporal bias only on the first half of layers.
                use_temporal_bias=(use_temporal_bias and i < depth_e // 2),
                max_temporal_patches=max_temporal_patches,
            ) for i in range(depth_e)
        ])

        # Encoder-level filter bank: raw-signal SincConv + mask-aware covariance.
        # Instantiated once, outputs (B, K, C, C) reused across all layers.
        if self.use_filter_bank:
            self.filter_bank = FilterBankBias(
                sample_rate=fb_sample_rate,
                num_bands=fb_num_bands,
                kernel_size=fb_kernel_size,
                band_edges=fb_band_edges,
                learnable_cutoffs=fb_learnable_cutoffs,
            )
        else:
            self.filter_bank = None

        self.mask_prob = mask_prob
        self.mask_strategy = mask_strategy
        self.mask_block_size = mask_block_size
        self.patch_size = patch_size
        self.patch = PatchEEG(patch_size=patch_size, embed_dim=enc_dim)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, enc_dim)))
        # Output head: predicts (x, H[x]) if Hilbert target, else just x.
        # 2 × patch_size output when Hilbert is active — first half is the
        # raw-signal prediction, second half is the Hilbert prediction.
        _out_dim = 2 * patch_size if use_hilbert_target else patch_size
        self.fc = nn.Linear(enc_dim, _out_dim)
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

        # Keep a reference to the raw signal for the filter bank BEFORE
        # patching — FB runs on raw time-domain samples (128 Hz).
        x_raw = x

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

        # Default path: PE added to all tokens before masking.
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

        # ── Filter-bank bias (Run 4 FB-C1) ──
        # Build a (B, C, T) visibility mask from the token mask and compute
        # the K per-band centered log-covariance tensors ONCE here; reuse
        # across all encoder layers. Leakage-critical: the raw-signal samples
        # corresponding to masked patches are zeroed before SincConv inside
        # FilterBankBias.forward (mask-aware covariance, pairwise normalization).
        fb_log_S = None
        if self.filter_bank is not None:
            with torch.no_grad():
                # Token layout: (B, N*C) with order [t0_c0, t0_c1, …, t0_cC, t1_c0, …]
                # → (B, N, C) → (B, C, N). Then upsample each patch to
                # patch_size raw-time samples (uniform expansion).
                token_mask_BNC = rearrange(mask, 'b (n c) -> b n c', c=C)
                vis_BNC = (~token_mask_BNC).to(x_raw.dtype)                # (B, N, C)
                vis_BCN = vis_BNC.permute(0, 2, 1).contiguous()            # (B, C, N)
                # Expand each of the N patches to patch_size raw samples
                # → (B, C, N*patch_size). Trim / pad to the true T length.
                vis_BCT = vis_BCN.unsqueeze(-1).expand(
                    B, C, N, self.patch_size
                ).reshape(B, C, N * self.patch_size)
                if vis_BCT.shape[-1] > T:
                    vis_BCT = vis_BCT[..., :T]
                elif vis_BCT.shape[-1] < T:
                    # Pad the tail as visible (unmasked) — rare path when
                    # patch_size doesn't divide T exactly.
                    pad = torch.ones(
                        B, C, T - vis_BCT.shape[-1],
                        device=vis_BCT.device, dtype=vis_BCT.dtype,
                    )
                    vis_BCT = torch.cat([vis_BCT, pad], dim=-1)

            fb_log_S = self.filter_bank(x_raw, vis_BCT, channel_idx)   # (B, K, C, C)

        # Pass through adaptive Riemannian transformer layers.
        # mask is passed so masked channels are zeroed before covariance
        # computation (prevents mask token from contaminating the geometry).
        for transformer in self.encoder:
            x = transformer(x, C, channel_idx=channel_idx, mask=mask,
                            fb_log_S=fb_log_S)

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

        x_raw = eeg
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

        # FB bias with all-visible mask (no masking during feature extraction).
        fb_log_S = None
        if self.filter_bank is not None:
            fb_log_S = self.filter_bank(x_raw, None, channel_idx)  # (B, K, C, C)

        for layer in self.encoder:
            x = layer(x, C, channel_idx=channel_idx, fb_log_S=fb_log_S)

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

        # ── Hilbert analytic-signal target (cheap phase supervision) ──
        # Split the doubled output head into (pred_x, pred_h): first half
        # predicts the raw signal, second half predicts its Hilbert transform.
        # Target: the Hilbert transform is computed on the full trial then
        # patchified identically to the raw target. Loss = MSE(x) + MSE(H[x]).
        # This is mathematically equivalent to predicting the analytic signal
        # z = x + i·H[x] with joint MSE — phase is in the target, so gradient
        # signal cannot be optimized away by amplitude-only features.
        if self.use_hilbert_target:
            # Split output head halves
            pred_x, pred_h = pred.chunk(2, dim=-1)  # each (B, L, P)

            # Compute Hilbert of the raw trial and patchify to match target
            eeg_h = hilbert_transform_last_dim(eeg)
            target_h, _ = self.patchify_1d(eeg_h, self.patch_size)
            target_h = target_h.view(B, Seq * Ch, P)
            if self.norm_pix_loss:
                mean_h = target_h.mean(dim=-1, keepdim=True)
                var_h = target_h.var(dim=-1, unbiased=False, keepdim=True)
                target_h = (target_h - mean_h) / (var_h + 1.e-6) ** .5

            # Per-component MSE over masked positions
            loss_per_patch_x = ((pred_x - target) ** 2).mean(dim=-1)
            loss_per_patch_h = ((pred_h - target_h) ** 2).mean(dim=-1)
            loss_time = (loss_per_patch_x * mask).sum() / mask.sum().clamp_min(1.0)
            loss_hilbert = (loss_per_patch_h * mask).sum() / mask.sum().clamp_min(1.0)

            # Store component losses for per-step logging
            self._last_loss_time = loss_time.detach()
            self._last_loss_hilbert = loss_hilbert.detach()

            # Use pred_x as the returned pred for downstream diagnostics
            # (val_pred_std, per-band MSE etc. operate on raw-signal prediction)
            pred_for_diagnostics = pred_x
            loss = loss_time + loss_hilbert
        else:
            # Original path — raw-signal prediction only.
            loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
            loss_time = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
            pred_for_diagnostics = pred
            loss = loss_time
            self._last_loss_time = loss_time.detach()

        # ── Spectral auxiliary loss (orthogonal to Hilbert) ──
        # Only applied when explicitly enabled; operates on the raw-signal
        # prediction component (pred_for_diagnostics) for consistency.
        if self.spectral_loss_weight > 0:
            pred_spec = torch.fft.rfft(pred_for_diagnostics, dim=-1).abs()
            tgt_spec  = torch.fft.rfft(target, dim=-1).abs()
            spec_loss_per_patch = (
                (torch.log1p(pred_spec) - torch.log1p(tgt_spec)) ** 2
            ).mean(dim=-1)
            loss_spec = (spec_loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.spectral_loss_weight * loss_spec
            self._last_loss_spec = loss_spec.detach()

        return loss, pred_for_diagnostics, mask

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

        # Log Hilbert target components separately.
        # loss_time = MSE on raw signal reconstruction.
        # loss_hilbert = MSE on Hilbert (phase-shifted) component reconstruction.
        # If loss_hilbert stays near its init value while loss_time drops, the
        # model is learning amplitude only — early warning for phase collapse.
        if self.use_hilbert_target and hasattr(self, '_last_loss_hilbert'):
            self.log("loss_time_raw", self._last_loss_time, on_step=False, on_epoch=True)
            self.log("loss_hilbert", self._last_loss_hilbert, on_step=False, on_epoch=True)

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

            # Filter-bank β_{h,k,l} (Run 4 FB-C1: per-head per-band scalars).
            # Kill-condition for FB: if fb_beta_norm stays < 0.05 at epoch 30,
            # the bands are not being used → FB contributes nothing and the
            # model is effectively reduced to Run 2. Log Frobenius norm and
            # per-band magnitude (mean over heads) for diagnostics.
            if hasattr(layer.attn, 'fb_beta'):
                fb_beta = layer.attn.fb_beta.detach()  # (H_spatial, K)
                self.log(f"fb_beta_norm/layer_{i}", fb_beta.norm(),
                         on_step=False, on_epoch=True)
                self.log(f"fb_beta_max/layer_{i}", fb_beta.abs().max(),
                         on_step=False, on_epoch=True)
                # Per-band mean |β| — exposes which bands are "winning"
                # (δ θ α β γ for K=5 with default cutoffs).
                per_band = fb_beta.abs().mean(dim=0)  # (K,)
                for k in range(per_band.shape[0]):
                    self.log(f"fb_beta_band{k}/layer_{i}", per_band[k],
                             on_step=False, on_epoch=True)

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

            # Run 6: temporal Riemannian bias diagnostics.
            # head_scales_temporal grows if the model finds Σ_temporal useful.
            # mu_temporal_norm grows toward log-Euclidean Fréchet mean of
            # patch-sequence autocorrelation across the data distribution.
            if hasattr(layer.attn, 'temporal_riemannian_bias') and layer.attn.temporal_riemannian_bias is not None:
                t_scales = layer.attn.temporal_riemannian_bias.head_scales.detach()
                self.log(f"temporal_head_scale_mean/layer_{i}", t_scales.mean(),
                         on_step=False, on_epoch=True)
                self.log(f"temporal_head_scale_std/layer_{i}", t_scales.std(),
                         on_step=False, on_epoch=True)
                t_mu = layer.attn.temporal_riemannian_bias.mu_log
                if t_mu is not None:
                    self.log(f"temporal_mu_frobenius/layer_{i}", t_mu.detach().norm(),
                             on_step=False, on_epoch=True)
                    self.log(f"temporal_mu_max_abs/layer_{i}", t_mu.detach().abs().max(),
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

        # ── FB sparsity penalty (Run 4 FB-C1) ──
        # Tiny L1 on the per-layer (H_spatial × K) β tensor encourages band
        # specialization: each head settles on 1–2 bands instead of a flat
        # mixture. Off by default (fb_l1_weight=0). Recommended ~1e-5 for
        # fresh pretraining; logged as `fb_l1_loss` so we can confirm it's
        # not dominating the reconstruction loss.
        if self.use_filter_bank and self.fb_l1_weight > 0.0:
            fb_l1 = sum(
                layer.attn.fb_beta.abs().sum()
                for layer in self.encoder
                if hasattr(layer.attn, 'fb_beta')
            )
            fb_penalty = self.fb_l1_weight * fb_l1
            self.log("fb_l1_loss", fb_penalty, on_step=False, on_epoch=True)
            loss = loss + fb_penalty

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
        # (temporal branch not learning). Track first and last layer.
        # Only compute on first validation batch per epoch (cheap enough).
        if batch_idx == 0:
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

        # ── 3. Online temporal diagnostic ──
        # Mirrors the offline temporal_importance_diagnostic.py so you can
        # compare directly with C1 results (shuffle=0.994, constant=0.725,
        # channel_only=0.811) and C1+RoPE (shuffle=0.996, constant=0.477,
        # channel_only=0.556).
        # Runs once per epoch (batch_idx==0): 4 encoder forward passes.
        # ~10-15s overhead per epoch.
        if batch_idx == 0:
            with torch.no_grad():
                B_s, C_s, T_s = data.shape
                feat_normal = self._encode_features(data, channel_list)
                feat_flat = feat_normal.reshape(B_s, -1)

                # ── Shuffle: randomize temporal order of patches ──
                # Tests if model uses temporal ORDER.
                # C1: 0.994, C1+RoPE: 0.996 (no effect = model ignores order)
                perm = torch.randperm(T_s // self.patch_size, device=data.device)
                data_patched = data.reshape(B_s, C_s, -1, self.patch_size)
                data_shuffled = data_patched[:, :, perm, :].reshape(B_s, C_s, T_s)
                feat_shuffled = self._encode_features(data_shuffled, channel_list)
                cos_shuffle = F.cosine_similarity(
                    feat_flat, feat_shuffled.reshape(B_s, -1), dim=-1
                ).mean()
                self.log("val_shuffle_cosine", cos_shuffle, on_step=False, on_epoch=True)

                # ── Constant: replace all patches with channel mean ──
                # Tests if model uses temporal CONTENT.
                # C1: 0.725, C1+RoPE: 0.477 (lower = stronger content dependence)
                chan_mean = data.mean(dim=-1, keepdim=True).expand_as(data)
                feat_const = self._encode_features(chan_mean, channel_list)
                cos_const = F.cosine_similarity(
                    feat_flat, feat_const.reshape(B_s, -1), dim=-1
                ).mean()
                self.log("val_constant_cosine", cos_const, on_step=False, on_epoch=True)

                # ── Channel-only: keep channel identity, remove temporal variation ──
                # Each patch replaced by the channel's global mean patch.
                # Tests if model uses temporal VARIATION.
                # C1: 0.811, C1+RoPE: 0.556
                data_patched_co = data.reshape(B_s, C_s, -1, self.patch_size)
                mean_patch = data_patched_co.mean(dim=2, keepdim=True)  # (B, C, 1, P)
                data_chanonly = mean_patch.expand_as(data_patched_co).reshape(B_s, C_s, T_s)
                feat_chanonly = self._encode_features(data_chanonly, channel_list)
                cos_chanonly = F.cosine_similarity(
                    feat_flat, feat_chanonly.reshape(B_s, -1), dim=-1
                ).mean()
                self.log("val_chanonly_cosine", cos_chanonly, on_step=False, on_epoch=True)

                # ── Channel shuffle: permute channel assignments ──
                # Tests if model uses SPATIAL (cross-channel) structure.
                # If spatial branch learned good features, shuffling which
                # channel each patch belongs to should destroy features (low cosine).
                # High cosine = spatial branch not contributing much.
                chan_perm = torch.randperm(C_s, device=data.device)
                data_chan_shuffled = data[:, chan_perm, :]
                # Also permute channel_list to match shuffled data
                chan_list_shuffled = torch.stack([ch[chan_perm] for ch in channel_list])
                feat_chan_shuffled = self._encode_features(data_chan_shuffled, chan_list_shuffled)
                cos_chan_shuffle = F.cosine_similarity(
                    feat_flat, feat_chan_shuffled.reshape(B_s, -1), dim=-1
                ).mean()
                self.log("val_chan_shuffle_cosine", cos_chan_shuffle, on_step=False, on_epoch=True)


if __name__ == "__main__":
    seed_everything(42)
    L.seed_everything(42, workers=True)
