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
from MAE_pretraining.transformer_variants import AdaptiveRiemannianParallelTransformer, TemporalCovarianceAttentionBias


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
                 mask_prob=0.5, patch_size=16, norm_pix_loss=False,
                 use_frechet=False, frechet_path=None,
                 use_corr_masking=True,
                 use_riemannian_metric=False, metric_reg=0.001):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.num_channels = num_channels
        self.use_corr_masking = use_corr_masking
        self.use_riemannian_metric = use_riemannian_metric

        # ── Load frozen Fréchet mean reference (optional) ──
        # When use_frechet=True, the tangent-space projection pre-whitens S
        # using R^{-1/2} computed offline from the training covariance
        # distribution. This makes Padé [1,1] accurate by centering S near I.
        # When use_frechet=False, projection is at identity (no pre-whitening).
        frechet_R_inv_sqrt = None
        if use_frechet and frechet_path is not None:
            frechet_data = torch.load(frechet_path, map_location='cpu')
            frechet_R_inv_sqrt = frechet_data['R_inv_sqrt']
            print(f"[Fréchet] Loaded R^(-1/2) from {frechet_path}, "
                  f"shape={frechet_R_inv_sqrt.shape}")
        elif use_frechet:
            print("[Fréchet] WARNING: use_frechet=True but no frechet_path "
                  "provided. Falling back to identity reference.")
            use_frechet = False

        # Adaptive Riemannian parallel transformer layers
        # log_mode='pade' → Padé [1,1] approximant: log(S) ≈ 2(S-I)(I+S)^{-1}
        # use_frechet=True → pre-whiten S with offline Fréchet mean (more accurate)
        self.encoder = nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='pade',
                use_frechet=use_frechet,
                frechet_R_inv_sqrt=frechet_R_inv_sqrt,
                use_riemannian_metric=use_riemannian_metric,
                metric_reg=metric_reg,
            ) for _ in range(depth_e)
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
        # Temporal covariance dynamics bias — computes pairwise Frobenius
        # distance between per-timestep covariances. Bias is computed once
        # from clean patch embeddings and shared across all encoder layers.
        self.temporal_cov_bias = TemporalCovarianceAttentionBias(
            num_heads=n_head // 2,  # temporal heads = half of total
        )

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
        """Standard random token masking (BERT-style)."""
        device = x.device
        B, L, D = x.shape
        mask = torch.rand(B, L, device=device) < self.mask_prob
        mask_token = self.mask_token.expand_as(x)
        mask_float = mask.unsqueeze(-1).float()
        x = x * (1 - mask_float) + mask_token * mask_float
        return x, mask

    def mask_corr_channels(self, x, corr, num_chan, seed_prob=0.2,
                           chan_mask_ratio=0.2, min_channels=8):
        """
        Anti-correlated (connectivity-aware) channel masking + random patch masking.

        Two-round masking that trains BOTH spatial and temporal attention:

        Round 1 — Channel masking (anti-correlated):
            Masks ~20% of channels (all time patches for those channels).
            Targets independent/uncorrelated channels preferentially using
            Gumbel-top-k sampling on inverted connectivity scores.
            → Trains spatial attention (Riemannian branch)

        Round 2 — Random patch masking on survivors:
            Masks random individual patches among the ~80% surviving tokens
            to bring total masking to self.mask_prob (50%).
            → Trains temporal attention

        Total masking = self.mask_prob (50%), split across both rounds.

        Args:
            x:               (B, L, D) embedded tokens, L = N * C
            corr:            (B, C, C) per-sample Pearson correlation matrix
            num_chan:         C — number of EEG channels in this batch
            seed_prob:       (unused, kept for API compat)
            chan_mask_ratio:  fraction of channels to mask in round 1 (default 0.2)
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
        # ROUND 1: Anti-correlated channel masking (~30% of tokens)
        # ══════════════════════════════════════════════════════════════

        with torch.no_grad():
            # 1a. Compute per-channel independence score
            #     High |corr| with neighbors = well-connected = easy to reconstruct
            #     Low |corr| = independent = hard to reconstruct → mask these
            abs_corr = corr.abs()
            abs_corr.diagonal(dim1=-2, dim2=-1).zero_()  # ignore self-correlation
            connectivity = abs_corr.mean(dim=-1)  # (B, C) — mean |corr| per channel

            # 1b. Invert connectivity: independent channels get higher scores
            temperature = 1.0
            inv_connectivity = 1.0 - connectivity
            mask_logits = inv_connectivity / temperature

            # 1c. Number of channels to mask in round 1
            num_chan_mask = max(1, round(chan_mask_ratio * num_chan))

            # 1d. Gumbel-top-k: stochastic but biased toward independent channels
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(mask_logits).clamp(min=1e-8)
            ))
            noisy_scores = mask_logits + gumbel_noise  # (B, C)

            # 1e. Select top-k most independent channels
            _, topk_idx = noisy_scores.topk(num_chan_mask, dim=-1)

            # 1f. Build boolean channel mask
            chan_mask = torch.zeros(B, num_chan, device=device, dtype=torch.bool)
            chan_mask.scatter_(dim=-1, index=topk_idx, value=True)

        # 1g. Expand channel mask to all time patches
        #     Token layout: [t0_c0, t0_c1, …, t0_cC, t1_c0, …]
        token_mask = chan_mask.unsqueeze(1).expand(-1, N, -1).reshape(B, L)

        # 1h. Apply round-1 mask
        mask_float = token_mask.unsqueeze(-1).float()
        mask_tok = self.mask_token.expand_as(x)
        x = x * (1 - mask_float) + mask_tok * mask_float

        # ══════════════════════════════════════════════════════════════
        # ROUND 2: Random patch masking on survivors → reach self.mask_prob total
        # ══════════════════════════════════════════════════════════════
        # After round 1: (1 - chan_mask_ratio) fraction of tokens survive.
        # We need (self.mask_prob - chan_mask_ratio) more tokens masked.
        # So patch_prob = (target - round1) / (1 - round1)
        #              = (0.50 - 0.20) / (1 - 0.20) = 0.30 / 0.80 = 0.375

        remaining_frac = 1.0 - chan_mask_ratio
        extra_needed = self.mask_prob - chan_mask_ratio
        patch_mask_prob = max(0.0, extra_needed / remaining_frac)

        noise = torch.rand(B, L, device=device)
        noise[token_mask] = 1.0                    # already masked → never re-mask
        extra_mask = noise < patch_mask_prob        # (B, L)

        extra_float = extra_mask.unsqueeze(-1).float()
        x = x * (1 - extra_float) + mask_tok * extra_float

        # Combined mask for loss: union of both rounds
        combined_mask = token_mask | extra_mask

        return x, combined_mask

    def encoder_forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device
        channel_list = torch.tensor(channel_list, dtype=torch.long, device=device) if not isinstance(channel_list, torch.Tensor) else channel_list.to(device)

        # Patch the EEG: (B, C, T) → (B, N, C, D) → (B, N*C, D)
        x = self.patch(x)
        N = x.shape[1]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(B, L, -1)

        # Compute temporal covariance dynamics bias from clean patch embeddings
        # (before channel/positional embeddings contaminate the spatial structure).
        # Computed once here, shared across all encoder layers.
        temporal_cov_bias = self.temporal_cov_bias.compute_bias(x, C)  # (B, H2, N, N)

        # Channel embeddings
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        chan_embedding = self.channel_embedding_e(chan_id)
        x += chan_embedding

        # Sinusoidal temporal embeddings
        seq_idx = torch.arange(0, N, device=device, dtype=torch.long)
        eeg_seq_indices = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
        tp = self.temporal_embedding_e(eeg_seq_indices)
        x += tp

        # ── Masking (standard random BERT masking) ──
        x, mask = self.mask_bert(x)

        # Extract channel indices for the adaptive Riemannian reference
        channel_idx = channel_list[0]  # (C,) global channel indices

        # Pass through adaptive Riemannian transformer layers
        for transformer in self.encoder:
            x = transformer(x, C, channel_idx=channel_idx, temporal_cov_bias=temporal_cov_bias)

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
        # Separate param groups: metric_U gets 0.1× learning rate for stability
        if self.use_riemannian_metric:
            metric_params = []
            other_params = []
            for name, param in self.named_parameters():
                if 'metric_U' in name:
                    metric_params.append(param)
                else:
                    other_params.append(param)
            param_groups = [
                {"params": other_params, "lr": 1e-3, "weight_decay": 0.05},
                {"params": metric_params, "lr": 1e-4, "weight_decay": 0.0},  # 0.1× lr, no wd
            ]
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)

        total_steps = self.trainer.estimated_stepping_batches

        # OneCycleLR with per-group max_lr: metric_U gets 0.1× to stay stable
        if self.use_riemannian_metric:
            max_lr_list = [5e-4, 5e-5]  # [main params, metric_U]
        else:
            max_lr_list = 5e-4
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr_list,
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
            # Check Fréchet whitening output in each layer
            for i, layer in enumerate(self.encoder):
                log_map = layer.attn.riemannian_bias.adaptive_log
                if log_map.use_frechet and log_map.R_inv_sqrt is not None:
                    R = log_map.R_inv_sqrt
                    print(f"  Layer {i} R_inv_sqrt range: "
                          f"[{R.min().item():.4f}, {R.max().item():.4f}], "
                          f"cond≈{R.norm():.2f}")
            print(f"{'='*60}\n")
            # Raise to stop training — Lightning will catch and report
            raise ValueError(
                f"NaN/Inf detected in training loss at epoch {self.current_epoch}, "
                f"batch {batch_idx}. Check logs above for diagnostics."
            )

        # ── Riemannian metric regularization: λ * Σ ||M_h - I||² ──
        # Keeps learned metric near identity to prevent collapse/divergence
        if self.use_riemannian_metric:
            metric_reg_loss = sum(
                layer.attn.metric_regularization_loss() for layer in self.encoder
            )
            loss = loss + metric_reg_loss
            self.log("metric_reg_loss", metric_reg_loss, on_step=False, on_epoch=True)

            # Log per-layer metric stats for monitoring
            for i, layer in enumerate(self.encoder):
                if layer.attn.use_riemannian_metric:
                    with torch.no_grad():
                        U = layer.attn.metric_U  # (H2, d, r)
                        # ||U||_F per head — how far M has moved from I
                        u_norm = (U ** 2).sum(dim=(-2, -1)).sqrt().mean()
                        self.log(f"metric_U_norm/layer_{i}", u_norm,
                                 on_step=False, on_epoch=True)
                        # Singular values of U — shows which rank directions are active
                        # Only compute occasionally (every 100 steps) to save time
                        if batch_idx % 100 == 0:
                            sv = torch.linalg.svdvals(U.float())  # (H2, r)
                            self.log(f"metric_sv_max/layer_{i}", sv[:, 0].mean(),
                                     on_step=False, on_epoch=True)
                            self.log(f"metric_sv_min/layer_{i}", sv[:, -1].mean(),
                                     on_step=False, on_epoch=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log mask ratio — useful to monitor correlation masking coverage
        mask_ratio = mask.float().mean()
        self.log("mask_ratio", mask_ratio, on_step=False, on_epoch=True, prog_bar=True)

        # Log the learned head scales across layers for analysis
        for i, layer in enumerate(self.encoder):
            scales = layer.attn.riemannian_bias.head_scales.detach()
            self.log(f"head_scale_mean/layer_{i}", scales.mean(), on_step=False, on_epoch=True)
            self.log(f"head_scale_std/layer_{i}", scales.std(), on_step=False, on_epoch=True)

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
