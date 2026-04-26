import torch.nn as nn
import torch
import numpy as np
from MAE_pretraining.pretraining import PatchEEG, ChannelPositionalEmbed, TemporalPositionalEncoding
from MAE_pretraining.pretraining import TemporalEncoding
from einops import rearrange
from MAE_pretraining.graph_embedding import GraphDataset
from MAE_pretraining.gnn import GATModel
import yaml
from torch_geometric.data import Data
from MAE_pretraining.transformer_variants import (
    TransformerLayerViT,
    RiemannianCrissCrossTransformer,
    AdaptiveRiemannianParallelTransformer,
    FilterBankBias,
)


class RiemannDownstreamHead(nn.Module):
    """
    Riemannian SPD classification head.

    Pipeline:
        (B, C*N, D) → reshape (B, N, C, D) → mean over N → (B, C, D)
        → spatial covariance (B, C, C) → regularize + εI
        → approx log map (Σ - I) → vectorize upper triangle
        → LayerNorm → Dropout → Linear → (B, num_classes)
    """
    def __init__(self, embed_dim, num_channels, num_classes, dropout=0.1, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        # Upper triangle (including diagonal) of C×C symmetric matrix
        tri_size = num_channels * (num_channels + 1) // 2
        self.norm = nn.LayerNorm(tri_size)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.final = nn.Linear(tri_size, num_classes)

    def forward(self, x, num_channels):
        """
        Args:
            x: (B, L, D) where L = C * N (all patch tokens)
            num_channels: int, number of EEG channels C
        Returns:
            logits: (B, num_classes)
        """
        B, L, D = x.shape
        C = num_channels

        # 1. Reshape to separate channels and time, pool over time
        x = rearrange(x, "b (n c) d -> b n c d", c=C)
        x = x.mean(dim=1)                # (B, C, D)

        # 2. Spatial covariance + regularize
        eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
        cov = x @ x.transpose(-1, -2) / D + self.eps * eye  # (B, C, C)

        # 3. Approx log map: log(Σ) ≈ Σ - I (Taylor first-order, stable gradients)
        log_cov = cov - eye

        # 4. Vectorize upper triangle (symmetric matrix → vector)
        idx = torch.triu_indices(C, C, device=log_cov.device)
        features = log_cov[:, idx[0], idx[1]]  # (B, C*(C+1)/2)

        # 5. Classify
        features = self.norm(features)
        features = self.dropout(features)
        return self.final(features)


class CombinedRiemannDownstreamHead(nn.Module):
    """
    Contribution 2: Tangent-Augmented Classification Head.

    Concatenates standard pooled encoder output with tangent-space features
    extracted from the encoder's last-layer spatial covariance via Padé log map.

    Pipeline:
        (B, N*C, D) → split into:
          Branch A: mean-pool → LayerNorm → (B, D)          [standard path]
          Branch B: reshape (B,N,C,D) → mean over N → cov → Padé log(Σ)
                    → upper triangle → LayerNorm → (B, tri)  [tangent path]
        Concatenate → Dropout → Linear → (B, num_classes)

    Why this works:
    - Branch A captures content (what the brain is doing)
    - Branch B captures spatial geometry (how channels co-activate)
    - Tangent-space classification is the gold standard in BCI
      (Barachant et al. 2012) — but here applied to pretrained
      representations instead of raw EEG, which should be strictly
      better because C1 biases spatial attention to maintain
      meaningful geometry through 8 layers.

    Uses the SAME Padé [1,1] log map as C1's encoder, ensuring geometric
    consistency between pretraining and downstream classification.

    The synergy with C1:
    - C1 biases spatial attention to follow manifold structure during
      pretraining → encoder covariance is geometrically meaningful
    - C2 harvests that geometry for classification
    - Without C1, Branch B is just raw-signal tangent features (modest)
    - With C1, Branch B uses pretrained manifold-aware features (better)
    """
    def __init__(self, embed_dim, num_channels, num_classes, dropout=0.1, eps=1e-5,
                 aggregation="cov_averaged", log_mode="linear"):
        """
        Args:
            aggregation:
                "time_averaged" — time-average features, then one covariance.
                "cov_averaged" — per-timestep covariance, averaged in SPD, 1 log map.
                "per_timestep" — per-timestep covariance, log-Euclidean mean (N logs).
            log_mode:
                "pade"   — Padé [1,1] approximant (accurate, requires linalg.solve).
                "linear" — first-order Taylor log(S) ≈ S − I (no solve, fast).
                           Recommended on MPS where linalg.solve has high overhead.
        """
        super().__init__()
        assert aggregation in {"time_averaged", "cov_averaged", "per_timestep"}
        assert log_mode in {"pade", "linear"}
        self.log_mode = log_mode
        self.num_channels = num_channels
        self.eps = eps
        self.aggregation = aggregation

        # Upper triangle (including diagonal) of C×C symmetric matrix
        tri_size = num_channels * (num_channels + 1) // 2

        # Branch A: pooled encoder output
        self.norm_pool = nn.LayerNorm(embed_dim)

        # Branch B: tangent-space features
        self.norm_tangent = nn.LayerNorm(tri_size)

        # Combined classifier
        combined_dim = embed_dim + tri_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.final = nn.Linear(combined_dim, num_classes)

    @staticmethod
    def _pade_log(S, eye):
        """Padé [1,1] log map: log(S) ≈ 2(S - I)(S + I)^{-1}.
        Same formula used in C1's AdaptiveRiemannianAttentionBias."""
        A = S - eye
        B = S + eye
        L = 2.0 * torch.linalg.solve(B, A)
        return (L + L.transpose(-2, -1)) * 0.5

    @staticmethod
    def _linear_log(S, eye):
        """First-order Taylor: log(S) ≈ S − I. No solve, trivially cheap.
        Good when S is close to I (ensured by LayerNorm-normalized encoder output
        and the eps ridge). Recommended on devices where batched linalg.solve
        has high per-call overhead (e.g., MPS)."""
        return S - eye

    def _log_map(self, S, eye):
        return self._pade_log(S, eye) if self.log_mode == "pade" else self._linear_log(S, eye)

    def forward(self, x, num_channels):
        """
        Args:
            x: (B, L, D) where L = C * N (all patch tokens from encoder)
            num_channels: int, number of EEG channels C
        Returns:
            logits: (B, num_classes)
        """
        B, L, D = x.shape
        C = num_channels

        # ── Branch A: Standard pooled output ──
        pooled = self.norm_pool(x).mean(dim=1)  # (B, D)

        # ── Branch B: Tangent-space features from spatial covariance ──
        x_spatial = rearrange(x, "b (n c) d -> b n c d", c=C)  # (B, N, C, D)
        eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)

        if self.aggregation == "time_averaged":
            # Original: average features over time, then covariance
            x_avg = x_spatial.mean(dim=1)  # (B, C, D)
            cov = x_avg.float() @ x_avg.float().transpose(-1, -2) / D + self.eps * eye
            log_cov = self._log_map(cov, eye).to(x.dtype)
        elif self.aggregation == "cov_averaged":
            # Per-timestep covariance averaged in SPD space, single log map.
            # Key identity: mean_t(x_t @ x_t^T / D) = (1/(N*D)) * X_stack @ X_stack^T
            # where X_stack is x reshaped to (B, C, N*D). Computable as a single
            # GEMM with no (B, N, C, C) intermediate — same cost structure as
            # time_averaged, just with N*D inner dimension instead of D.
            N = x_spatial.size(1)
            x_stack = x_spatial.transpose(1, 2).reshape(B, C, N * D).contiguous()
            cov_mean = torch.bmm(x_stack, x_stack.transpose(-1, -2)) / (N * D)
            cov_mean = cov_mean.float() + self.eps * eye
            log_cov = self._pade_log(cov_mean, eye).to(x.dtype)
        else:
            # per_timestep: log-Euclidean mean (N log maps). Most principled,
            # most expensive. Kept for ablation comparisons.
            eye_n = eye.unsqueeze(0)
            x_f = x_spatial.float()
            cov_t = torch.einsum('bncd,bned->bnce', x_f, x_f) / D + self.eps * eye_n
            BN = B * x_spatial.size(1)
            cov_flat = cov_t.reshape(BN, C, C)
            eye_flat = torch.eye(C, device=x.device, dtype=cov_flat.dtype).unsqueeze(0)
            log_flat = self._log_map(cov_flat, eye_flat)
            log_t = log_flat.reshape(B, x_spatial.size(1), C, C)
            log_cov = log_t.mean(dim=1).to(x.dtype)

        # Vectorize upper triangle
        idx = torch.triu_indices(C, C, device=log_cov.device)
        tangent_features = log_cov[:, idx[0], idx[1]]  # (B, C*(C+1)/2)
        tangent_features = self.norm_tangent(tangent_features)

        # ── Concatenate and classify ──
        combined = torch.cat([pooled, tangent_features], dim=-1)  # (B, D + tri)
        combined = self.dropout(combined)
        return self.final(combined)


class GeodesicPrototypeHead(nn.Module):
    """
    Contribution 2: Geodesic Prototype Classifier.

    Classifies by Log-Euclidean distance between the trial's spatial
    geometry (from the encoder's last-layer covariance) and learnable
    class-specific prototypes on the tangent space of the SPD manifold.

    Pipeline:
        (B, N*C, D) → reshape (B, N, C, D) → mean over N → (B, C, D)
        → covariance → Padé [1,1] log map → upper triangle → (B, tri)
        → distance to K prototypes → logits = −d_k

    Why this links to C1:
        C1 biases spatial attention to preserve meaningful geometry through
        8 encoder layers. This head reads that geometry directly — classifying
        by *how channels co-activate*, not by raw feature magnitude. Each
        prototype captures a class-specific spatial pattern (e.g., left-motor
        vs. right-motor channel coupling).

    Uses the SAME Padé [1,1] log map as C1's encoder, ensuring consistency
    between the Riemannian geometry used during pretraining and downstream.
    """
    def __init__(self, embed_dim, num_channels, num_classes, dropout=0.1, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        tri_size = num_channels * (num_channels + 1) // 2

        # Normalize tangent vectors before distance computation
        self.norm = nn.LayerNorm(tri_size)

        # Learnable class prototypes — small random init to break symmetry
        self.prototypes = nn.Parameter(torch.randn(num_classes, tri_size) * 0.02)

        # Learnable temperature for softmax over distances
        self.temperature = nn.Parameter(torch.tensor(10.0))

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @staticmethod
    def _pade_log(S, eye):
        """Padé [1,1] log map: log(S) ≈ 2(S - I)(S + I)^{-1}.
        Same formula used in C1's AdaptiveRiemannianAttentionBias."""
        A = S - eye                          # numerator
        B = S + eye                          # denominator
        L = 2.0 * torch.linalg.solve(B, A)  # (S+I)^{-1} (S-I) * 2
        # Symmetrize to kill numerical asymmetry
        return (L + L.transpose(-2, -1)) * 0.5

    def forward(self, x, num_channels):
        """
        Args:
            x: (B, L, D) where L = C * N (all patch tokens from encoder)
            num_channels: int, number of EEG channels C
        Returns:
            logits: (B, num_classes) — negative scaled distances
        """
        B, L, D = x.shape
        C = num_channels

        # Time-averaged spatial embeddings: (B, C, D)
        x_spatial = rearrange(x, "b (n c) d -> b n c d", c=C)
        x_spatial = x_spatial.mean(dim=1)  # (B, C, D)

        # Covariance → SPD → Padé log map → upper triangle
        eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
        cov = x_spatial.float() @ x_spatial.float().transpose(-1, -2) / D + self.eps * eye
        log_cov = self._pade_log(cov, eye).to(x.dtype)

        idx = torch.triu_indices(C, C, device=log_cov.device)
        tangent = log_cov[:, idx[0], idx[1]]  # (B, tri)
        tangent = self.norm(tangent)
        tangent = self.dropout(tangent)

        # Log-Euclidean distance to each prototype: (B, K)
        dists = torch.cdist(tangent, self.prototypes, p=2)  # (B, K)

        # Negative distance as logits, scaled by learnable temperature
        return -dists * self.temperature.abs()


class DownstreamHead(nn.Module):
    """
    Classification head for the downstream model.

    Modes:
        "token"      – classify from [CLS] token (index 0): LayerNorm → Dropout → Linear
        "avg"        – mean-pool over patch tokens, skip [CLS]: LayerNorm → Dropout → Linear
        "all_simple" – per-token linear projection → flatten → Dropout → Linear
    """
    def __init__(self, embed_dim, num_classes, mode="avg", num_tokens=None, dropout=0.1):
        super().__init__()
        assert mode in {"token", "avg", "all_simple"}
        self.mode = mode

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if mode in {"token", "avg"}:
            self.norm = nn.LayerNorm(embed_dim)
            self.final = nn.Linear(embed_dim, num_classes)

        elif mode == "all_simple":
            assert num_tokens is not None and num_tokens > 0, \
                "num_tokens must be provided for mode='all_simple'."
            self.num_tokens = int(num_tokens)
            self.per_token = nn.Linear(embed_dim, 64)
            self.final = nn.Linear(self.num_tokens * 64, num_classes)

    def forward(self, cls_token, patch_tokens):
        if self.mode == "token":
            x = cls_token.squeeze(1) if cls_token.dim() == 3 else cls_token
            x = self.norm(x)
            x = self.dropout(x)
            return self.final(x)

        elif self.mode == "avg":
            x = self.norm(patch_tokens).mean(dim=1)
            x = self.dropout(x)
            return self.final(x)

        elif self.mode == "all_simple":
            x = self.per_token(patch_tokens)  # (B, L, 64)
            x = x.flatten(1)                  # (B, L*64)
            x = self.dropout(x)
            return self.final(x)


# ════════════════════════════════════════════════════════════════
# Novelty 1 — Embedding channel positional encoding  (baseline)
# ════════════════════════════════════════════════════════════════

class Downstream(nn.Module):
    """
    Base downstream model: frozen pretrained ViT encoder + trainable head.
    Channel encoding: nn.Embedding lookup (standard positional encoding).
    Encoder: TransformerLayerViT.
    """
    def __init__(self, checkpoint_path, config=None,
                 max_embedding=2000, enc_dim=512, depth_e=8,
                 patch_size=16, aggregation="avg", num_classes=9, head_dropout=0.1, head_choice = 'linear'):
        super().__init__()

        self.config = config
        self.enc_dim = enc_dim
        self.patch_size = patch_size

        # ── Encoder backbone ──
        self.encoder = self._build_encoder(enc_dim, depth_e)
        self.patch = PatchEEG(patch_size=patch_size, embed_dim=enc_dim)
        self.norm_enc = nn.LayerNorm(enc_dim)
        self.temporal_embedding = TemporalPositionalEncoding(d_model=enc_dim, max_len=max_embedding)
        self.class_token = nn.Parameter(torch.zeros(1, 1, enc_dim))

        # ── Channel embedding (novelty: nn.Embedding) ──
        self.channel_embedding = ChannelPositionalEmbed(embedding_dim=enc_dim)

        # ── Load pretrained weights ──
        if checkpoint_path is not None:
            self._load_pretrained(checkpoint_path)

        # ── Freeze encoder ──
        self._freeze_encoder()

        # ── Classification head ──
        head_mode = "token" if aggregation == "class" else "avg"
        self.head_choice = head_choice
        if head_choice == "linear":
            self.head = DownstreamHead(
                embed_dim=enc_dim, num_classes=num_classes,
                mode=head_mode, dropout=head_dropout,
            )
        elif head_choice == "riemann":
            num_ch = len(config["channel_list"]) if config is not None else 22
            self.head = RiemannDownstreamHead(
                embed_dim=enc_dim, num_channels=num_ch,
                num_classes=num_classes, dropout=head_dropout,
            )
        elif head_choice == "riemann_combined":
            num_ch = len(config["channel_list"]) if config is not None else 22
            self.head = CombinedRiemannDownstreamHead(
                embed_dim=enc_dim, num_channels=num_ch,
                num_classes=num_classes, dropout=head_dropout,
            )
        elif head_choice == "geodesic_prototype":
            num_ch = len(config["channel_list"]) if config is not None else 22
            self.head = GeodesicPrototypeHead(
                embed_dim=enc_dim, num_channels=num_ch,
                num_classes=num_classes, dropout=head_dropout,
            )
        self.fc = self.head  # alias for TrainerDownstream linear_probe detection
        self.aggregation = aggregation

    # ── Builder (override in subclasses for different transformer types) ──

    @staticmethod
    def _build_encoder(enc_dim, depth_e):
        return nn.ModuleList([
            TransformerLayerViT(enc_dim, nhead=8, mlp_ratio=4, qkv_bias=True, norm=nn.LayerNorm)
            for _ in range(depth_e)
        ])

    # ── Shared helpers ──

    def _freeze_encoder(self):
        """Freeze all pretrained encoder parameters."""
        for module in [self.patch, self.channel_embedding, self.encoder, self.norm_enc, self.temporal_embedding]:
            for p in module.parameters():
                p.requires_grad = False
        self.class_token.requires_grad = False

    def _load_pretrained(self, checkpoint_path):
        """
        Load encoder weights from a full MAE (EncoderDecoder) checkpoint.

        Handles:
          - Lightning checkpoints (state_dict under "state_dict" key)
          - Key remapping: EncoderDecoder uses *_e suffix for encoder embeddings
            (channel_embedding_e, temporal_embedding_e) but Downstream drops the suffix.
          - Skips all decoder-related keys.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # ── Skip decoder keys ──
        # EncoderDecoder keys to skip: decoder.*, encoder_decoder.*, mask_token,
        # channel_embedding_d.*, temporal_embedding_d.*, norm_dec.*, fc.* (decoder proj)
        SKIP_PREFIXES = (
            "decoder.", "encoder_decoder.", "mask_token",
            "channel_embedding_d.", "temporal_embedding_d.",
            "norm_dec.", "fc.", "criterion.",
        )

        # ── Remap encoder embedding names ──
        # EncoderDecoder: channel_embedding_e.* → Downstream: channel_embedding.*
        # EncoderDecoder: temporal_embedding_e.* → Downstream: temporal_embedding.*
        KEY_REMAP = {
            "channel_embedding_e.": "channel_embedding.",
            "temporal_embedding_e.": "temporal_embedding.",
        }

        encoder_keys = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) or k == p for p in SKIP_PREFIXES):
                continue

            # Apply key remapping
            new_k = k
            for old_prefix, new_prefix in KEY_REMAP.items():
                if k.startswith(old_prefix):
                    new_k = new_prefix + k[len(old_prefix):]
                    break

            encoder_keys[new_k] = v

        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)

        # head/fc are expected to be missing (new classification head)
        missing_real = [k for k in missing if not k.startswith("fc") and not k.startswith("head")]
        if missing_real:
            print(f"  [Downstream] Missing encoder keys: {missing_real}")
        if unexpected:
            print(f"  [Downstream] Unexpected keys (ignored): {unexpected}")
        print(f"  [Downstream] Loaded pretrained encoder from {checkpoint_path}")

    def _get_channel_embedding(self, channel_list, N, B, L):
        """Compute channel positional embedding. Override in subclasses for different novelties."""
        chan_id = channel_list.unsqueeze(1).repeat(1, N, 1).view(B, L)
        return self.channel_embedding(chan_id)

    def _run_encoder(self, x, C):
        """Run through encoder blocks. Override in subclasses that need extra args (e.g. num_chan)."""
        for transformer in self.encoder:
            x = transformer(x)
        return x

    # ── Forward ──

    def forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device

        # Patch embed
        x = self.patch(x)
        N = x.shape[1]
        x = rearrange(x, "b n c d -> b (n c) d")
        L = x.shape[1]

        # Channel embedding
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)
        x = x + self._get_channel_embedding(channel_list, N, B, L)

        # Temporal embedding
        seq_idx = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
        x = x + self.temporal_embedding(seq_idx)

        # Class token
        class_token = self.class_token + self.temporal_embedding.get_class_token().view(1, 1, -1).to(device)
        class_token = class_token.expand(B, 1, -1)
        x = torch.concat([class_token, x], dim=1)

        # Encoder
        x = self._run_encoder(x, C)
        x = self.norm_enc(x)

        cls_out, patch_out = x[:, :1, :], x[:, 1:, :]

        if self.head_choice == "riemann":
            return self.head(patch_out, C)
        return self.head(cls_out, patch_out)


# ════════════════════════════════════════════════════════════════
# Novelty 2 — GNN-based channel positional encoding
# ════════════════════════════════════════════════════════════════

class DownstreamGNN(Downstream):
    """
    Channel encoding: Graph Attention Network over 3D electrode positions.
    Everything else inherited from Downstream.
    """
    def __init__(self, checkpoint_path, config=None,
                 max_embedding=2000, enc_dim=512, depth_e=8,
                 patch_size=16, aggregation="avg", num_classes=9, head_dropout=0.1):
        # Build the base (delay checkpoint loading — we need to set up GNN first)
        super().__init__(
            checkpoint_path=None, config=config, max_embedding=max_embedding,
            enc_dim=enc_dim, depth_e=depth_e, patch_size=patch_size,
            aggregation=aggregation, num_classes=num_classes, head_dropout=head_dropout,
        )

        # ── Replace channel embedding with GNN ──
        del self.channel_embedding
        with open("MAE_pretraining/info_dataset/channel_info.yaml") as f:
            ch_config = yaml.safe_load(f)
        ch_total = ch_config["channels_mapping"]
        ordered_channels = [k for k, v in sorted(ch_total.items(), key=lambda item: item[1])]

        gnn_data = GraphDataset()
        g = gnn_data.create_graph(ch_names=ordered_channels, radius=0.4)
        self.register_buffer("g_x", g.x)
        self.register_buffer("g_edge_index", g.edge_index)
        self.gnn_enc = GATModel(num_head=3, enc_dim=enc_dim)

        # ── Load pretrained weights now ──
        if checkpoint_path is not None:
            self._load_pretrained(checkpoint_path)

        # ── Freeze encoder (re-freeze + freeze GNN) ──
        self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze base encoder + GNN channel encoder."""
        for module in [self.patch, self.encoder, self.norm_enc, self.temporal_embedding]:
            for p in module.parameters():
                p.requires_grad = False
        if hasattr(self, "gnn_enc"):
            for p in self.gnn_enc.parameters():
                p.requires_grad = False
        self.class_token.requires_grad = False

    def _get_channel_embedding(self, channel_list, N, B, L):
        """GNN-based channel encoding: run GAT on electrode graph, index by channel."""
        g_device = Data(x=self.g_x, edge_index=self.g_edge_index)
        chan_total = self.gnn_enc(g_device)
        chan_total = chan_total[channel_list]
        return chan_total.unsqueeze(1).repeat(1, N, 1, 1).view(B, L, -1)


# ════════════════════════════════════════════════════════════════
# Novelty 3 — Riemannian SPD loss  (ViT encoder, novelty is the loss)
# ════════════════════════════════════════════════════════════════

class DownstreamRiemannLoss(Downstream):
    """
    Same ViT encoder as base Downstream, same embedding channel encoding.
    The novelty (Riemannian SPD loss) is applied externally at training time.
    """
    pass


# ════════════════════════════════════════════════════════════════
# Novelty 4 — Parallel Riemannian Transformer
# ════════════════════════════════════════════════════════════════

class DownstreamRiemannTransformerPara(Downstream):
    """
    Encoder: AdaptiveRiemannianParallelTransformer (head-split spatial-temporal
    attention with adaptive Riemannian bias on spatial heads, Padé log map).
    Channel encoding: nn.Embedding (inherited from base).

    The adaptive transformer needs global channel indices (channel_idx) to
    extract the correct submatrix from the learned 144×144 SPD reference.
    These are passed from forward() → _run_encoder().

    When use_frechet=True, the Fréchet mean R^{-1/2} is loaded from the
    pretrained checkpoint (saved as a buffer). This makes the Padé [1,1]
    approximation accurate by pre-whitening S near I.

    NOTE: No [CLS] token — adaptive Riemannian attention requires L = N * C.
    aggregation="class" is NOT supported; use "avg" (default).
    """

    def __init__(self, *args, aggregation="avg", **kwargs):
        # Pop kwargs that are specific to this subclass and not accepted by base Downstream
        kwargs.pop("log_mode", None)
        kwargs.pop("use_riemannian_metric", None)
        kwargs.pop("merge_k", None)
        kwargs.pop("use_frechet", None)
        # Value bias config — store before super().__init__
        # so _build_encoder can use it
        self._value_bias_layers = kwargs.pop("value_bias_layers", 4)
        # C3: learnable tangent-space centering (replaces prior residual-stream experiments)
        self._learn_mu_reference = kwargs.pop("learn_mu_reference", True)
        # EEG-RoPE temporal position encoding
        self._use_rope = kwargs.pop("use_rope", False)
        self._rope_freq_min = kwargs.pop("rope_freq_min", 0.5)
        self._rope_freq_max = kwargs.pop("rope_freq_max", 50.0)
        self._rope_learnable = kwargs.pop("rope_learnable", True)
        # Branch gate: per-layer learnable scalars scaling each branch's
        # contribution before final projection. Fine-tune-only addition —
        # gates init at 1.0 so loading pretrained weights preserves behavior.
        self._use_branch_gate = kwargs.pop("use_branch_gate", False)
        # Filter-bank C1 (Run 4): FBCSP-style per-band static spatial biases
        # summed into each layer's Riemannian bias with learnable β_{k,h,l}.
        # β init = 0 → matches pure C1+C3 bit-exactly at epoch 0 when loading
        # an old non-FB checkpoint with --use_filter_bank.
        self._use_filter_bank = kwargs.pop("use_filter_bank", False)
        self._fb_num_bands = kwargs.pop("fb_num_bands", 5)
        self._fb_sample_rate = kwargs.pop("fb_sample_rate", 128.0)
        self._fb_kernel_size = kwargs.pop("fb_kernel_size", 65)
        self._fb_band_edges = kwargs.pop("fb_band_edges", None)
        self._fb_learnable_cutoffs = kwargs.pop("fb_learnable_cutoffs", False)
        # Baseline ablation: disable Riemannian bias computation entirely.
        # When True, AdaptiveRiemannianAttentionBias.forward returns zeros
        # WITHOUT computing covariance or Padé log-map → real wall-time
        # speedup, especially on MPS (no CPU fallback for linalg.solve).
        self._disable_bias = kwargs.pop("disable_bias", False)
        # Run 6: temporal Riemannian bias (Σ_temporal on temporal heads).
        # Must match the value used at pretraining or the checkpoint will
        # fail to load (temporal_riemannian_bias submodules differ).
        self._use_temporal_bias = kwargs.pop("use_temporal_bias", False)
        self._max_temporal_patches = kwargs.pop("max_temporal_patches", 128)
        if aggregation == "class":
            raise ValueError(
                "DownstreamRiemannTransformerPara does not use a [CLS] token. "
                "Use aggregation='avg' instead."
            )
        super().__init__(*args, aggregation=aggregation, **kwargs)

        # Instantiate encoder-level filter bank AFTER super().__init__ so we
        # only build it if enabled — keeps backward compat for non-FB runs.
        if self._use_filter_bank:
            self.filter_bank = FilterBankBias(
                sample_rate=self._fb_sample_rate,
                num_bands=self._fb_num_bands,
                kernel_size=self._fb_kernel_size,
                band_edges=self._fb_band_edges,
                learnable_cutoffs=self._fb_learnable_cutoffs,
            )
        else:
            self.filter_bank = None

    def _freeze_encoder(self):
        """Freeze pretrained encoder, but keep branch-gate scalars trainable.

        The branch gate is a fine-tune-only addition: it initializes at 1.0
        (multiplicative identity) and is meant to learn per-task
        spatial/temporal mixing. The base class freezes all of self.encoder,
        which would pin the gates at 1.0 forever — making use_branch_gate=True
        produce bit-for-bit identical outputs to use_branch_gate=False.
        Re-enable the gate parameters here after the base freeze.

        FB (filter-bank) parameters are treated the same way as the branch
        gate ONLY when `_fb_linear_probe_trainable` is set — i.e., when the
        user is probing FB on top of a non-FB pretrained checkpoint. In the
        normal pipeline (FB trained during pretraining), FB params should
        freeze with the rest of the encoder at linear-probe time.
        """
        super()._freeze_encoder()
        if getattr(self, '_use_branch_gate', False):
            n_unfrozen = 0
            for block in self.encoder:
                attn = block.attn
                if getattr(attn, 'use_branch_gate', False):
                    attn.branch_gate_s.requires_grad = True
                    attn.branch_gate_t.requires_grad = True
                    n_unfrozen += 2
            print(f"  [branch gate] Unfrozen {n_unfrozen} gate scalars "
                  f"({n_unfrozen // 2} layers × 2) — encoder otherwise frozen")

        # FB params: keep trainable in linear-probe ONLY if explicitly flagged.
        # Default (pretrained-FB ckpt) = frozen with rest of encoder.
        if (getattr(self, '_use_filter_bank', False)
                and getattr(self, '_fb_linear_probe_trainable', False)):
            n_unfrozen = 0
            for block in self.encoder:
                attn = block.attn
                if getattr(attn, 'use_filter_bank', False):
                    attn.fb_beta.requires_grad = True
                    n_unfrozen += attn.fb_beta.numel()
            if self.filter_bank is not None:
                # SincConv cutoffs frozen by default at the bias level already;
                # mu_log_bank should train if FB is being probed onto an old ckpt.
                if self.filter_bank.mu_log_bank is not None:
                    self.filter_bank.mu_log_bank.requires_grad = True
                    n_unfrozen += self.filter_bank.mu_log_bank.numel()
                if self.filter_bank.sinc.learnable:
                    self.filter_bank.sinc.f_lo.requires_grad = True
                    self.filter_bank.sinc.band_hz.requires_grad = True
            print(f"  [filter bank] Unfrozen {n_unfrozen} FB params "
                  f"(fb_beta per layer + mu_log_bank) — encoder otherwise frozen")

    def _build_encoder(self, enc_dim, depth_e):
        value_bias_layers = getattr(self, '_value_bias_layers', 4)
        learn_mu_reference = getattr(self, '_learn_mu_reference', True)
        use_rope = getattr(self, '_use_rope', False)
        rope_freq_min = getattr(self, '_rope_freq_min', 0.5)
        rope_freq_max = getattr(self, '_rope_freq_max', 50.0)
        rope_learnable = getattr(self, '_rope_learnable', True)
        use_branch_gate = getattr(self, '_use_branch_gate', False)
        use_filter_bank = getattr(self, '_use_filter_bank', False)
        fb_num_bands = getattr(self, '_fb_num_bands', 5)
        disable_bias = getattr(self, '_disable_bias', False)
        # Run 6: temporal Riemannian bias config (must match pretraining)
        use_temporal_bias = getattr(self, '_use_temporal_bias', False)
        max_temporal_patches = getattr(self, '_max_temporal_patches', 128)
        return nn.ModuleList([
            AdaptiveRiemannianParallelTransformer(
                enc_dim, nhead=8, mlp_ratio=4, log_mode='pade',
                use_value_bias=(i < value_bias_layers),
                learn_mu_reference=learn_mu_reference,
                use_rope=use_rope,
                rope_freq_min=rope_freq_min,
                rope_freq_max=rope_freq_max,
                rope_learnable=rope_learnable,
                use_branch_gate=use_branch_gate,
                use_filter_bank=use_filter_bank,
                fb_num_bands=fb_num_bands,
                # Match pretraining schedule: spatial bias OFF on last layer,
                # temporal bias ON for ALL layers (Run 6 final schedule —
                # bf16 made the per-layer cost negligible so we kept all 8).
                disable_bias=(disable_bias or (i == depth_e - 1)),
                use_temporal_bias=use_temporal_bias,
                max_temporal_patches=max_temporal_patches,
            ) for i in range(depth_e)
        ])

    def _run_encoder(self, x, C, channel_idx=None, fb_log_S=None):
        """Adaptive Riemannian parallel transformer. No mask during downstream."""
        for transformer in self.encoder:
            x = transformer(x, num_chan=C, channel_idx=channel_idx,
                            fb_log_S=fb_log_S)
        return x

    def get_branch_gates(self):
        """Return per-layer (gate_s, gate_t) tuples. Empty list if gate disabled.

        Use after fine-tune to inspect which branch the model preferred on the
        current downstream dataset. Expected pattern: gate_s > gate_t on
        narrow-band tasks (MI-style), balanced on broader-spectrum tasks.
        """
        if not getattr(self, '_use_branch_gate', False):
            return []
        gates = []
        for block in self.encoder:
            attn = block.attn
            if getattr(attn, 'use_branch_gate', False):
                gs = attn.branch_gate_s.detach().float().item()
                gt = attn.branch_gate_t.detach().float().item()
                gates.append((gs, gt))
        return gates

    def forward(self, x, channel_list):
        B, C, T = x.shape
        device = x.device

        # Keep raw signal for the filter bank BEFORE patching. At downstream
        # there is no masking → the FB visibility mask is all-ones.
        x_raw = x

        # Patch embed
        x = self.patch(x)
        N = x.shape[1]

        x = rearrange(x, "b n c d -> b (n c) d")
        L = x.shape[1]

        # Channel embedding
        if channel_list.dim() == 1:
            channel_list = channel_list.unsqueeze(0).expand(B, -1)
        x = x + self._get_channel_embedding(channel_list, N, B, L)

        # Temporal embedding — skipped when RoPE is enabled (position injected
        # by rotation inside the temporal attention).
        if not getattr(self, '_use_rope', False):
            seq_idx = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
            x = x + self.temporal_embedding(seq_idx)

        # NOTE: No class token before encoder — the adaptive Riemannian
        # attention requires L = N * C (asserts L % num_chan == 0).
        # The pretraining code also omits the class token.

        # Extract global channel indices for the adaptive Riemannian reference
        # Within a batch all samples share the same channel set (same dataset)
        channel_idx = channel_list[0]  # (C,) global channel indices

        # ── Defensive guard ──
        # channel_idx is used to index mu_log (global channel space, size=TOTAL_GLOBAL_CHANNELS)
        # Bad dtype / shape / out-of-range values cause async CUDA failures that
        # surface at the next .item() sync with a garbage index — almost
        # impossible to diagnose. Catch them here, in Python, with context.
        if channel_idx.dtype not in (torch.int64, torch.int32, torch.long):
            raise RuntimeError(
                f"channel_idx must be integer dtype for embedding indexing; "
                f"got dtype={channel_idx.dtype}. "
                f"Check the dataloader — channel_list should be long, not float."
            )
        if channel_idx.dim() != 1 or channel_idx.shape[0] != C:
            raise RuntimeError(
                f"channel_idx should be shape (C,) with C={C}; "
                f"got shape={tuple(channel_idx.shape)}."
            )
        # Global channel space is 144 (TOTAL_GLOBAL_CHANNELS in transformer_variants.py).
        # Re-check here rather than import — layering-safe.
        _ci_min = int(channel_idx.min().item())
        _ci_max = int(channel_idx.max().item())
        if _ci_min < 0 or _ci_max >= 144:
            raise RuntimeError(
                f"channel_idx out of range for global channel space (0..143): "
                f"min={_ci_min}, max={_ci_max}. "
                f"Check channel_info.yaml / dataloader global-id mapping."
            )

        # ── Filter-bank bias (Run 4 FB-C1) ──
        # At downstream there's no masking → pass vis_mask=None (treated as
        # all-visible by FilterBankBias). Computed once, reused across layers.
        fb_log_S = None
        if getattr(self, 'filter_bank', None) is not None:
            fb_log_S = self.filter_bank(x_raw, None, channel_idx)  # (B, K, C, C)

        # Encoder (no class token — sequence is exactly N*C)
        x = self._run_encoder(x, C, channel_idx=channel_idx, fb_log_S=fb_log_S)
        x = self.norm_enc(x)

        # Head: dispatch based on head_choice
        cls_out = x[:, :1, :]   # first patch token as dummy CLS
        patch_out = x            # all tokens are patch tokens
        if self.head_choice in ("riemann", "riemann_combined", "geodesic_prototype"):
            return self.head(patch_out, C)
        else:
            return self.head(cls_out, patch_out)



# ════════════════════════════════════════════════════════════════
# Novelty 5 — Sequential Riemannian Transformer
# ════════════════════════════════════════════════════════════════

class DownstreamRiemannTransformerSeq(Downstream):
    """
    Encoder: RiemannianCrissCrossTransformer (sequential time→space attention
    with Riemannian bias on spatial attention).
    Channel encoding: nn.Embedding (inherited from base).
    """

    @staticmethod
    def _build_encoder(enc_dim, depth_e):
        return nn.ModuleList([
            RiemannianCrissCrossTransformer(enc_dim, nhead=8, mlp_ratio=4)
            for _ in range(depth_e)
        ])

    def _run_encoder(self, x, C):
        """Riemannian sequential transformer needs num_chan."""
        for transformer in self.encoder:
            x = transformer(x, num_chan=C)
        return x
