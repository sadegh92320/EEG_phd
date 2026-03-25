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
    RiemannianParallelCrissCrossTransformer,
)


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
                 patch_size=16, aggregation="avg", num_classes=9, head_dropout=0.1):
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
        self.head = DownstreamHead(
            embed_dim=enc_dim, num_classes=num_classes,
            mode=head_mode, dropout=head_dropout,
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
        for module in [self.patch, self.channel_embedding, self.encoder, self.norm_enc]:
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
        for module in [self.patch, self.encoder, self.norm_enc]:
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
    Encoder: RiemannianParallelCrissCrossTransformer (head-split spatial-temporal
    attention with Riemannian bias on spatial heads).
    Channel encoding: nn.Embedding (inherited from base).
    """

    @staticmethod
    def _build_encoder(enc_dim, depth_e):
        return nn.ModuleList([
            RiemannianParallelCrissCrossTransformer(enc_dim, nhead=8, mlp_ratio=4)
            for _ in range(depth_e)
        ])

    def _run_encoder(self, x, C):
        """Riemannian parallel transformer needs num_chan."""
        for transformer in self.encoder:
            x = transformer(x, num_chan=C)
        return x


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
