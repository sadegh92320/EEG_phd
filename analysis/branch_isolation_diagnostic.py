"""
Branch Isolation Diagnostic
============================

Tests whether temporal head outputs help or hurt downstream separability.

Runs the encoder 3 ways:
  1. Normal (both branches)          → baseline separability
  2. Temporal heads zeroed out        → spatial-only separability
  3. Spatial heads zeroed out         → temporal-only separability

If spatial-only > normal → temporal heads are HURTING (diluting spatial features)
If spatial-only < normal → temporal heads are HELPING
If temporal-only is low  → temporal features alone aren't class-discriminative

Usage:
    python analysis/branch_isolation_diagnostic.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --use_rope
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.downstream_model import DownstreamRiemannTransformerPara as Downstream
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader


def make_branch_hook(mode):
    """
    Returns a hook for AdaptiveRiemannianParallelAttention.forward
    that zeros out one branch's output before concatenation.

    We hook into the attention module and modify the internal computation
    by patching the forward method.
    """
    def hook(module, input, output):
        # output is the final (B, L, D) tensor after fc projection
        # We can't separate branches here — we need to go deeper.
        # Instead, we'll use a monkey-patch approach (see below).
        return output
    return hook


def extract_features_branch(model, dataloader, device, branch="both", max_batches=20):
    """
    Extract features with branch isolation.
    branch: "both", "spatial_only", "temporal_only"
    """
    model.eval()
    model.to(device)

    # Monkey-patch attention layers to zero out one branch
    original_forwards = []
    if branch != "both":
        for transformer in model.encoder:
            attn = transformer.attn
            original_forward = attn.forward

            def make_patched_forward(orig_fwd, attn_module, branch_mode):
                def patched_forward(x_norm, num_chan, residual=None, channel_idx=None, mask=None):
                    from einops import rearrange
                    import torch.nn.functional as F

                    B, L, D = x_norm.shape
                    C = num_chan
                    N = L // C
                    H = attn_module.num_heads
                    H2 = attn_module.heads_per_branch
                    d = attn_module.dim_head

                    # Riemannian bias
                    bias_source = residual if residual is not None else x_norm
                    x_space = rearrange(bias_source, 'b (n c) d -> (b n) c d', c=C)
                    mask_space = None
                    if mask is not None:
                        mask_space = rearrange(mask, 'b (n c) -> (b n) c', c=C)
                        x_space = x_space * (~mask_space).unsqueeze(-1).float()
                    riem_bias, L_n = attn_module.riemannian_bias(
                        x_space, channel_idx, mask_space=mask_space
                    )

                    # Shared QKV
                    qkv = attn_module.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    q_t, q_s = q[:, :H2], q[:, H2:]
                    k_t, k_s = k[:, :H2], k[:, H2:]
                    v_t, v_s = v[:, :H2], v[:, H2:]

                    # Temporal attention
                    q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
                    k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)
                    v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=C)

                    if attn_module.use_rope:
                        q_t, k_t = attn_module.temporal_rope(q_t, k_t)

                    if getattr(attn_module, 'use_luna_temporal', False) and attn_module.luna_temporal is not None:
                        out_t = attn_module.luna_temporal(q_t, k_t, v_t, L_n=L_n, channel_idx=channel_idx, C=C)
                    else:
                        out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0)
                    out_t = rearrange(out_t, '(b c) h n d -> b h (n c) d', b=B, c=C)

                    # Spatial attention
                    q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=C)
                    k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=C)
                    v_s = rearrange(v_s, 'b h (n c) d -> (b n) h c d', c=C)

                    if hasattr(attn_module, 'value_beta') and attn_module.use_value_bias:
                        beta_h = attn_module.value_beta.view(1, H2, 1, 1)
                        L_exp = L_n.unsqueeze(1).to(v_s.dtype)
                        v_for_geo = v_s
                        if mask is not None:
                            mask_v = rearrange(mask, 'b (n c) -> (b n) c', c=C)
                            v_for_geo = v_s * (~mask_v).unsqueeze(1).unsqueeze(-1).float()
                        v_geo = L_exp @ v_for_geo
                        v_s = v_s + beta_h * v_geo

                    with torch.amp.autocast('cuda', enabled=False), \
                         torch.amp.autocast('cpu', enabled=False):
                        score = (q_s.float() @ k_s.float().transpose(-2, -1)) / (d ** 0.5)
                        score = score + riem_bias.float()
                        score = score.softmax(dim=-1)
                    out_s = score.to(v_s.dtype) @ v_s
                    out_s = rearrange(out_s, '(b n) h c d -> b h (n c) d', b=B, n=N)

                    # Branch isolation: zero out one branch
                    if branch_mode == "spatial_only":
                        out_t = torch.zeros_like(out_t)
                    elif branch_mode == "temporal_only":
                        out_s = torch.zeros_like(out_s)

                    # Concatenate and project
                    out = torch.cat([out_t, out_s], dim=1)
                    out = rearrange(out, 'b h l d -> b l (h d)')
                    out = attn_module.fc(out)
                    return attn_module.dropout(out)

                return patched_forward

            patched = make_patched_forward(original_forward, attn, branch)
            original_forwards.append((attn, original_forward))
            attn.forward = patched

    # Extract features
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            eeg, labels, chan_ids = batch
            eeg = eeg.to(device)
            chan_ids = chan_ids.to(device)

            B, C, T = eeg.shape
            x = model.patch(eeg)
            N = x.shape[1]
            from einops import rearrange
            x = rearrange(x, "b n c d -> b (n c) d")
            L = x.shape[1]

            cl = chan_ids
            if cl.dim() == 1:
                cl = cl.unsqueeze(0).expand(B, -1)
            x = x + model._get_channel_embedding(cl, N, B, L)

            if not getattr(model, '_use_rope', False):
                seq_idx = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1)
                seq_idx = seq_idx.repeat(B, 1, C).view(B, L)
                x = x + model.temporal_embedding(seq_idx)

            channel_idx = cl[0]
            x = model._run_encoder(x, C, channel_idx=channel_idx)
            x = model.norm_enc(x)

            features = x.mean(dim=1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    # Restore original forwards
    for attn, orig_fwd in original_forwards:
        attn.forward = orig_fwd

    return torch.cat(all_features), torch.cat(all_labels)


def compute_separability(features, labels):
    """Compute inter-class / intra-class distance ratio."""
    unique_labels = labels.unique()
    class_means = {}
    for c in unique_labels:
        mask = labels == c
        class_means[c.item()] = features[mask].mean(dim=0)

    inter_dists = []
    classes = list(class_means.keys())
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            d = (class_means[classes[i]] - class_means[classes[j]]).norm().item()
            inter_dists.append(d)
    mean_inter = np.mean(inter_dists)

    intra_dists = []
    for c in unique_labels:
        mask = labels == c
        class_feats = features[mask]
        cm = class_means[c.item()]
        intra_dists.append((class_feats - cm).norm(dim=-1).mean().item())
    mean_intra = np.mean(intra_dists)

    return mean_inter / (mean_intra + 1e-8), mean_inter, mean_intra


def run_diagnostic(checkpoint_path, data_path="downstream/data/bci_comp_2a",
                   config_path="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
                   batch_size=32, use_rope=False, max_batches=20):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"RoPE: {use_rope}")
    print()

    model = Downstream(
        checkpoint_path=checkpoint_path,
        enc_dim=512, depth_e=8, patch_size=16,
        num_classes=4,
        use_rope=use_rope,
    )
    model.to(device)
    model.eval()

    loader = DownstreamDataLoader(
        data_path=data_path,
        config=config_path,
        custom_dataset_class=Downstream_Dataset,
        base_sfreq=250,
    )
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    def collate_fn(batch):
        eegs, labels, chan_ids = zip(*batch)
        return torch.stack(eegs), torch.stack(labels), torch.stack(chan_ids)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    results = {}
    for branch in ["both", "spatial_only", "temporal_only"]:
        print(f"  Extracting features: {branch}...")
        feats, labels = extract_features_branch(
            model, test_loader, device, branch=branch, max_batches=max_batches
        )
        sep, inter, intra = compute_separability(feats, labels)
        results[branch] = {"separability": sep, "inter": inter, "intra": intra}

    print()
    print("=" * 65)
    print("BRANCH ISOLATION DIAGNOSTIC")
    print("=" * 65)
    print()
    print(f"{'Branch':<20} {'Separability':>14} {'Inter-class':>14} {'Intra-class':>14}")
    print("-" * 65)
    for branch in ["both", "spatial_only", "temporal_only"]:
        r = results[branch]
        print(f"   {branch:<20} {r['separability']:>12.4f} {r['inter']:>12.4f} {r['intra']:>12.4f}")

    print()
    both_sep = results["both"]["separability"]
    spatial_sep = results["spatial_only"]["separability"]
    temporal_sep = results["temporal_only"]["separability"]

    if spatial_sep > both_sep * 1.05:
        print("   VERDICT: Temporal heads are HURTING separability.")
        print("   → Spatial-only features are more class-discriminative.")
        print("   → Consider CLS-token fusion or separate QKV to protect spatial branch.")
    elif spatial_sep < both_sep * 0.95:
        print("   VERDICT: Temporal heads are HELPING separability.")
        print("   → Both branches contribute positively.")
    else:
        print("   VERDICT: Temporal heads have NEUTRAL effect on separability.")
        print("   → Temporal features neither help nor hurt significantly.")

    print()
    print(f"   Temporal contribution: {temporal_sep:.4f} (standalone class separability)")
    if temporal_sep < 0.05:
        print("   → Temporal features alone carry almost no class information.")
    elif temporal_sep > both_sep * 0.5:
        print("   → Temporal features carry meaningful class information.")
    print()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="downstream/data/bci_comp_2a")
    parser.add_argument("--config_path", type=str,
                        default="MAE_pretraining/info_dataset/bci_comp_2a.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    run_diagnostic(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config_path,
        batch_size=args.batch_size,
        use_rope=args.use_rope,
        max_batches=args.max_batches,
    )
