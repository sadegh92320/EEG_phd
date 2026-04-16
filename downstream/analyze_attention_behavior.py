"""
Analyze how the Riemannian bias shapes attention inside the trained encoder.

Produces a TEXT FILE summary covering three questions:

1. How did per-head α_h converge across layers?
     → which heads and which layers actually use the geometric bias?

2. What is the entropy of spatial attention per head per layer?
     → does the bias sharpen or soften attention distributions?

3. How well does spatial attention align with the log(S) bias?
     → is the bias actively shaping attention (high correlation) or being
       overridden by the Q·K term (near-zero correlation)?

Usage:
    python -m downstream.analyze_attention_behavior \
        --checkpoint path/to/your.ckpt \
        --dataset bci_comp_2a \
        --num_batches 5 \
        --batch_size 16 \
        --output downstream/results/attention_analysis.txt
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from einops import rearrange

from downstream.get_benchmark_foundation_model import (
    DATASET_CONFIGS,
    build_riemann_transformer_para,
)
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.downstream_dataset import Downstream_Dataset


# ────────────────────────────────────────────────────────────────
# Capture infrastructure
# ────────────────────────────────────────────────────────────────

def install_hooks(model):
    """Install hooks to capture L (log-S) and normalized attention input per layer.

    We do NOT hook into the softmax directly — we compute spatial attention
    ourselves from captured Q/K + L, which is deterministic and lets us match
    the paper's description of the bias behavior exactly.
    """
    captures = defaultdict(lambda: {"L": [], "x_norm": [], "channel_idx": None})

    def make_bias_hook(layer_idx):
        # Hook fires after riemannian_bias.forward returns — result is (bias, L)
        def hook(module, inputs, outputs):
            bias, L = outputs
            captures[layer_idx]["L"].append(L.detach().cpu())
        return hook

    def make_attn_pre_hook(layer_idx):
        # Hook fires before attention.forward — we capture x_norm (first arg)
        # plus channel_idx (kwarg), which we need to compute spatial attention.
        def hook(module, inputs):
            # inputs is (x_norm, num_chan) at minimum; kwargs pass channel_idx
            # The actual forward signature is:
            #   (x_norm, num_chan, residual=None, channel_idx=None, mask=None)
            # When called positionally: inputs = (x_norm, num_chan, residual)
            # When called with kwargs, we need to handle both. Simplest: capture
            # x_norm (first positional) and read channel_idx from the most recent
            # call via a class attribute set by the caller, but that's fragile.
            # Better: record x_norm here, and record channel_idx from the outer
            # forward loop (set as module attribute before calling).
            captures[layer_idx]["x_norm"].append(inputs[0].detach().cpu())
        return hook

    handles = []
    for i, layer in enumerate(model.encoder):
        handles.append(layer.attn.riemannian_bias.register_forward_hook(
            make_bias_hook(i)))
        handles.append(layer.attn.register_forward_pre_hook(
            make_attn_pre_hook(i)))
    return captures, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def compute_spatial_attention(x_norm, attn_module, L, C, alpha_h):
    """Recompute spatial attention from captured x_norm and L.

    Mirrors the relevant portion of AdaptiveRiemannianParallelAttention.forward
    but stops at the post-softmax attention weights (no value mixing).

    Args:
        x_norm: (B, L_tok, D) post-LayerNorm input to the attention module
        attn_module: the attention module (we read its qkv linear)
        L: (B*N, C, C) captured tangent vectors
        C: number of channels
        alpha_h: (H2,) per-head scales for the spatial branch
    Returns:
        attn_weights: (B*N, H2, C, C) post-softmax spatial attention weights
    """
    with torch.no_grad():
        B, L_tok, D = x_norm.shape
        N = L_tok // C
        H = attn_module.num_heads
        H2 = attn_module.heads_per_branch
        d = attn_module.dim_head

        # Shared QKV projection
        qkv = attn_module.qkv(x_norm)                                       # (B, L_tok, 3D)
        qkv = qkv.reshape(B, L_tok, 3, H, d).permute(2, 0, 3, 1, 4)          # (3, B, H, L_tok, d)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        q_s = q[:, H2:]                                                      # (B, H2, L_tok, d)
        k_s = k[:, H2:]

        q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=C)
        k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=C)

        # Score = Q·K^T / sqrt(d) + α_h · L
        score = (q_s @ k_s.transpose(-2, -1)) / (d ** 0.5)                    # (B*N, H2, C, C)
        alpha = alpha_h.view(1, H2, 1, 1).to(score.dtype)
        bias = alpha * L.unsqueeze(1)                                         # (B*N, 1, C, C) * per-head
        score = score + bias
        attn = score.softmax(dim=-1)
    return attn  # (B*N, H2, C, C)


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────

def entropy_per_row(attn):
    """Per-row entropy of an attention matrix, averaged over batch dim.
    attn: (..., C, C) with rows summing to 1
    Returns: (..., C) per-row entropy, or average over the batch dim if .mean called later.
    """
    # Clamp to avoid log(0)
    p = attn.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # (..., C)
    return ent


def correlation_per_sample(A, B):
    """Pearson correlation between flattened per-sample matrices.

    A, B: (N_samples, C, C). Computes correlation for each sample, returns (N_samples,).
    """
    a = A.reshape(A.size(0), -1).float()
    b = B.reshape(B.size(0), -1).float()
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    num = (a * b).sum(dim=-1)
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return num / den


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention behavior in the trained encoder."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bci_comp_2a",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str,
                        default="downstream/results/attention_analysis.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds_cfg = DATASET_CONFIGS[args.dataset]

    # ── Build model ──
    model = build_riemann_transformer_para(
        num_classes=ds_cfg["num_classes"],
        checkpoint_path=args.checkpoint,
        num_channels=ds_cfg["num_channels"],
        data_length=ds_cfg["data_length"],
    )
    model = model.to(args.device).eval()

    # ── Read α_h per layer per head (static, no forward pass needed) ──
    alpha_values = []  # list of (H2,) arrays per layer
    mu_present = []
    mu_norms = []
    for layer in model.encoder:
        bias_mod = layer.attn.riemannian_bias
        alpha_values.append(bias_mod.head_scales.detach().cpu().clone())
        if getattr(bias_mod, "mu_log", None) is not None:
            mu_present.append(True)
            mu_norms.append(bias_mod.mu_log.detach().cpu().norm().item())
        else:
            mu_present.append(False)
            mu_norms.append(None)
    num_layers = len(alpha_values)
    H2 = alpha_values[0].shape[0]

    # ── Install hooks and run forward passes ──
    captures, handles = install_hooks(model)

    loader = DownstreamDataLoader(
        data_path=ds_cfg["data_path"],
        config=ds_cfg["config_yaml"],
        custom_dataset_class=Downstream_Dataset,
        norm_mode="riemann_para",
        base_sfreq=ds_cfg["sampling_rate"],
    )
    train_ds, _, _ = loader.get_data_for_population()
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[analyze] Running {args.num_batches} batches through encoder…")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if batch_idx >= args.num_batches:
                break
            x, _, channel_list = batch
            x = x.to(args.device)
            channel_list = channel_list.to(args.device)
            _ = model(x, channel_list)

    remove_hooks(handles)

    # ── Compute spatial attention per layer ──
    print(f"[analyze] Computing spatial attention weights per layer…")
    C = ds_cfg["num_channels"]

    # Aggregators per layer:
    entropy_per_head_per_layer = []      # list of (H2,) mean entropy per head
    corr_per_head_per_layer = []         # list of (H2,) mean attn-L correlation
    corr_abs_per_head_per_layer = []     # list of (H2,) mean |corr|
    n_samples_per_layer = []

    for layer_idx in range(num_layers):
        xn_list = captures[layer_idx]["x_norm"]
        L_list = captures[layer_idx]["L"]
        if len(xn_list) == 0 or len(L_list) == 0:
            entropy_per_head_per_layer.append(None)
            corr_per_head_per_layer.append(None)
            corr_abs_per_head_per_layer.append(None)
            n_samples_per_layer.append(0)
            continue

        x_norm = torch.cat(xn_list, dim=0)        # (B_total, L_tok, D)
        L_cap = torch.cat(L_list, dim=0)          # (B_total*N, C, C)
        attn_module = model.encoder[layer_idx].attn
        alpha_h = alpha_values[layer_idx]

        # Spatial attention
        attn = compute_spatial_attention(x_norm, attn_module, L_cap, C, alpha_h)
        # attn: (BN, H2, C, C), rows sum to 1
        n_samples_per_layer.append(attn.size(0))

        # Entropy per head (average over B*N and over row dim)
        ent = entropy_per_row(attn)                # (BN, H2, C)
        ent_per_head = ent.mean(dim=(0, 2))        # (H2,)
        entropy_per_head_per_layer.append(ent_per_head.cpu())

        # Attention-L correlation per sample per head
        # L_cap is (BN, C, C), attn is (BN, H2, C, C). For each head:
        # correlate attn[:, h] with L_cap per sample → (BN,), then average.
        corrs = []
        corrs_abs = []
        for h in range(H2):
            c = correlation_per_sample(attn[:, h], L_cap)  # (BN,)
            corrs.append(c.mean().item())
            corrs_abs.append(c.abs().mean().item())
        corr_per_head_per_layer.append(np.array(corrs))
        corr_abs_per_head_per_layer.append(np.array(corrs_abs))

    # ── Write text report ──
    print(f"[analyze] Writing report to {args.output}")
    with open(args.output, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CHANNEL-AWARE ATTENTION BEHAVIOR ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset:    {args.dataset}\n")
        f.write(f"Num batches processed: {args.num_batches}\n")
        f.write(f"Channels (C): {C}\n")
        f.write(f"Spatial heads (H2): {H2}\n")
        f.write(f"Encoder layers: {num_layers}\n")
        f.write(f"Mu whitening active: {any(mu_present)}\n")
        f.write("\n")

        # ── 1. α_h per head per layer ──
        f.write("-" * 70 + "\n")
        f.write("1. PER-HEAD α_h — learnable scalar controlling Riemannian bias strength\n")
        f.write("-" * 70 + "\n")
        f.write("Interpretation: α close to 0 → bias disabled for that head.\n")
        f.write("                α large (positive or negative) → head relies on geometry.\n\n")
        f.write(f"{'layer':<7}" + "  ".join(f"h{h:<6}" for h in range(H2))
                + f"  {'|α|_mean':<10}{'|α|_max':<10}{'head_var':<10}\n")
        all_abs = []
        for lyr, alpha in enumerate(alpha_values):
            row = f"{lyr:<7}"
            for h in range(H2):
                row += f"{alpha[h].item():+.4f}  "
            mean_abs = alpha.abs().mean().item()
            max_abs = alpha.abs().max().item()
            var = alpha.var().item()
            row += f"{mean_abs:<10.4f}{max_abs:<10.4f}{var:<10.4f}\n"
            f.write(row)
            all_abs.append(mean_abs)
        f.write("\n")
        f.write(f"Layer ranking by |α| mean:\n")
        for lyr in np.argsort(-np.array(all_abs)):
            f.write(f"  layer {lyr}: mean|α|={all_abs[lyr]:.4f}\n")
        f.write("\n")

        # ── 2. μ magnitudes (if whitening active) ──
        if any(mu_present):
            f.write("-" * 70 + "\n")
            f.write("2. μ (learnable tangent-space reference) — Frobenius norm per layer\n")
            f.write("-" * 70 + "\n")
            f.write("Interpretation: μ grows toward the Fréchet mean of log-covariances.\n")
            f.write("                Should roughly match ‖mean logS‖_F values from diagnostic.\n\n")
            for lyr, (present, norm) in enumerate(zip(mu_present, mu_norms)):
                if present:
                    f.write(f"  layer {lyr}: ‖μ‖_F = {norm:.4f}\n")
                else:
                    f.write(f"  layer {lyr}: μ not present\n")
            f.write("\n")

        # ── 3. Attention entropy ──
        f.write("-" * 70 + "\n")
        f.write("3. ATTENTION ENTROPY per head per layer (averaged across samples)\n")
        f.write("-" * 70 + "\n")
        uniform_ent = float(np.log(C))
        f.write(f"Uniform baseline (maximum entropy): log(C) = {uniform_ent:.4f}\n")
        f.write("Lower entropy = sharper attention. Higher = more spread.\n\n")
        f.write(f"{'layer':<7}" + "  ".join(f"h{h:<6}" for h in range(H2))
                + f"  {'mean':<10}{'vs uniform':<12}\n")
        for lyr in range(num_layers):
            ent = entropy_per_head_per_layer[lyr]
            if ent is None:
                continue
            row = f"{lyr:<7}"
            for h in range(H2):
                row += f"{ent[h].item():.4f}  "
            mean_ent = ent.mean().item()
            frac = mean_ent / uniform_ent
            row += f"{mean_ent:<10.4f}{frac:<12.2%}\n"
            f.write(row)
        f.write("\n")

        # ── 4. Attention-L correlation ──
        f.write("-" * 70 + "\n")
        f.write("4. ATTENTION ↔ log(S) CORRELATION per head per layer\n")
        f.write("-" * 70 + "\n")
        f.write("Pearson correlation between post-softmax attention weights and\n")
        f.write("the tangent vector log(S) - μ used in the bias.\n")
        f.write("  |corr| ~ 0      : bias had negligible effect on attention (α≈0 or Q·K dominates)\n")
        f.write("  |corr| 0.1-0.3  : bias weakly shapes attention\n")
        f.write("  |corr| > 0.3    : bias actively shapes attention patterns\n")
        f.write("  sign: positive = attention follows L's positive entries; negative = opposes\n\n")
        f.write(f"{'layer':<7}" + "  ".join(f"h{h:<6}" for h in range(H2))
                + f"  {'mean|corr|':<12}\n")
        for lyr in range(num_layers):
            c = corr_per_head_per_layer[lyr]
            cabs = corr_abs_per_head_per_layer[lyr]
            if c is None:
                continue
            row = f"{lyr:<7}"
            for h in range(H2):
                row += f"{c[h]:+.3f}   "
            row += f"{cabs.mean():<12.4f}\n"
            f.write(row)
        f.write("\n")

        # ── Cross-quantity summary ──
        f.write("-" * 70 + "\n")
        f.write("5. SUMMARY — joint interpretation per layer\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'layer':<7}{'|α|_mean':<12}{'entropy':<12}{'|corr|':<12}"
                f"{'regime':<30}\n")
        for lyr in range(num_layers):
            if entropy_per_head_per_layer[lyr] is None:
                continue
            a = alpha_values[lyr].abs().mean().item()
            e = entropy_per_head_per_layer[lyr].mean().item()
            cabs = corr_abs_per_head_per_layer[lyr].mean().item() \
                if corr_abs_per_head_per_layer[lyr] is not None else float("nan")

            if a < 0.05:
                regime = "bias near-zero (dormant)"
            elif cabs < 0.1:
                regime = "α used but Q·K dominates"
            elif cabs < 0.3:
                regime = "bias weakly shapes attention"
            else:
                regime = "bias strongly shapes attention"
            f.write(f"{lyr:<7}{a:<12.4f}{e:<12.4f}{cabs:<12.4f}{regime:<30}\n")
        f.write("\n")
        f.write("END OF ANALYSIS\n")

    print(f"[analyze] Done. See {args.output}")


if __name__ == "__main__":
    main()
