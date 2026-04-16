"""
Analyze temporal attention behavior in the trained encoder.

Produces a TEXT FILE covering:

1. Temporal attention entropy per head per layer
     → does temporal attention actually do anything, or is it near-uniform?

2. Diagonality — fraction of attention mass on the diagonal
     → high diagonal = temporal heads are near-identity (not really attending)

3. Distance distribution — short-range vs long-range attention
     → where along the time axis each head looks

4. Head diversity — pairwise correlation between heads' attention patterns
     → are the 4 temporal heads redundant or do they specialize?

5. Per-channel temporal covariance analysis
     → per-channel N×N SPD matrix structure. Could a temporal Riemannian bias
       (analog of C1 but across time) extract additional signal?

Usage:
    python -m downstream.analyze_temporal_behavior \
        --checkpoint path/to/your.ckpt \
        --dataset bci_comp_2a \
        --num_batches 5 \
        --output downstream/results/temporal_analysis.txt
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
# Capture
# ────────────────────────────────────────────────────────────────

def install_hooks(model):
    """Capture x_norm (input to attention) per layer so we can re-compute
    temporal attention weights manually (F.scaled_dot_product_attention
    does not return weights)."""
    captures = defaultdict(lambda: {"x_norm": []})

    def make_pre_hook(layer_idx):
        def hook(module, inputs):
            captures[layer_idx]["x_norm"].append(inputs[0].detach().cpu())
        return hook

    handles = []
    for i, layer in enumerate(model.encoder):
        handles.append(layer.attn.register_forward_pre_hook(make_pre_hook(i)))
    return captures, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def compute_temporal_attention(x_norm, attn_module, C):
    """Recompute temporal attention weights from captured x_norm.

    Mirrors the temporal branch of the attention module but stops at
    post-softmax weights (no value mixing).

    Returns:
        attn: (B*C, H2, N, N) temporal attention weights (rows sum to 1)
    """
    with torch.no_grad():
        B, L_tok, D = x_norm.shape
        N = L_tok // C
        H = attn_module.num_heads
        H2 = attn_module.heads_per_branch
        d = attn_module.dim_head

        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(B, L_tok, 3, H, d).permute(2, 0, 3, 1, 4)   # (3, B, H, L_tok, d)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        q_t = q[:, :H2]                                                # (B, H2, L_tok, d)
        k_t = k[:, :H2]

        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)

        score = (q_t @ k_t.transpose(-2, -1)) / (d ** 0.5)            # (B*C, H2, N, N)
        attn = score.softmax(dim=-1)
    return attn  # (B*C, H2, N, N)


def compute_per_channel_temporal_cov(x_norm, C, eps=1e-5):
    """For each (sample, channel) pair, compute N×N temporal covariance.

    x_norm: (B, L_tok, D) = (B, N*C, D)
    Returns: (B*C, N, N) SPD matrices (float32)
    """
    with torch.no_grad():
        B, L_tok, D = x_norm.shape
        N = L_tok // C
        x = rearrange(x_norm, 'b (n c) d -> (b c) n d', c=C).float()   # (B*C, N, D)
        T = torch.bmm(x, x.transpose(-2, -1)) / D                      # (B*C, N, N)
        T = T + eps * torch.eye(N).unsqueeze(0)
    return T


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────

def entropy_rows(attn):
    """Per-row entropy. Input (..., N, N). Output (..., N)."""
    p = attn.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def diagonality(attn):
    """Average diagonal mass per row. Input (..., N, N). Output scalar.
    Measures how much each query attends to itself.
    Values: 1/N (no diagonal preference) to 1.0 (pure identity)."""
    diag = attn.diagonal(dim1=-2, dim2=-1)        # (..., N)
    return diag.mean(dim=-1).mean().item()


def distance_distribution(attn, n_bins=5):
    """Distribution of attention mass by distance |i - j|.

    Input: (..., N, N). Returns: distribution over bins of relative distance.
    Splits max distance into n_bins buckets; returns mean fraction per bucket.
    """
    shape = attn.shape
    N = shape[-1]
    i = torch.arange(N).unsqueeze(1)
    j = torch.arange(N).unsqueeze(0)
    dist = (i - j).abs()                          # (N, N)

    # Normalize distance into bins
    max_d = N - 1
    bin_edges = np.linspace(0, max_d + 1e-6, n_bins + 1)
    results = np.zeros(n_bins)
    # Flatten everything except the last two dims
    flat = attn.reshape(-1, N, N)                 # (M, N, N)
    mass_per_dist = flat.mean(dim=0)              # (N, N) average attention per position pair
    for b in range(n_bins):
        mask = (dist >= bin_edges[b]) & (dist < bin_edges[b + 1])
        if mask.sum() > 0:
            results[b] = mass_per_dist[mask].sum().item()
    # Normalize so bins sum to ~1
    if results.sum() > 0:
        results /= results.sum()
    return results


def head_pairwise_correlation(attn):
    """Pairwise Pearson correlation between flattened attention patterns of
    different heads, averaged across samples.

    attn: (B*C, H2, N, N). Returns: (H2, H2) average pairwise correlation.
    """
    M, H2, N, _ = attn.shape
    flat = attn.reshape(M, H2, -1).float()        # (M, H2, N*N)
    # Center per sample per head
    flat_centered = flat - flat.mean(dim=-1, keepdim=True)
    norms = flat_centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    flat_norm = flat_centered / norms
    # Per-sample correlation matrix across heads
    corr_per_sample = torch.einsum('mhe,mke->mhk', flat_norm, flat_norm)   # (M, H2, H2)
    return corr_per_sample.mean(dim=0).numpy()


def temporal_cov_eigenvalue_stats(T):
    """Eigenvalue analysis of temporal covariances.

    T: (B*C, N, N). Returns: eig_min, eig_max, mean_cond, frob_T_minus_I.
    """
    with torch.no_grad():
        N = T.size(-1)
        eye = torch.eye(N, dtype=T.dtype).unsqueeze(0)
        eigvals = torch.linalg.eigvalsh(T.double())
        eig_min = eigvals.min().item()
        eig_max = eigvals.max().item()
        cond = (eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-12)).mean().item()
        frob_diff = (T - eye).reshape(T.size(0), -1).norm(dim=-1).mean().item()
    return eig_min, eig_max, cond, frob_diff


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bci_comp_2a",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str,
                        default="downstream/results/temporal_analysis.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds_cfg = DATASET_CONFIGS[args.dataset]

    model = build_riemann_transformer_para(
        num_classes=ds_cfg["num_classes"],
        checkpoint_path=args.checkpoint,
        num_channels=ds_cfg["num_channels"],
        data_length=ds_cfg["data_length"],
    ).to(args.device).eval()

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

    print(f"[temporal] Running {args.num_batches} batches…")
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if bi >= args.num_batches:
                break
            x, _, channel_list = batch
            x = x.to(args.device)
            channel_list = channel_list.to(args.device)
            _ = model(x, channel_list)
    remove_hooks(handles)

    C = ds_cfg["num_channels"]
    num_layers = len(model.encoder)
    H2 = model.encoder[0].attn.heads_per_branch

    # ── Compute temporal attention and temporal covariance stats per layer ──
    print("[temporal] Computing per-layer statistics…")
    entropy_per_layer = []        # list of (H2,) tensors
    diag_per_layer = []           # list of floats (averaged over heads)
    diag_per_head_per_layer = []  # list of (H2,) arrays
    dist_per_head_per_layer = []  # list of (H2, n_bins) arrays
    corr_per_layer = []           # list of (H2, H2) matrices
    cov_stats_per_layer = []      # list of tuples (eig_min, eig_max, cond, frob)
    N_actual = None

    for lyr in range(num_layers):
        xn_list = captures[lyr]["x_norm"]
        if not xn_list:
            entropy_per_layer.append(None)
            diag_per_layer.append(None)
            diag_per_head_per_layer.append(None)
            dist_per_head_per_layer.append(None)
            corr_per_layer.append(None)
            cov_stats_per_layer.append(None)
            continue

        x_norm = torch.cat(xn_list, dim=0)                   # (B_total, L_tok, D)
        attn_t = compute_temporal_attention(x_norm, model.encoder[lyr].attn, C)
        # attn_t: (B*C, H2, N, N)
        N = attn_t.size(-1)
        N_actual = N

        # Entropy per head (average over B*C and over query positions)
        ent_rows = entropy_rows(attn_t)                      # (B*C, H2, N)
        ent_per_head = ent_rows.mean(dim=(0, 2))             # (H2,)
        entropy_per_layer.append(ent_per_head)

        # Diagonality per head
        diag_per_head = []
        for h in range(H2):
            diag_per_head.append(diagonality(attn_t[:, h]))
        diag_per_head_per_layer.append(np.array(diag_per_head))
        diag_per_layer.append(float(np.mean(diag_per_head)))

        # Distance distribution per head
        dist_per_head = []
        for h in range(H2):
            dist_per_head.append(distance_distribution(attn_t[:, h], n_bins=5))
        dist_per_head_per_layer.append(np.array(dist_per_head))

        # Pairwise head correlation
        corr_per_layer.append(head_pairwise_correlation(attn_t))

        # Temporal covariance stats
        T = compute_per_channel_temporal_cov(x_norm, C)      # (B*C, N, N)
        cov_stats_per_layer.append(temporal_cov_eigenvalue_stats(T))

    # ── Write report ──
    print(f"[temporal] Writing report to {args.output}")
    with open(args.output, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TEMPORAL ATTENTION BEHAVIOR ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset:    {args.dataset}\n")
        f.write(f"Batches processed: {args.num_batches}\n")
        f.write(f"Channels (C): {C}    Temporal patches (N): {N_actual}\n")
        f.write(f"Temporal heads per layer (H2): {H2}\n")
        f.write(f"Layers: {num_layers}\n\n")

        # ── 1. Entropy ──
        uniform_ent = float(np.log(N_actual))
        f.write("-" * 70 + "\n")
        f.write("1. TEMPORAL ATTENTION ENTROPY per head per layer\n")
        f.write("-" * 70 + "\n")
        f.write(f"Uniform baseline (log(N)): {uniform_ent:.4f}\n")
        f.write(f"Lower entropy = more concentrated attention over time\n\n")
        f.write(f"{'layer':<7}" + "  ".join(f"h{h:<6}" for h in range(H2))
                + f"  {'mean':<10}{'vs uniform':<12}\n")
        for lyr in range(num_layers):
            ent = entropy_per_layer[lyr]
            if ent is None:
                continue
            row = f"{lyr:<7}"
            for h in range(H2):
                row += f"{ent[h].item():.4f}  "
            mean = ent.mean().item()
            row += f"{mean:<10.4f}{mean/uniform_ent:<12.2%}\n"
            f.write(row)
        f.write("\n")

        # ── 2. Diagonality ──
        f.write("-" * 70 + "\n")
        f.write("2. DIAGONALITY — average mass on diagonal (self-attention) per head\n")
        f.write("-" * 70 + "\n")
        f.write(f"Uniform baseline (1/N): {1/N_actual:.4f}\n")
        f.write("Higher = temporal head behaves like near-identity (doesn't integrate)\n\n")
        f.write(f"{'layer':<7}" + "  ".join(f"h{h:<6}" for h in range(H2))
                + f"  {'mean':<10}\n")
        for lyr in range(num_layers):
            d = diag_per_head_per_layer[lyr]
            if d is None:
                continue
            row = f"{lyr:<7}"
            for h in range(H2):
                row += f"{d[h]:.4f}  "
            row += f"{d.mean():<10.4f}\n"
            f.write(row)
        f.write("\n")

        # ── 3. Distance distribution ──
        f.write("-" * 70 + "\n")
        f.write("3. ATTENTION DISTANCE DISTRIBUTION per head per layer\n")
        f.write("-" * 70 + "\n")
        f.write(f"Split |i-j| into 5 bins: [very-short, short, medium, long, very-long]\n")
        f.write("Each row sums to ~1. Concentration in bin 0 = mostly short-range.\n\n")
        for lyr in range(num_layers):
            d = dist_per_head_per_layer[lyr]
            if d is None:
                continue
            f.write(f"layer {lyr}:\n")
            for h in range(H2):
                pct = [f"{v*100:5.1f}%" for v in d[h]]
                f.write(f"  head {h}: " + " | ".join(pct) + "\n")
            f.write("\n")

        # ── 4. Head diversity ──
        f.write("-" * 70 + "\n")
        f.write("4. HEAD PAIRWISE CORRELATION per layer\n")
        f.write("-" * 70 + "\n")
        f.write("Values near 1 = heads are redundant. Near 0 = heads specialize.\n\n")
        for lyr in range(num_layers):
            c = corr_per_layer[lyr]
            if c is None:
                continue
            # Take average of off-diagonal entries
            mask = ~np.eye(H2, dtype=bool)
            offdiag_mean = float(c[mask].mean())
            f.write(f"layer {lyr}: mean off-diagonal correlation = {offdiag_mean:+.3f}\n")
            f.write(f"  matrix:\n")
            for r in range(H2):
                line = "    " + "  ".join(f"{c[r, k]:+.3f}" for k in range(H2))
                f.write(line + "\n")
            f.write("\n")

        # ── 5. Temporal covariance regime ──
        f.write("-" * 70 + "\n")
        f.write("5. PER-CHANNEL TEMPORAL COVARIANCE REGIME\n")
        f.write("-" * 70 + "\n")
        f.write(f"N×N covariance of temporal features within each channel.\n")
        f.write("Does it show SPREAD regime like the spatial covariance?\n")
        f.write("If yes, a temporal Riemannian bias (analogous to C1) might help.\n\n")
        f.write(f"{'layer':<7}{'eig_min':<12}{'eig_max':<12}{'cond':<12}"
                f"{'‖T-I‖_F':<12}{'regime':<30}\n")
        for lyr in range(num_layers):
            stats = cov_stats_per_layer[lyr]
            if stats is None:
                continue
            eig_min, eig_max, cond, frob = stats
            if eig_min > 0.2 and eig_max < 5:
                regime = "NARROW (bias unlikely to help)"
            elif eig_max / max(eig_min, 1e-12) > 1e3 or frob > 10:
                regime = "SPREAD (temporal bias plausible)"
            else:
                regime = "INTERMEDIATE"
            f.write(f"{lyr:<7}{eig_min:<12.4f}{eig_max:<12.4f}{cond:<12.1f}"
                    f"{frob:<12.2f}{regime:<30}\n")

        f.write("\nEND OF ANALYSIS\n")

    print(f"[temporal] Done. See {args.output}")


if __name__ == "__main__":
    main()
