"""
Analyze the coupling between spatial covariance similarity and temporal
attention redundancy.

Core hypothesis for C2 (Riemannian Luna temporal compression):
    Token pairs with similar spatial covariance (close on SPD manifold)
    also have similar temporal attention patterns. If true, this justifies
    compressing tokens by covariance regime — tokens that the model treats
    identically in attention can be merged without information loss.

Produces:
    1. Scatter plot: SPD distance vs attention cosine similarity
         → each dot is a token pair (t_i, t_j) at a given layer.
           Negative correlation = covariance proximity drives attention redundancy.

    2. Per-layer Pearson/Spearman correlation statistics
         → quantifies how tightly SPD distance predicts attention overlap.

    3. Binned analysis: mean attention similarity within SPD distance bins
         → shows the functional relationship (monotonic decay = good motivation).

    4. Summary text file with all statistics.

Usage:
    python -m downstream.analyze_covariance_attention_coupling \
        --checkpoint path/to/your.ckpt \
        --dataset bci_comp_2a \
        --num_batches 5 \
        --output_dir downstream/results/cov_attn_coupling
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy import stats as sp_stats

from downstream.get_benchmark_foundation_model import (
    DATASET_CONFIGS,
    build_riemann_transformer_para,
)
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.downstream_dataset import Downstream_Dataset


# ────────────────────────────────────────────────────────────────
# Hook infrastructure (reused from temporal analysis)
# ────────────────────────────────────────────────────────────────

def install_hooks(model):
    """Capture x_norm AND raw residual per layer."""
    captures = defaultdict(lambda: {"x_norm": [], "residual": []})

    def make_pre_hook(layer_idx):
        def hook(module, inputs):
            # inputs[0] = x_norm, inputs[2] = residual (or None)
            captures[layer_idx]["x_norm"].append(inputs[0].detach().cpu())
            if len(inputs) > 2 and inputs[2] is not None:
                captures[layer_idx]["residual"].append(inputs[2].detach().cpu())
            else:
                captures[layer_idx]["residual"].append(inputs[0].detach().cpu())
        return hook

    handles = []
    for i, layer in enumerate(model.encoder):
        handles.append(layer.attn.register_forward_pre_hook(make_pre_hook(i)))
    return captures, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ────────────────────────────────────────────────────────────────
# Core computations
# ────────────────────────────────────────────────────────────────

def compute_temporal_attention(x_norm, attn_module, C):
    """Recompute temporal attention weights from captured x_norm.

    Returns:
        attn: (B*C, H2, N, N) temporal attention weights
    """
    with torch.no_grad():
        B, L_tok, D = x_norm.shape
        N = L_tok // C
        H = attn_module.num_heads
        H2 = attn_module.heads_per_branch
        d = attn_module.dim_head

        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(B, L_tok, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        q_t = q[:, :H2]
        k_t = k[:, :H2]

        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)

        score = (q_t @ k_t.transpose(-2, -1)) / (d ** 0.5)
        attn = score.softmax(dim=-1)
    return attn


def compute_spatial_log_covariance(x_residual, attn_module, C, channel_idx):
    """Compute per-timestep log(S_t) using the model's own Riemannian bias.

    x_residual: (B, N*C, D) — raw residual stream (pre-LayerNorm)
    Returns:
        L: (B*N, C, C) tangent vectors (same as what C1 produces)
    """
    with torch.no_grad():
        B, L_tok, D = x_residual.shape
        N = L_tok // C

        x_space = rearrange(x_residual, 'b (n c) d -> (b n) c d', c=C)

        # Use the model's own Riemannian bias module to compute log(S)
        riem_bias = attn_module.riemannian_bias
        eps = riem_bias.eps

        x_f32 = x_space.float()
        S = torch.bmm(x_f32, x_f32.transpose(-2, -1)) / D
        eye = torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)
        S = S + eps * eye

        # Padé [1,1] log map (reuse adaptive log)
        L = riem_bias.adaptive_log(S, channel_idx)  # (B*N, C, C)

        # Apply mu centering if present (same as forward pass)
        if riem_bias.mu_log is not None:
            mu_sub = riem_bias.mu_log[channel_idx][:, channel_idx]
            mu_sub = 0.5 * (mu_sub + mu_sub.transpose(-2, -1))
            L = L - mu_sub.unsqueeze(0)

    return L  # (B*N, C, C)


def compute_pairwise_spd_distance(L, B, N):
    """Compute pairwise Frobenius distance between token log-covariances.

    L: (B*N, C, C)
    Returns:
        dist: (B, N, N) — ‖L_i - L_j‖_F for each pair within each sample
    """
    with torch.no_grad():
        L_reshaped = L.reshape(B, N, -1).float()  # (B, N, C*C)
        # Pairwise Frobenius distance using expansion trick
        L_sqnorm = (L_reshaped ** 2).sum(dim=-1)  # (B, N)
        cross = torch.bmm(L_reshaped, L_reshaped.transpose(-2, -1))  # (B, N, N)
        dist_sq = L_sqnorm.unsqueeze(-1) + L_sqnorm.unsqueeze(-2) - 2 * cross
        dist = dist_sq.clamp(min=0).sqrt()
    return dist  # (B, N, N)


def compute_attention_cosine_similarity(attn_t, B, C, N, H2):
    """Compute pairwise cosine similarity of attention patterns.

    For each token pair (t_i, t_j), compare their attention row vectors
    (averaged over channels and heads).

    attn_t: (B*C, H2, N, N)
    Returns:
        attn_sim: (B, N, N) — cosine similarity of attention patterns
    """
    with torch.no_grad():
        # Average over heads: (B*C, N, N)
        attn_avg = attn_t.mean(dim=1)
        # Average over channels: (B*C, N, N) → (B, C, N, N) → (B, N, N)
        attn_avg = rearrange(attn_avg, '(b c) n1 n2 -> b c n1 n2', b=B, c=C)
        attn_avg = attn_avg.mean(dim=1)  # (B, N, N)

        # Each row attn_avg[b, i, :] is token i's attention distribution over time
        # Cosine similarity between row i and row j
        attn_norm = F.normalize(attn_avg, dim=-1)  # (B, N, N) — normalize rows
        attn_sim = torch.bmm(attn_norm, attn_norm.transpose(-2, -1))  # (B, N, N)

    return attn_sim


# ────────────────────────────────────────────────────────────────
# Analysis
# ────────────────────────────────────────────────────────────────

def analyze_coupling(spd_dist, attn_sim, n_bins=10):
    """Compute coupling statistics between SPD distance and attention similarity.

    Args:
        spd_dist: (B, N, N) pairwise SPD distances
        attn_sim: (B, N, N) pairwise attention cosine similarities

    Returns:
        dict with pearson_r, spearman_r, binned_means, binned_stds, etc.
    """
    B, N, _ = spd_dist.shape

    # Extract upper triangle (exclude diagonal — self-similarity is trivially 1)
    triu_idx = torch.triu_indices(N, N, offset=1)
    dist_flat = spd_dist[:, triu_idx[0], triu_idx[1]].reshape(-1).numpy()
    sim_flat = attn_sim[:, triu_idx[0], triu_idx[1]].reshape(-1).numpy()

    # Remove any NaN
    valid = np.isfinite(dist_flat) & np.isfinite(sim_flat)
    dist_flat = dist_flat[valid]
    sim_flat = sim_flat[valid]

    # Pearson and Spearman correlation
    pearson_r, pearson_p = sp_stats.pearsonr(dist_flat, sim_flat)
    spearman_r, spearman_p = sp_stats.spearmanr(dist_flat, sim_flat)

    # Binned analysis: mean attention similarity per SPD distance bin
    bin_edges = np.percentile(dist_flat, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # include max
    binned_means = np.zeros(n_bins)
    binned_stds = np.zeros(n_bins)
    binned_counts = np.zeros(n_bins, dtype=int)
    bin_centers = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (dist_flat >= bin_edges[i]) & (dist_flat < bin_edges[i + 1])
        binned_counts[i] = mask.sum()
        if mask.sum() > 0:
            binned_means[i] = sim_flat[mask].mean()
            binned_stds[i] = sim_flat[mask].std()
            bin_centers[i] = dist_flat[mask].mean()

    # Temporal distance control: do adjacent tokens (|i-j|=1) have smaller SPD distance?
    adj_dist = spd_dist[:, triu_idx[0], triu_idx[1]]
    token_dist = (triu_idx[1] - triu_idx[0]).float().unsqueeze(0).expand(B, -1)
    adj_mask = (token_dist == 1).reshape(-1).numpy()
    far_mask = (token_dist > 5).reshape(-1).numpy()
    adj_spd_mean = dist_flat[adj_mask[valid]].mean() if adj_mask[valid].sum() > 0 else float('nan')
    far_spd_mean = dist_flat[far_mask[valid]].mean() if far_mask[valid].sum() > 0 else float('nan')

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "binned_means": binned_means,
        "binned_stds": binned_stds,
        "binned_counts": binned_counts,
        "bin_centers": bin_centers,
        "n_pairs": len(dist_flat),
        "dist_flat": dist_flat,
        "sim_flat": sim_flat,
        "adj_spd_mean": adj_spd_mean,
        "far_spd_mean": far_spd_mean,
    }


# ────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────

def plot_coupling(results_per_layer, output_dir):
    """Generate coupling plots per layer."""
    import matplotlib.pyplot as plt

    num_layers = len(results_per_layer)

    # ── Plot 1: Scatter (subsample for visibility) + regression line ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for lyr, res in enumerate(results_per_layer):
        if res is None:
            axes[lyr].set_title(f"Layer {lyr}: no data")
            continue

        ax = axes[lyr]
        # Subsample for scatter (too many points otherwise)
        n = len(res["dist_flat"])
        idx = np.random.choice(n, min(5000, n), replace=False)
        ax.scatter(res["dist_flat"][idx], res["sim_flat"][idx],
                   alpha=0.15, s=3, c="steelblue", rasterized=True)

        # Regression line
        slope, intercept = np.polyfit(res["dist_flat"], res["sim_flat"], 1)
        x_line = np.array([res["dist_flat"].min(), res["dist_flat"].max()])
        ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2,
                label=f"r={res['pearson_r']:.3f}")

        ax.set_xlabel("SPD distance ‖log(S_i)−log(S_j)‖_F")
        ax.set_ylabel("Attention cosine similarity")
        ax.set_title(f"Layer {lyr}  r={res['pearson_r']:.3f}  ρ={res['spearman_r']:.3f}")
        ax.legend(fontsize=8)

    for ax in axes[num_layers:]:
        ax.axis("off")
    plt.suptitle("SPD Distance vs Temporal Attention Similarity", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_spd_vs_attn.png"), dpi=150)
    plt.close()

    # ── Plot 2: Binned relationship (cleaner view) ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for lyr, res in enumerate(results_per_layer):
        if res is None:
            axes[lyr].set_title(f"Layer {lyr}: no data")
            continue

        ax = axes[lyr]
        centers = res["bin_centers"]
        means = res["binned_means"]
        stds = res["binned_stds"]

        ax.errorbar(centers, means, yerr=stds, fmt='o-', capsize=3,
                     color="steelblue", markersize=5)
        ax.set_xlabel("SPD distance (bin center)")
        ax.set_ylabel("Mean attention cosine similarity")
        ax.set_title(f"Layer {lyr}  r={res['pearson_r']:.3f}")

    for ax in axes[num_layers:]:
        ax.axis("off")
    plt.suptitle("Binned: SPD Distance → Attention Similarity", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "binned_spd_vs_attn.png"), dpi=150)
    plt.close()

    # ── Plot 3: Correlation summary across layers ──
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = []
    pearson_vals = []
    spearman_vals = []
    for lyr, res in enumerate(results_per_layer):
        if res is not None:
            layers.append(lyr)
            pearson_vals.append(res["pearson_r"])
            spearman_vals.append(res["spearman_r"])

    ax.plot(layers, pearson_vals, 'o-', label="Pearson r", color="steelblue")
    ax.plot(layers, spearman_vals, 's--', label="Spearman ρ", color="coral")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation")
    ax.set_title("SPD distance ↔ Attention similarity correlation per layer")
    ax.legend()
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_per_layer.png"), dpi=150)
    plt.close()

    print(f"[coupling] Saved plots to {output_dir}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze coupling between spatial covariance and temporal attention."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bci_comp_2a",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str,
                        default="downstream/results/cov_attn_coupling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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

    print(f"[coupling] Running {args.num_batches} batches on {args.dataset}…")
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

    # Get channel_idx (same for all samples in a dataset)
    if hasattr(train_ds, 'channel_list'):
        channel_idx = torch.tensor(train_ds.channel_list, dtype=torch.long)
    else:
        channel_idx = torch.arange(C, dtype=torch.long)

    # ── Per-layer analysis ──
    print("[coupling] Computing per-layer coupling statistics…")
    results_per_layer = []

    for lyr in range(num_layers):
        xn_list = captures[lyr]["x_norm"]
        res_list = captures[lyr]["residual"]

        if not xn_list:
            results_per_layer.append(None)
            continue

        x_norm = torch.cat(xn_list, dim=0)
        x_residual = torch.cat(res_list, dim=0)
        B = x_norm.size(0)
        N = x_norm.size(1) // C

        print(f"  Layer {lyr}: B={B}, N={N}, C={C}")

        # 1. Temporal attention patterns
        attn_t = compute_temporal_attention(x_norm, model.encoder[lyr].attn, C)
        H2 = attn_t.size(1)

        # 2. Spatial log-covariance per timestep
        L = compute_spatial_log_covariance(
            x_residual, model.encoder[lyr].attn, C, channel_idx
        )  # (B*N, C, C)

        # 3. Pairwise SPD distance between tokens
        spd_dist = compute_pairwise_spd_distance(L, B, N)  # (B, N, N)

        # 4. Pairwise attention cosine similarity
        attn_sim = compute_attention_cosine_similarity(attn_t, B, C, N, H2)

        # 5. Analyze coupling
        res = analyze_coupling(spd_dist, attn_sim)
        results_per_layer.append(res)

        print(f"    Pearson r={res['pearson_r']:.4f} (p={res['pearson_p']:.2e}), "
              f"Spearman ρ={res['spearman_r']:.4f}")
        print(f"    Adjacent-token SPD distance: {res['adj_spd_mean']:.4f}, "
              f"Far-token (>5 apart): {res['far_spd_mean']:.4f}")

    # ── Plots ──
    try:
        plot_coupling(results_per_layer, args.output_dir)
    except ImportError:
        print("[coupling] matplotlib not available, skipping plots")

    # ── Summary text file ──
    summary_path = os.path.join(args.output_dir, "coupling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COVARIANCE–ATTENTION COUPLING ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset:    {args.dataset}\n")
        f.write(f"Batches:    {args.num_batches} × {args.batch_size}\n\n")

        f.write("Hypothesis: Token pairs with similar spatial covariance\n")
        f.write("(small ‖log(S_i) - log(S_j)‖_F) have similar temporal\n")
        f.write("attention patterns (high cosine similarity).\n")
        f.write("→ Negative Pearson r supports this hypothesis.\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Layer':<8} {'Pearson r':<12} {'p-value':<12} "
                f"{'Spearman ρ':<12} {'n_pairs':<10} "
                f"{'adj SPD':<10} {'far SPD':<10}\n")
        f.write("-" * 70 + "\n")

        for lyr, res in enumerate(results_per_layer):
            if res is None:
                f.write(f"{lyr:<8} {'N/A':<12}\n")
                continue
            f.write(f"{lyr:<8} {res['pearson_r']:<12.4f} {res['pearson_p']:<12.2e} "
                    f"{res['spearman_r']:<12.4f} {res['n_pairs']:<10} "
                    f"{res['adj_spd_mean']:<10.4f} {res['far_spd_mean']:<10.4f}\n")

        f.write("\n")
        f.write("Binned analysis (SPD distance → mean attention similarity):\n")
        f.write("-" * 70 + "\n")

        for lyr, res in enumerate(results_per_layer):
            if res is None:
                continue
            f.write(f"\nLayer {lyr}:\n")
            f.write(f"  {'Bin center':<15} {'Mean sim':<12} {'Std':<12} {'Count':<8}\n")
            for i in range(len(res["binned_means"])):
                f.write(f"  {res['bin_centers'][i]:<15.4f} "
                        f"{res['binned_means'][i]:<12.4f} "
                        f"{res['binned_stds'][i]:<12.4f} "
                        f"{res['binned_counts'][i]:<8}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("• Negative Pearson r: tokens with similar covariance have\n")
        f.write("  similar attention → covariance proximity predicts temporal\n")
        f.write("  attention redundancy. SUPPORTS Luna SPD-biased compression.\n\n")
        f.write("• Adjacent-token SPD < far-token SPD: temporal neighbors are\n")
        f.write("  also covariance neighbors, confirming neurophysiology prior\n")
        f.write("  (brain states are locally smooth at 64ms granularity).\n\n")
        f.write("• Monotonically decreasing binned means: the relationship is\n")
        f.write("  functional, not just statistical → SPD distance is a valid\n")
        f.write("  metric for grouping tokens for compression.\n")

    print(f"[coupling] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
