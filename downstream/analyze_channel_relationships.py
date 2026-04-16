"""
Analyze what the trained encoder has learned about channel relationships.

Produces multiple views of channel structure from a pretrained checkpoint:

1. Channel embedding similarity matrix
     → cosine similarity between the learned (144, D) channel embedding vectors.
       Tells us which channels the model considers "similar" in its identity
       representation. If anatomically-proximate channels cluster, the model
       recovered spatial structure without being told about it.

2. 2D projection of channel embeddings
     → PCA + t-SNE, colored by anatomical region (parsed from channel name
       prefix: F=frontal, C=central, P=parietal, O=occipital, T=temporal, Fp=frontopolar,
       AF/FC/CP/PO=intermediates). Useful as a paper figure.

3. Learned μ structure per layer (if C3 whitening is enabled)
     → visualizes the tangent-space reference μ^(l). Should converge toward the
       log-Euclidean Fréchet mean of covariances. Magnitude across layers should
       roughly match the diagnostic's ‖mean logS‖_F values.

4. Average attention matrix per layer over a small sample
     → spatial head attention weights averaged over samples. Shows which
       channels actually attend to which during inference.

Outputs PNG figures + a summary text file.

Usage:
    python -m downstream.analyze_channel_relationships \
        --checkpoint path/to/your.ckpt \
        --dataset bci_comp_2a \
        --output_dir downstream/results/channel_analysis
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
import yaml


# ────────────────────────────────────────────────────────────────
# Anatomical region classification from channel name
# ────────────────────────────────────────────────────────────────

# Standard 10-20 / 10-10 channel prefixes → coarse region.
# Picks the longest matching prefix, so "Fp1" matches "Fp" not "F".
REGION_PREFIX = [
    ("Fp",  "frontopolar"),
    ("AF",  "antero-frontal"),
    ("AFF", "antero-frontal"),
    ("AFp", "antero-frontal"),
    ("FT",  "fronto-temporal"),
    ("FTT", "fronto-temporal"),
    ("FC",  "fronto-central"),
    ("FCC", "fronto-central"),
    ("F",   "frontal"),
    ("CP",  "centro-parietal"),
    ("CCP", "centro-parietal"),
    ("CPP", "centro-parietal"),
    ("C",   "central"),
    ("TP",  "temporo-parietal"),
    ("TPP", "temporo-parietal"),
    ("T",   "temporal"),
    ("PO",  "parieto-occipital"),
    ("POO", "parieto-occipital"),
    ("PPO", "parieto-occipital"),
    ("P",   "parietal"),
    ("OI",  "occipital"),
    ("O",   "occipital"),
    ("M",   "mastoid"),
    ("I",   "inion"),
]

def classify_region(ch_name: str) -> str:
    """Return coarse anatomical region for a channel name like 'Fp1' or 'CPP6h'."""
    # Strip trailing h/z/number for matching
    letters = re.match(r"[A-Za-z]+", ch_name).group(0) if re.match(r"[A-Za-z]+", ch_name) else ""
    # Longest prefix match
    best = ("unknown", 0)
    for pfx, region in REGION_PREFIX:
        if letters.startswith(pfx) and len(pfx) > best[1]:
            best = (region, len(pfx))
    return best[0]


# ────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────

def load_channel_map(path: str) -> dict:
    """channel_name → global_index (from channel_info.yaml)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["channels_mapping"]


def analyze_embeddings(channel_embeds: torch.Tensor,
                       name_to_idx: dict,
                       output_dir: str,
                       top_k: int = 5):
    """Compute and save channel embedding similarity analysis."""
    import matplotlib.pyplot as plt

    # Keep only channels whose embedding has moved from zero init.
    # Unused channel slots (never sampled during pretraining) stay at zero.
    norms = channel_embeds.norm(dim=-1)
    active_mask = norms > 1e-6
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    print(f"[analyze] Active channels (non-zero embedding): {active_mask.sum().item()}/144")

    # Reverse map: global_index → name
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    active_names = [idx_to_name.get(i.item(), f"ch{i.item()}") for i in active_idx]
    active_regions = [classify_region(n) for n in active_names]

    # ── Cosine similarity matrix (active channels only) ──
    E = channel_embeds[active_idx]                       # (n_active, D)
    E_norm = torch.nn.functional.normalize(E, dim=-1)
    sim = (E_norm @ E_norm.T).numpy()                    # (n_active, n_active)

    # Save numeric matrix
    np.save(os.path.join(output_dir, "channel_embedding_cosine.npy"),
            {"names": active_names, "regions": active_regions, "sim": sim},
            allow_pickle=True)

    # ── Plot 1: similarity heatmap grouped by region ──
    # Sort channels by region for clearer block structure
    order = sorted(range(len(active_names)),
                   key=lambda i: (active_regions[i], active_names[i]))
    sim_sorted = sim[np.ix_(order, order)]
    sorted_names = [active_names[i] for i in order]
    sorted_regions = [active_regions[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_yticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
    ax.set_yticklabels(sorted_names, fontsize=5)

    # Annotate region boundaries
    region_starts = {}
    for i, r in enumerate(sorted_regions):
        region_starts.setdefault(r, i)
    for r, start in region_starts.items():
        end = max(i for i, x in enumerate(sorted_regions) if x == r)
        ax.axhline(end + 0.5, color="black", lw=0.3)
        ax.axvline(end + 0.5, color="black", lw=0.3)
        ax.text(-2, (start + end) / 2, r, ha="right", va="center",
                fontsize=7, rotation=0)

    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Learned channel-embedding cosine similarity, grouped by region")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "channel_embedding_similarity.png"), dpi=150)
    plt.close()
    print(f"[analyze] Saved channel_embedding_similarity.png")

    # ── Plot 2: 2D projection (PCA + t-SNE) ──
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        pca = PCA(n_components=2).fit_transform(E.numpy())
        tsne = TSNE(n_components=2, perplexity=min(30, len(E) - 1),
                    init="pca", random_state=42).fit_transform(E.numpy())

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        unique_regions = sorted(set(active_regions))
        cmap = plt.cm.tab20
        region_color = {r: cmap(i / len(unique_regions)) for i, r in enumerate(unique_regions)}

        for projection, ax, title in [(pca, axes[0], "PCA"), (tsne, axes[1], "t-SNE")]:
            for r in unique_regions:
                mask = [rr == r for rr in active_regions]
                ax.scatter(projection[mask, 0], projection[mask, 1],
                           c=[region_color[r]], label=r, s=50, alpha=0.8,
                           edgecolors="black", linewidths=0.5)
            for i, name in enumerate(active_names):
                ax.annotate(name, projection[i], fontsize=5, alpha=0.7)
            ax.set_title(f"Channel embeddings ({title})")
            ax.legend(fontsize=7, loc="best", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "channel_embedding_2d.png"), dpi=150)
        plt.close()
        print(f"[analyze] Saved channel_embedding_2d.png")
    except ImportError:
        print("[analyze] sklearn not available, skipping 2D projection")

    # ── Top-k nearest neighbours per channel ──
    # Useful for eyeballing: "what does the model think is similar to Cz?"
    summary_path = os.path.join(output_dir, "channel_nearest_neighbors.txt")
    with open(summary_path, "w") as f:
        f.write("Top nearest neighbours per channel (cosine similarity)\n")
        f.write("=" * 60 + "\n\n")
        for i, (name, region) in enumerate(zip(active_names, active_regions)):
            # Exclude self (diagonal)
            row = sim[i].copy()
            row[i] = -np.inf
            top = np.argsort(-row)[:top_k]
            nbrs = [(active_names[j], active_regions[j], row[j]) for j in top]
            f.write(f"{name:<8} [{region}]\n")
            for n_name, n_region, s in nbrs:
                same = "*" if n_region == region else " "
                f.write(f"  {same} {n_name:<8} [{n_region}]   sim={s:.3f}\n")
            f.write("\n")
    print(f"[analyze] Saved channel_nearest_neighbors.txt")

    # Fraction of top-1 neighbors that share the same region (simple metric)
    same_region_top1 = 0
    for i, region in enumerate(active_regions):
        row = sim[i].copy()
        row[i] = -np.inf
        nearest = int(np.argmax(row))
        if active_regions[nearest] == region:
            same_region_top1 += 1
    frac = same_region_top1 / len(active_regions)
    print(f"[analyze] Fraction of channels whose nearest neighbor is in the same region: {frac:.2%}")
    return frac


def analyze_mu(encoder, output_dir: str, name_to_idx: dict):
    """Visualize learned μ^(l) matrices per layer (if C3 whitening is enabled)."""
    import matplotlib.pyplot as plt

    mu_layers = []
    for i, layer in enumerate(encoder):
        mu = getattr(layer.attn.riemannian_bias, 'mu_log', None)
        if mu is None:
            continue
        mu_layers.append((i, mu.detach().cpu()))

    if not mu_layers:
        print("[analyze] No learnable μ in encoder — skipping μ analysis "
              "(likely a pre-whitening checkpoint)")
        return

    # Index into submatrix of active channels for a cleaner view
    norms_pre_layer_0 = None  # will compute after loop

    n = len(mu_layers)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel() if n > 1 else [axes]

    for ax, (layer_idx, mu) in zip(axes, mu_layers):
        # Symmetrize for display
        mu_sym = 0.5 * (mu + mu.T)
        vmax = max(mu_sym.abs().max().item(), 1e-6)
        im = ax.imshow(mu_sym.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"μ layer {layer_idx}  ‖μ‖_F={mu_sym.norm().item():.2f}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(mu_layers):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mu_per_layer.png"), dpi=120)
    plt.close()
    print(f"[analyze] Saved mu_per_layer.png with {n} layers")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze channel relationships learned by the encoder."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="downstream/results/channel_analysis")
    parser.add_argument("--channel_info", type=str,
                        default="downstream/info_dataset/channel_info.yaml")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # ── Extract channel embeddings ──
    # Key could be channel_embedding.channel_transformation.weight (downstream)
    # or channel_embedding_e.channel_transformation.weight (pretraining EncoderDecoder)
    candidates = [
        "channel_embedding.channel_transformation.weight",
        "channel_embedding_e.channel_transformation.weight",
    ]
    channel_embeds = None
    for key in candidates:
        if key in state_dict:
            channel_embeds = state_dict[key]
            print(f"[analyze] Loaded channel embeddings from '{key}', shape {tuple(channel_embeds.shape)}")
            break
    if channel_embeds is None:
        # Fuzzy match
        for k, v in state_dict.items():
            if "channel_embedding" in k and "weight" in k and v.dim() == 2 and v.shape[0] == 144:
                channel_embeds = v
                print(f"[analyze] Loaded channel embeddings from '{k}', shape {tuple(v.shape)}")
                break
    if channel_embeds is None:
        raise RuntimeError("Could not find channel embedding weight in checkpoint.")

    # ── Load name → index map ──
    name_to_idx = load_channel_map(args.channel_info)
    print(f"[analyze] Loaded {len(name_to_idx)} channel name mappings")

    # ── Embedding analysis ──
    frac = analyze_embeddings(channel_embeds, name_to_idx, args.output_dir)

    # ── Optional μ analysis (requires building the model to find μ_log params) ──
    # If the checkpoint has mu_log.* keys we can plot them directly without building the model.
    mu_keys = [k for k in state_dict.keys() if "mu_log" in k]
    if mu_keys:
        import matplotlib.pyplot as plt
        mu_layers = []
        for k in sorted(mu_keys):
            # Extract layer index from key path heuristically
            match = re.search(r"encoder\.(\d+)", k)
            if match:
                layer_idx = int(match.group(1))
            else:
                layer_idx = len(mu_layers)
            mu_layers.append((layer_idx, state_dict[k].detach().cpu()))
        mu_layers.sort(key=lambda x: x[0])

        n = len(mu_layers)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.ravel() if n > 1 else [axes]

        for ax, (layer_idx, mu) in zip(axes, mu_layers):
            mu_sym = 0.5 * (mu + mu.T)
            vmax = max(mu_sym.abs().max().item(), 1e-6)
            im = ax.imshow(mu_sym.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"μ layer {layer_idx}  ‖μ‖_F={mu_sym.norm().item():.2f}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        for ax in axes[len(mu_layers):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "mu_per_layer.png"), dpi=120)
        plt.close()
        print(f"[analyze] Saved mu_per_layer.png (found {n} μ matrices in checkpoint)")
    else:
        print("[analyze] No μ_log keys in checkpoint — checkpoint predates whitening")

    # ── Final summary ──
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Channel relationship analysis summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Active channel count: (non-zero embedding norm)\n\n")
        f.write(f"Same-region top-1 neighbor fraction: {frac:.2%}\n")
        f.write("  (higher = embeddings recovered anatomical structure implicitly)\n\n")
        f.write("Outputs:\n")
        for fname in sorted(os.listdir(args.output_dir)):
            f.write(f"  {fname}\n")
    print(f"[analyze] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
