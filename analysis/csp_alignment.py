"""
CSP Alignment Analysis
======================

Measures how closely the spatial attention filters learned by the Riemannian
transformer align with classical Common Spatial Patterns (CSP) filters.

CSP is the gold standard for motor imagery BCI — it finds spatial filters
that maximize variance ratio between classes. If our Riemannian attention
implicitly rediscovers CSP-like structure WITHOUT hand-engineering, that's
a compelling interpretability result.

How it works:
    1. Load a trained checkpoint + a labeled downstream dataset (e.g., BCI 2a)
    2. Compute classical CSP filters from the raw EEG data
    3. Extract the effective spatial filters from Riemannian attention:
       - The attention bias is L = log(S) where S is the covariance
       - The per-head scales weight these biases
       - We extract the top eigenvectors of the average attention bias
         per class — these are the learned "spatial filters"
    4. Measure alignment: cosine similarity between CSP filters and
       the learned spatial filter eigenvectors

Usage:
    python analysis/csp_alignment.py \
        --checkpoint path/to/checkpoint.ckpt \
        --data_config path/to/config.yaml \
        --num_csp_pairs 3 \
        --output analysis/csp_alignment_results.npz
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh
from einops import rearrange
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_csp_filters(X_class1, X_class2, n_pairs=3):
    """
    Compute classical CSP spatial filters.

    CSP finds projections W that maximize:
        W^T @ Sigma_1 @ W / (W^T @ (Sigma_1 + Sigma_2) @ W)

    This is solved via generalized eigenvalue problem:
        Sigma_1 @ w = lambda * (Sigma_1 + Sigma_2) @ w

    Args:
        X_class1: (n_trials_1, C, T) — EEG trials for class 1
        X_class2: (n_trials_2, C, T) — EEG trials for class 2
        n_pairs:  number of CSP filter pairs (top + bottom eigenvalues)

    Returns:
        W_csp: (C, 2*n_pairs) — CSP spatial filters (columns)
    """
    # Compute per-class average covariance
    def avg_cov(X):
        # X: (n_trials, C, T)
        n = X.shape[0]
        covs = []
        for i in range(n):
            x = X[i]  # (C, T)
            x = x - x.mean(axis=-1, keepdims=True)
            cov = (x @ x.T) / x.shape[-1]
            cov /= np.trace(cov)  # trace-normalize for numerical stability
            covs.append(cov)
        return np.mean(covs, axis=0)

    Sigma1 = avg_cov(X_class1)
    Sigma2 = avg_cov(X_class2)
    Sigma_total = Sigma1 + Sigma2

    # Solve generalized eigenvalue problem
    # Sigma1 @ W = Lambda @ Sigma_total @ W
    eigenvalues, eigenvectors = eigh(Sigma1, Sigma_total)

    # CSP filters: take n_pairs from each end (most discriminative)
    # Columns of eigenvectors are sorted ascending by eigenvalue
    W_csp = np.concatenate([
        eigenvectors[:, :n_pairs],        # lowest eigenvalues (favor class 2)
        eigenvectors[:, -n_pairs:]         # highest eigenvalues (favor class 1)
    ], axis=1)

    return W_csp, eigenvalues


def extract_riemannian_spatial_filters(model, dataloader, device, num_layers=None):
    """
    Extract effective spatial filters from the Riemannian attention branch.

    For each encoder layer, we:
    1. Run data through the model up to that layer
    2. Collect the Riemannian attention bias L = log(S) per sample
    3. Average L per class
    4. Eigendecompose the class-average L to get spatial filter directions

    Args:
        model:      trained downstream or pretrained model
        dataloader: labeled dataloader yielding (eeg, channel_list, labels)
        device:     torch device
        num_layers: how many layers to analyze (None = all)

    Returns:
        filters_per_layer: dict of {layer_idx: {class_label: eigenvectors}}
        biases_per_layer:  dict of {layer_idx: {class_label: avg_bias_matrix}}
    """
    model.eval()
    model.to(device)

    # Determine number of encoder layers
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'module') and hasattr(model.module, 'encoder'):
        encoder = model.module.encoder
    else:
        raise ValueError("Cannot find encoder in model")

    n_layers = len(encoder) if num_layers is None else min(num_layers, len(encoder))

    # Collect biases per class per layer
    # We hook into the riemannian_bias module to capture the bias output
    biases_by_class = {l: {} for l in range(n_layers)}
    hook_storage = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is either bias (B*N, H, C, C) or (bias, S) if return_covariance
            if isinstance(output, tuple):
                bias = output[0]
            else:
                bias = output
            # Average over heads and store: (B*N, C, C)
            hook_storage[layer_idx] = bias.mean(dim=1).detach().cpu()
        return hook_fn

    # Register hooks on riemannian_bias modules
    hooks = []
    for i in range(n_layers):
        layer = encoder[i]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'riemannian_bias'):
            h = layer.attn.riemannian_bias.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Run data through model
    all_labels = []
    all_biases = {l: [] for l in range(n_layers)}

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                eeg, channel_list, labels = batch
            elif len(batch) == 2:
                eeg, channel_list = batch
                labels = torch.zeros(eeg.shape[0])  # dummy
            else:
                continue

            eeg = eeg.to(device)
            if isinstance(channel_list, torch.Tensor):
                channel_list = channel_list.to(device)

            # Forward pass (triggers hooks)
            try:
                _ = model(eeg, channel_list)
            except Exception:
                # If model needs different args, try common variants
                try:
                    _ = model(eeg)
                except Exception:
                    continue

            all_labels.append(labels.cpu())

            B = eeg.shape[0]
            C = eeg.shape[1]

            for l in range(n_layers):
                if l in hook_storage:
                    # hook_storage[l] is (B*N, C, C), reshape to (B, N, C, C)
                    bias = hook_storage[l]
                    N = bias.shape[0] // B
                    bias = bias.view(B, N, C, C)
                    # Average over time steps: (B, C, C)
                    bias_avg = bias.mean(dim=1)
                    all_biases[l].append(bias_avg)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Organize by class
    all_labels = torch.cat(all_labels, dim=0).numpy()
    unique_classes = np.unique(all_labels)

    filters_per_layer = {}
    biases_per_layer = {}

    for l in range(n_layers):
        if not all_biases[l]:
            continue
        bias_all = torch.cat(all_biases[l], dim=0).numpy()  # (total_samples, C, C)

        filters_per_layer[l] = {}
        biases_per_layer[l] = {}

        for cls in unique_classes:
            cls_mask = all_labels == cls
            if cls_mask.sum() == 0:
                continue

            # Average bias matrix for this class
            avg_bias = bias_all[cls_mask].mean(axis=0)  # (C, C)
            biases_per_layer[l][cls] = avg_bias

            # Eigendecompose to get spatial filter directions
            eigvals, eigvecs = np.linalg.eigh(avg_bias)
            # Sort descending by absolute eigenvalue
            order = np.argsort(np.abs(eigvals))[::-1]
            filters_per_layer[l][cls] = eigvecs[:, order]

    return filters_per_layer, biases_per_layer


def compute_alignment(W_csp, W_learned, n_compare=None):
    """
    Compute alignment between CSP filters and learned spatial filters.

    Alignment = average absolute cosine similarity between the top
    CSP filters and the top learned eigenvectors.

    Args:
        W_csp:     (C, n_csp) — CSP filter columns
        W_learned: (C, C) — learned eigenvectors (sorted by importance)
        n_compare: number of filters to compare (default: n_csp)

    Returns:
        alignment_matrix: (n_compare, n_compare) — pairwise |cos_sim|
        mean_alignment:   scalar — average of best matches
    """
    if n_compare is None:
        n_compare = W_csp.shape[1]

    W_csp_sub = W_csp[:, :n_compare]
    W_learned_sub = W_learned[:, :n_compare]

    # Normalize columns
    W_csp_norm = W_csp_sub / (np.linalg.norm(W_csp_sub, axis=0, keepdims=True) + 1e-8)
    W_learned_norm = W_learned_sub / (np.linalg.norm(W_learned_sub, axis=0, keepdims=True) + 1e-8)

    # Pairwise absolute cosine similarity
    alignment_matrix = np.abs(W_csp_norm.T @ W_learned_norm)  # (n_compare, n_compare)

    # Best match for each CSP filter (greedy)
    best_matches = []
    for i in range(n_compare):
        best_j = alignment_matrix[i].argmax()
        best_matches.append(alignment_matrix[i, best_j])

    mean_alignment = np.mean(best_matches)

    return alignment_matrix, mean_alignment


def run_full_analysis(model, dataloader, device, n_csp_pairs=3, num_layers=4):
    """
    Full CSP alignment analysis pipeline.

    Args:
        model:       trained model with Riemannian attention
        dataloader:  labeled dataloader (eeg, channel_list, labels)
        device:      torch device
        n_csp_pairs: number of CSP filter pairs
        num_layers:  number of encoder layers to analyze

    Returns:
        results: dict with alignment scores per layer, per class pair
    """
    print("=" * 60)
    print("CSP ALIGNMENT ANALYSIS")
    print("=" * 60)

    # Step 1: Collect all data for CSP computation
    print("\n[Step 1] Collecting data for CSP computation...")
    all_eeg = []
    all_labels = []
    for batch in dataloader:
        if len(batch) == 3:
            eeg, _, labels = batch
        elif len(batch) == 2:
            eeg, labels = batch
        else:
            continue
        all_eeg.append(eeg.numpy() if isinstance(eeg, torch.Tensor) else eeg)
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    all_eeg = np.concatenate(all_eeg, axis=0)      # (N_total, C, T)
    all_labels = np.concatenate(all_labels, axis=0)  # (N_total,)
    unique_classes = np.unique(all_labels)
    n_classes = len(unique_classes)
    print(f"   Data: {all_eeg.shape[0]} trials, {all_eeg.shape[1]} channels, "
          f"{n_classes} classes: {unique_classes}")

    # Step 2: Compute CSP filters for each class pair
    print("\n[Step 2] Computing classical CSP filters...")
    csp_filters = {}
    for i, c1 in enumerate(unique_classes):
        for c2 in unique_classes[i + 1:]:
            X1 = all_eeg[all_labels == c1]
            X2 = all_eeg[all_labels == c2]
            W, evals = compute_csp_filters(X1, X2, n_pairs=n_csp_pairs)
            csp_filters[(c1, c2)] = W
            print(f"   CSP({int(c1)} vs {int(c2)}): eigenvalue range "
                  f"[{evals[0]:.4f}, {evals[-1]:.4f}]")

    # Step 3: Extract learned spatial filters from Riemannian attention
    print(f"\n[Step 3] Extracting spatial filters from {num_layers} layers...")
    filters_per_layer, biases_per_layer = extract_riemannian_spatial_filters(
        model, dataloader, device, num_layers=num_layers
    )

    # Step 4: Compute alignment
    print("\n[Step 4] Computing CSP ↔ Riemannian alignment...")
    results = {}
    for layer_idx in sorted(filters_per_layer.keys()):
        results[layer_idx] = {}
        for (c1, c2), W_csp in csp_filters.items():
            # For each class pair, compare CSP filters with learned filters
            # Use the learned filters from the class that has highest variance ratio
            # (i.e., the class CSP was designed to separate)
            if c1 in filters_per_layer[layer_idx]:
                W_learned = filters_per_layer[layer_idx][c1]
                alignment_matrix, mean_align = compute_alignment(
                    W_csp, W_learned, n_compare=2 * n_csp_pairs
                )
                results[layer_idx][(c1, c2)] = {
                    'alignment_matrix': alignment_matrix,
                    'mean_alignment': mean_align,
                }
                print(f"   Layer {layer_idx}, classes ({int(c1)} vs {int(c2)}): "
                      f"mean alignment = {mean_align:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for layer_idx in sorted(results.keys()):
        alignments = [v['mean_alignment'] for v in results[layer_idx].values()]
        if alignments:
            avg = np.mean(alignments)
            print(f"   Layer {layer_idx}: avg alignment = {avg:.4f}")

    print("\nInterpretation:")
    print("   0.0-0.3 = low alignment (model uses different spatial structure)")
    print("   0.3-0.6 = moderate alignment (partial CSP-like structure)")
    print("   0.6-1.0 = high alignment (model implicitly rediscovers CSP)")
    print()

    return results, csp_filters, filters_per_layer, biases_per_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSP Alignment Analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--num_csp_pairs", type=int, default=3,
                        help="Number of CSP filter pairs")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of encoder layers to analyze")
    parser.add_argument("--output", type=str, default="csp_alignment_results.npz",
                        help="Output file for results")
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"CSP pairs:  {args.num_csp_pairs}")
    print(f"Layers:     {args.num_layers}")
    print()
    print("NOTE: You need to provide a labeled dataloader for your downstream task.")
    print("See run_full_analysis() for the expected interface.")
    print("Example integration:")
    print()
    print("    from analysis.csp_alignment import run_full_analysis")
    print("    results = run_full_analysis(model, test_loader, device)")
