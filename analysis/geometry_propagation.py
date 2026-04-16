"""
Geometry Propagation Analysis (Memory-Efficient)
=================================================

Measures how much Riemannian (SPD) structure is preserved across
transformer layers through the residual stream.

Uses streaming/running statistics to avoid storing all samples in memory.

Metrics per layer pair (k, k+1):
    1. Tangent vector correlation: Pearson r between flattened log-map
       outputs L_k and L_{k+1}, averaged over samples.
    2. Subspace overlap: average cosine similarity between top eigenvectors
       of the mean tangent at layers k and k+1.
    3. Geodesic distance: log-Euclidean distance between mean covariance
       matrices at consecutive layers.

Usage:
    from analysis.geometry_propagation import run_geometry_propagation
    results = run_geometry_propagation(model, dataloader, device, num_layers=8)
"""

import torch
import numpy as np
import gc
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_geometry_propagation(model, dataloader, device, num_layers=None,
                              max_batches=None, sample_cap=200):
    """
    Measure geometric information flow across layers (memory-efficient).

    Instead of storing all tangent vectors, we accumulate:
      - Running mean of tangent vectors per layer (for subspace overlap)
      - Per-sample correlation between consecutive layers (streaming)
      - Running mean of covariances per layer (for geodesic distance)

    Args:
        model:       trained model with Riemannian attention
        dataloader:  yields (eeg, channel_ids, labels)
        device:      torch device
        num_layers:  how many layers to analyze (None = all)
        max_batches: limit number of batches (None = all)
        sample_cap:  max samples for per-sample correlation (controls memory)

    Returns:
        dict with per-layer-pair metrics
    """
    model.eval()
    model.to(device)

    # Find encoder
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'module') and hasattr(model.module, 'encoder'):
        encoder = model.module.encoder
    else:
        raise ValueError("Cannot find encoder in model")

    n_layers = len(encoder) if num_layers is None else min(num_layers, len(encoder))

    # ── Hook storage — only keeps current batch ──
    tangent_storage = {}
    covariance_storage = {}

    def make_tangent_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                bias = output[0]
            else:
                bias = output
            # Average over heads → (B*N, C, C)
            tangent_storage[layer_idx] = bias.mean(dim=1).detach().cpu()
        return hook_fn

    def make_covariance_hook(layer_idx):
        def hook_fn(module, input, output):
            S = input[0].detach().cpu()
            covariance_storage[layer_idx] = S
        return hook_fn

    # Register hooks
    hooks = []
    for i in range(n_layers):
        layer = encoder[i]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'riemannian_bias'):
            rb = layer.attn.riemannian_bias
            h = rb.register_forward_hook(make_tangent_hook(i))
            hooks.append(h)
            if hasattr(rb, 'spd_log'):
                h2 = rb.spd_log.register_forward_hook(make_covariance_hook(i))
                hooks.append(h2)
            elif hasattr(rb, 'adaptive_log'):
                h2 = rb.adaptive_log.register_forward_hook(make_covariance_hook(i))
                hooks.append(h2)

    # ── Running accumulators ──
    # Running mean of tangent vectors per layer: sum_L[l] / count[l]
    sum_L = {}       # layer → (C, C) running sum
    sum_S = {}       # layer → (C, C) running sum of covariance
    count = {}       # layer → int

    # For per-sample correlation between consecutive layers,
    # we store a capped number of sample pairs
    sample_pairs = {l: {'curr': [], 'next': []} for l in range(n_layers - 1)}
    samples_collected = 0

    print(f"Running forward passes (max_batches={max_batches}, "
          f"sample_cap={sample_cap})...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                eeg, channel_ids, labels = batch
            elif len(batch) == 2:
                eeg, channel_ids = batch
            else:
                continue

            eeg = eeg.to(device)
            if isinstance(channel_ids, torch.Tensor):
                channel_ids = channel_ids.to(device)

            try:
                _ = model(eeg, channel_ids)
            except Exception:
                try:
                    _ = model(eeg)
                except Exception:
                    continue

            B = eeg.shape[0]
            N_tokens = eeg.shape[-1] // 16  # approximate

            # Process each layer's hook output
            for l in range(n_layers):
                if l not in tangent_storage:
                    continue

                L_batch = tangent_storage[l]  # (B*N, C, C)

                # Average over tokens within each sample: (B*N, C, C) → (B, C, C)
                BN = L_batch.shape[0]
                N_actual = BN // B
                L_per_sample = L_batch.reshape(B, N_actual, *L_batch.shape[1:])
                L_mean = L_per_sample.mean(dim=1)  # (B, C, C)

                # Update running mean
                if l not in sum_L:
                    sum_L[l] = torch.zeros_like(L_mean[0])
                    count[l] = 0
                sum_L[l] += L_mean.sum(dim=0)
                count[l] += B

                # Same for covariance if available
                if l in covariance_storage:
                    S_batch = covariance_storage[l]  # (B*N, C, C)
                    S_per_sample = S_batch.reshape(B, N_actual, *S_batch.shape[1:])
                    S_mean = S_per_sample.mean(dim=1)  # (B, C, C)
                    if l not in sum_S:
                        sum_S[l] = torch.zeros_like(S_mean[0])
                    sum_S[l] += S_mean.sum(dim=0)

                # Store sample pairs for correlation (capped)
                if l < n_layers - 1 and samples_collected < sample_cap:
                    n_to_store = min(B, sample_cap - samples_collected)
                    # Store flattened tangent for this layer
                    sample_pairs[l]['curr'].append(
                        L_mean[:n_to_store].reshape(n_to_store, -1).numpy()
                    )

            # Now fill in the 'next' side of pairs
            for l in range(n_layers - 1):
                l_next = l + 1
                if l_next in tangent_storage and samples_collected < sample_cap:
                    L_next_batch = tangent_storage[l_next]
                    BN = L_next_batch.shape[0]
                    N_actual = BN // B
                    L_next_mean = L_next_batch.reshape(
                        B, N_actual, *L_next_batch.shape[1:]
                    ).mean(dim=1)
                    n_to_store = min(B, sample_cap - samples_collected)
                    sample_pairs[l]['next'].append(
                        L_next_mean[:n_to_store].reshape(n_to_store, -1).numpy()
                    )

            samples_collected += B

            # Clear hook storage to free memory
            tangent_storage.clear()
            covariance_storage.clear()

            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1} done, "
                      f"{min(samples_collected, sample_cap)} samples collected")

    # Remove hooks
    for h in hooks:
        h.remove()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute metrics ──
    print()
    print("=" * 60)
    print("GEOMETRY PROPAGATION ANALYSIS")
    print("=" * 60)

    available_layers = sorted(sum_L.keys())
    if len(available_layers) < 2:
        print("Need at least 2 layers with data. Aborting.")
        return {}

    # Compute mean tangent per layer
    mean_L = {}
    mean_S = {}
    for l in available_layers:
        mean_L[l] = (sum_L[l] / count[l]).numpy()
        if l in sum_S:
            mean_S[l] = (sum_S[l] / count[l]).numpy()

    results = {}

    for i in range(len(available_layers) - 1):
        l_curr = available_layers[i]
        l_next = available_layers[i + 1]
        pair_key = f"layer_{l_curr}_to_{l_next}"
        results[pair_key] = {}

        # ── Metric 1: Per-sample tangent correlation ──
        if sample_pairs[l_curr]['curr'] and sample_pairs[l_curr]['next']:
            curr_flat = np.concatenate(sample_pairs[l_curr]['curr'], axis=0)
            next_flat = np.concatenate(sample_pairs[l_curr]['next'], axis=0)
            M = min(curr_flat.shape[0], next_flat.shape[0])
            curr_flat = curr_flat[:M]
            next_flat = next_flat[:M]

            correlations = []
            for s in range(M):
                r = np.corrcoef(curr_flat[s], next_flat[s])[0, 1]
                if not np.isnan(r):
                    correlations.append(r)
            mean_corr = np.mean(correlations) if correlations else 0.0
            std_corr = np.std(correlations) if correlations else 0.0
            results[pair_key]['tangent_correlation'] = mean_corr
            results[pair_key]['tangent_correlation_std'] = std_corr

        # ── Metric 2: Subspace overlap (from mean tangent) ──
        C = mean_L[l_curr].shape[0]
        eigvals_c, eigvecs_c = np.linalg.eigh(mean_L[l_curr])
        eigvals_n, eigvecs_n = np.linalg.eigh(mean_L[l_next])

        k = min(6, C)
        order_c = np.argsort(np.abs(eigvals_c))[::-1][:k]
        order_n = np.argsort(np.abs(eigvals_n))[::-1][:k]

        V_curr = eigvecs_c[:, order_c]
        V_next = eigvecs_n[:, order_n]

        overlap_matrix = np.abs(V_curr.T @ V_next)
        best_overlaps = overlap_matrix.max(axis=1)
        mean_overlap = np.mean(best_overlaps)
        results[pair_key]['subspace_overlap'] = mean_overlap

        # ── Metric 3: Log-Euclidean distance between mean covariances ──
        if l_curr in mean_S and l_next in mean_S:
            try:
                eigv_c, eigV_c = np.linalg.eigh(mean_S[l_curr])
                eigv_n, eigV_n = np.linalg.eigh(mean_S[l_next])
                eigv_c = np.maximum(eigv_c, 1e-8)
                eigv_n = np.maximum(eigv_n, 1e-8)
                logS_c = eigV_c @ np.diag(np.log(eigv_c)) @ eigV_c.T
                logS_n = eigV_n @ np.diag(np.log(eigv_n)) @ eigV_n.T
                d = np.linalg.norm(logS_c - logS_n, 'fro')
                results[pair_key]['logeuclidean_distance'] = d
            except Exception:
                results[pair_key]['logeuclidean_distance'] = float('nan')

    # ── Long-range: layer 0 vs layer L ──
    if len(available_layers) > 2:
        l_first = available_layers[0]
        l_last = available_layers[-1]
        lr_key = f"layer_{l_first}_to_{l_last}_longrange"
        results[lr_key] = {}

        # Subspace overlap
        eigvals_f, eigvecs_f = np.linalg.eigh(mean_L[l_first])
        eigvals_l, eigvecs_l = np.linalg.eigh(mean_L[l_last])
        k = min(6, mean_L[l_first].shape[0])
        order_f = np.argsort(np.abs(eigvals_f))[::-1][:k]
        order_l = np.argsort(np.abs(eigvals_l))[::-1][:k]
        V_first = eigvecs_f[:, order_f]
        V_last = eigvecs_l[:, order_l]
        overlap_lr = np.mean(np.abs(V_first.T @ V_last).max(axis=1))
        results[lr_key]['subspace_overlap'] = overlap_lr

        # Log-Euclidean distance
        if l_first in mean_S and l_last in mean_S:
            try:
                ev_f, eV_f = np.linalg.eigh(mean_S[l_first])
                ev_l, eV_l = np.linalg.eigh(mean_S[l_last])
                ev_f = np.maximum(ev_f, 1e-8)
                ev_l = np.maximum(ev_l, 1e-8)
                logS_f = eV_f @ np.diag(np.log(ev_f)) @ eV_f.T
                logS_l = eV_l @ np.diag(np.log(ev_l)) @ eV_l.T
                results[lr_key]['logeuclidean_distance'] = \
                    np.linalg.norm(logS_f - logS_l, 'fro')
            except Exception:
                pass

    # ── Print ──
    print("\nConsecutive layer pairs:")
    print("-" * 65)
    print(f"{'Pair':<20} {'Tangent r':>12} {'Subspace':>10} {'LogEuc d':>12}")
    print("-" * 65)

    for key in sorted(results.keys()):
        if 'longrange' in key:
            continue
        r = results[key]
        corr = r.get('tangent_correlation', float('nan'))
        corr_std = r.get('tangent_correlation_std', 0)
        overlap = r.get('subspace_overlap', float('nan'))
        dist = r.get('logeuclidean_distance', float('nan'))
        print(f"{key:<20} {corr:>7.4f}±{corr_std:<4.3f} {overlap:>10.4f} "
              f"{dist:>12.4f}")

    for key in sorted(results.keys()):
        if 'longrange' not in key:
            continue
        r = results[key]
        print(f"\nLong-range ({key}):")
        if 'subspace_overlap' in r:
            print(f"   Subspace overlap:      {r['subspace_overlap']:.4f}")
        if 'logeuclidean_distance' in r:
            print(f"   Log-Euclidean distance: {r['logeuclidean_distance']:.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("Tangent correlation (r):")
    print("   > 0.7  = strong — residual carries geometry forward")
    print("   0.3-0.7 = partial — layers refine but preserve geometry")
    print("   < 0.3  = weak — each layer computes independent geometry")
    print()
    print("Subspace overlap:")
    print("   > 0.8  = dominant spatial directions stable across layers")
    print("   < 0.5  = layers attend to different spatial subspaces")
    print()
    print("Log-Euclidean distance:")
    print("   Small = covariances evolve slowly (residual dominates)")
    print("   Large = covariances change substantially per layer")
    print()

    return results
