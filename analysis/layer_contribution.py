"""
Layer Contribution Analysis
============================

Tests whether each layer's learned transformation carries discriminative
spatial structure that gets diluted by the residual stream.

For each layer we compare:
  A) Covariance of FULL residual stream (x after residual add)
  B) Covariance of LAYER OUTPUT only (x_out - x_in, the residual delta)

If B has higher class separability than A, the layer learns useful
geometry that the residual dilutes.

Class separability = ratio of between-class to within-class covariance
distance (Fisher-like criterion in log-Euclidean SPD space).

Usage:
    from analysis.layer_contribution import run_layer_contribution
    results = run_layer_contribution(model, dataloader, device, num_layers=8)
"""

import torch
import numpy as np
import gc
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spd_from_tokens(x, num_chan, eps=1e-5):
    """
    Compute per-sample spatial covariance from token representations.

    x: (B, L, D) where L = N*C → reshape → covariance per sample.
    Returns: (B, C, C)
    """
    B, L, D = x.shape
    C = num_chan
    N = L // C

    # (B, N, C, D)
    x_r = x.reshape(B, N, C, D)
    # Average over time patches → (B, C, D)
    x_avg = x_r.mean(dim=1)
    # Center
    x_avg = x_avg - x_avg.mean(dim=-1, keepdim=True)
    # Covariance: (B, C, C)
    S = torch.bmm(x_avg, x_avg.transpose(-2, -1)) / D
    S = S + eps * torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)
    return S


def _log_euclidean(S):
    """log(S) for SPD matrix S: (C, C)."""
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _class_separability(covariances_by_class):
    """
    Fisher-like separability in log-Euclidean space.
    Between-class distance / within-class spread.
    """
    classes = sorted(covariances_by_class.keys())
    if len(classes) < 2:
        return 0.0

    log_means = {}
    log_samples = {}
    for c in classes:
        covs = covariances_by_class[c]
        logs = [_log_euclidean(s) for s in covs]
        log_means[c] = np.mean(logs, axis=0)
        log_samples[c] = logs

    overall_mean = np.mean([log_means[c] for c in classes], axis=0)

    between = 0.0
    for c in classes:
        between += np.linalg.norm(log_means[c] - overall_mean, 'fro') ** 2
    between /= len(classes)

    within = 0.0
    total_samples = 0
    for c in classes:
        for log_s in log_samples[c]:
            within += np.linalg.norm(log_s - log_means[c], 'fro') ** 2
            total_samples += 1
    within /= max(total_samples, 1)

    if within < 1e-10:
        return float('inf')
    return between / within


def run_layer_contribution(model, dataloader, device, num_layers=None,
                           max_batches=None, sample_cap=100):
    """
    Measure per-layer contribution to discriminative geometry.

    Uses register_forward_hook on each encoder layer to capture
    input and output tensors, then computes:
      - delta = output - input  (what the layer added)
      - covariance of delta vs covariance of output
      - Fisher separability for each

    Args:
        model:       trained downstream model
        dataloader:  yields (eeg, channel_ids, labels)
        device:      torch device
        num_layers:  layers to analyze (None = all)
        max_batches: limit batches
        sample_cap:  max samples per class
    """
    model.eval()
    model.to(device)

    if hasattr(model, 'encoder'):
        encoder = model.encoder
    elif hasattr(model, 'module') and hasattr(model.module, 'encoder'):
        encoder = model.module.encoder
    else:
        raise ValueError("Cannot find encoder")

    n_layers = len(encoder) if num_layers is None else min(num_layers, len(encoder))

    # ── Register forward hooks (captures input[0] and output) ──
    hook_data = {}  # layer_idx → {'input': tensor, 'output': tensor}

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            # inputs is a tuple; first arg is x (the token tensor)
            x_in = inputs[0].detach().cpu()
            x_out = output.detach().cpu()
            hook_data[layer_idx] = {
                'input': x_in,
                'output': x_out,
            }
        return hook_fn

    hooks = []
    for i in range(n_layers):
        h = encoder[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    # ── Collect per-class covariances ──
    class_covs_delta = {l: {} for l in range(n_layers)}
    class_covs_residual = {l: {} for l in range(n_layers)}
    samples_per_class = {}

    print(f"Running forward passes (max_batches={max_batches}, "
          f"sample_cap={sample_cap} per class)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                eeg, channel_ids, labels = batch
            elif len(batch) == 2:
                eeg, channel_ids = batch
                labels = torch.zeros(eeg.shape[0])
            else:
                continue

            eeg = eeg.to(device)
            if isinstance(channel_ids, torch.Tensor):
                channel_ids = channel_ids.to(device)

            # Check if we already have enough samples
            all_full = all(
                samples_per_class.get(int(l), 0) >= sample_cap
                for l in labels.numpy()
            ) if len(samples_per_class) > 0 else False
            if all_full:
                break

            try:
                _ = model(eeg, channel_ids)
            except Exception as e:
                print(f"   Forward failed at batch {batch_idx}: {e}")
                continue

            B = eeg.shape[0]
            num_chan = eeg.shape[1]
            labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

            for l in range(n_layers):
                if l not in hook_data:
                    continue

                x_in = hook_data[l]['input']    # (B, L, D)
                x_out = hook_data[l]['output']   # (B, L, D)
                delta = x_out - x_in             # what the layer added

                # Compute per-sample covariance
                S_delta = _spd_from_tokens(delta, num_chan).numpy()     # (B, C, C)
                S_residual = _spd_from_tokens(x_out, num_chan).numpy()  # (B, C, C)

                for b in range(B):
                    cls = int(labels_np[b])
                    if samples_per_class.get(cls, 0) >= sample_cap:
                        continue

                    if cls not in class_covs_delta[l]:
                        class_covs_delta[l][cls] = []
                        class_covs_residual[l][cls] = []

                    class_covs_delta[l][cls].append(S_delta[b])
                    class_covs_residual[l][cls].append(S_residual[b])

            # Update sample counts
            for b in range(B):
                cls = int(labels_np[b])
                samples_per_class[cls] = samples_per_class.get(cls, 0) + 1

            hook_data.clear()

            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1}, samples: {dict(samples_per_class)}")

    # Remove hooks
    for h in hooks:
        h.remove()
    gc.collect()

    # ── Compute separability ──
    print()
    print("=" * 60)
    print("LAYER CONTRIBUTION ANALYSIS")
    print("=" * 60)
    print()
    print(f"Samples per class: {samples_per_class}")
    print()
    print(f"{'Layer':<8} {'Layer Delta':>14} {'Full Residual':>14} {'Ratio':>10}")
    print("-" * 50)

    results = {}
    for l in range(n_layers):
        sep_delta = _class_separability(class_covs_delta[l])
        sep_residual = _class_separability(class_covs_residual[l])

        if sep_residual > 1e-10:
            ratio = sep_delta / sep_residual
        else:
            ratio = float('inf') if sep_delta > 0 else 0.0

        results[l] = {
            'layer_delta_separability': sep_delta,
            'full_residual_separability': sep_residual,
            'ratio': ratio,
        }

        print(f"{l:<8} {sep_delta:>14.4f} {sep_residual:>14.4f} {ratio:>10.2f}x")

    print()
    print("INTERPRETATION:")
    print("   Ratio > 1 = layer's own contribution is MORE class-discriminative")
    print("               than the full residual → residual DILUTES it")
    print("   Ratio < 1 = residual accumulation HELPS discrimination")
    print("   Ratio ~ 1 = layer contribution fully preserved in residual")
    print()

    ratios = [r['ratio'] for r in results.values()
              if 0 < r['ratio'] < float('inf')]
    if ratios:
        avg_ratio = np.mean(ratios)
        print(f"Average ratio: {avg_ratio:.2f}x")
        if avg_ratio > 1.5:
            print("=> STRONG evidence: layers learn discriminative geometry")
            print("   that gets diluted by residual. Consider decoupling")
            print("   SPD computation from residual stream.")
        elif avg_ratio > 1.0:
            print("=> MILD dilution: some loss, but mostly preserved.")
        else:
            print("=> Residual accumulation HELPS — current design is good.")

    return results
