"""
Diagnostic: inspect per-layer covariances produced by the trained encoder.

Answers the question "would Fréchet-mean / whitening help my model?" by
measuring whether the covariances S^(l) operate in a regime where Padé [1,1]
at identity is accurate (narrow eigenvalue range, Fréchet mean ≈ I) or
whether there's headroom from a different reference point.

Per layer, reports:
    eig_min, eig_max   — global eigenvalue range of S across samples
    cond_mean          — mean condition number (max/min eig per sample)
    frob_S_minus_I     — ⟨‖S − I‖_F⟩ — how far S drifts from identity on average
    frob_logS_mean     — ‖mean_n log(S_n)‖_F — distance of LE Fréchet mean from I
    between_var        — per-entry variance of log(S) across samples

Interpretation cheat-sheet:
    NARROW regime (eig_min ≳ 0.2, eig_max ≲ 5, frob_logS_mean small)
        → Padé at identity is accurate, whitening buys little. Ship as-is.
    SPREAD regime (eig span orders of magnitude, frob_logS_mean large)
        → Padé error grows; a learned reference (Fréchet-style) would help.
    HETEROGENEOUS regime (low within-sample, high between-sample variance)
        → Per-sample whitening (not global Fréchet) is the right add.

Usage:
    python -m downstream.diagnose_covariance_regime \
        --checkpoint path/to/your.ckpt \
        --dataset bci_comp_2a \
        --num_batches 10
"""
import argparse
import os
import sys

import numpy as np
import torch

from downstream.get_benchmark_foundation_model import (
    DATASET_CONFIGS,
    build_riemann_transformer_para,
)
from downstream.split_data_downstream import DownstreamDataLoader
from downstream.downstream_dataset import Downstream_Dataset


# ───────────────────────────────────────────────────────────────
# Diagnostics
# ───────────────────────────────────────────────────────────────

def analyze_layer(S_all: torch.Tensor, layer_idx: int) -> dict:
    """Compute per-layer stats from captured covariances.

    Args:
        S_all: (N_samples, C, C) float32 tensor of per-timestep covariances
        layer_idx: layer index for logging only
    Returns:
        dict of scalar stats
    """
    S_all = S_all.double()           # eig computations prefer float64 for stability
    N, C, _ = S_all.shape
    eye = torch.eye(C, dtype=S_all.dtype)

    # Eigenvalues per sample (symmetric assumed)
    eigvals = torch.linalg.eigvalsh(S_all)         # (N, C), ascending

    eig_min = eigvals.min().item()
    eig_max = eigvals.max().item()
    eig_mean = eigvals.mean().item()
    cond_per_sample = eigvals[:, -1] / eigvals[:, 0].clamp_min(1e-12)
    cond_mean = cond_per_sample.mean().item()
    cond_p95 = torch.quantile(cond_per_sample, 0.95).item()

    # ‖S − I‖_F per sample, averaged
    frob_S_minus_I = (S_all - eye).reshape(N, -1).norm(dim=-1).mean().item()

    # log-Euclidean log map via eigendecomposition (exact, offline cost ok).
    # log(S) = U · diag(log λ) · U^T
    eigvalsh, eigvecs = torch.linalg.eigh(S_all)   # (N, C), (N, C, C)
    log_eigvals = torch.log(eigvalsh.clamp_min(1e-12))
    log_S = torch.einsum('nce,ne,nde->ncd', eigvecs, log_eigvals, eigvecs)

    # LE Fréchet mean: μ = exp(mean_n log(S_n)). We only need ‖mean_n log(S_n)‖_F.
    mean_log_S = log_S.mean(dim=0)                # (C, C)
    frob_logS_mean = mean_log_S.norm().item()

    # Per-entry variance of log(S) across samples
    between_var = log_S.var(dim=0).mean().item()

    return {
        "layer": layer_idx,
        "n_samples": N,
        "C": C,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "eig_mean": eig_mean,
        "cond_mean": cond_mean,
        "cond_p95": cond_p95,
        "frob_S_minus_I": frob_S_minus_I,
        "frob_logS_mean": frob_logS_mean,
        "between_var": between_var,
    }


def classify_regime(stats: dict) -> str:
    """Heuristic regime classifier based on per-layer stats."""
    eig_spread = stats["eig_max"] / max(stats["eig_min"], 1e-12)
    if stats["eig_min"] > 0.2 and stats["eig_max"] < 5.0 \
            and stats["frob_logS_mean"] < 2.0:
        return "NARROW (Padé@I is fine — whitening unlikely to help)"
    if eig_spread > 1e3 or stats["frob_logS_mean"] > 10.0:
        return "SPREAD (Padé error likely — whitening would help)"
    if stats["between_var"] > 1.0:
        return "HETEROGENEOUS (per-sample whitening > global Fréchet)"
    return "INTERMEDIATE (mild drift; marginal whitening gains)"


# ───────────────────────────────────────────────────────────────
# Capture hooks
# ───────────────────────────────────────────────────────────────

def install_capture_hooks(model) -> list:
    """Monkey-patch each layer's AdaptiveLogMap to record S before log.
    Returns a list (one entry per encoder layer) of lists of tensors."""
    captured = [[] for _ in range(len(model.encoder))]

    for layer_idx, layer in enumerate(model.encoder):
        log_module = layer.attn.riemannian_bias.adaptive_log
        original_forward = log_module.forward

        def make_wrapped(orig, idx):
            def wrapped(S, channel_idx):
                captured[idx].append(S.detach().cpu())
                return orig(S, channel_idx)
            return wrapped

        log_module.forward = make_wrapped(original_forward, layer_idx)

    return captured


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose the covariance regime of a trained encoder "
                    "to decide whether whitening / Fréchet reference would help."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained .ckpt")
    parser.add_argument("--dataset", type=str, default="bci_comp_2a",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_batches", type=int, default=10,
                        help="How many batches to feed for statistics")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]

    # ── Build model via the same builder used for actual runs ──
    model = build_riemann_transformer_para(
        num_classes=ds_cfg["num_classes"],
        checkpoint_path=args.checkpoint,
        num_channels=ds_cfg["num_channels"],
        data_length=ds_cfg["data_length"],
    )
    model = model.to(args.device).eval()

    # Install hooks AFTER the model is built, so we capture the actual
    # forward path (bias module is inside the encoder layers).
    captured = install_capture_hooks(model)

    # ── Data loader: pull a few batches, no training, no shuffling ──
    loader = DownstreamDataLoader(
        data_path=ds_cfg["data_path"],
        config=ds_cfg["config_yaml"],
        custom_dataset_class=Downstream_Dataset,
        norm_mode=args.model if hasattr(args, "model") else "riemann_para",
        base_sfreq=ds_cfg["sampling_rate"],
    )
    train_ds, _, _ = loader.get_data_for_population()

    # Simple batched iteration — no DataLoader wrapper needed for offline stats
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[diagnose] dataset={args.dataset} C={ds_cfg['num_channels']} "
          f"num_batches={args.num_batches} batch_size={args.batch_size}")

    # ── Forward passes with hooks active ──
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if batch_idx >= args.num_batches:
                break
            # Dataset returns (data, label, channel_id)
            x, _, channel_list = batch
            x = x.to(args.device)
            channel_list = channel_list.to(args.device)
            # Forward through the Downstream model (encoder + head)
            _ = model(x, channel_list)
            if batch_idx == 0:
                for i, layer_cap in enumerate(captured):
                    if layer_cap:
                        print(f"[diagnose] layer {i} first-batch S shape: "
                              f"{tuple(layer_cap[0].shape)}")

    # ── Analyze ──
    print()
    header = (f"{'L':<3}{'N':<8}{'C':<4}"
              f"{'eig_min':<10}{'eig_max':<10}{'eig_mean':<10}"
              f"{'cond':<8}{'cond_p95':<10}"
              f"{'‖S-I‖_F':<10}{'‖mean logS‖_F':<14}{'between_var':<12}"
              f"regime")
    print(header)
    print("-" * len(header))

    all_stats = []
    for layer_idx, S_list in enumerate(captured):
        if not S_list:
            print(f"{layer_idx:<3}   — no captures —")
            continue
        S_all = torch.cat(S_list, dim=0).float()
        stats = analyze_layer(S_all, layer_idx)
        regime = classify_regime(stats)
        all_stats.append(stats)
        print(f"{stats['layer']:<3}{stats['n_samples']:<8}{stats['C']:<4}"
              f"{stats['eig_min']:<10.3f}{stats['eig_max']:<10.3f}"
              f"{stats['eig_mean']:<10.3f}"
              f"{stats['cond_mean']:<8.1f}{stats['cond_p95']:<10.1f}"
              f"{stats['frob_S_minus_I']:<10.2f}"
              f"{stats['frob_logS_mean']:<14.3f}{stats['between_var']:<12.3f}"
              f"{regime}")

    # ── Overall verdict ──
    print()
    if not all_stats:
        print("No statistics collected — check that the encoder actually ran.")
        return

    regimes = [classify_regime(s) for s in all_stats]
    narrow = sum(1 for r in regimes if r.startswith("NARROW"))
    spread = sum(1 for r in regimes if r.startswith("SPREAD"))
    hetero = sum(1 for r in regimes if r.startswith("HETEROGENEOUS"))

    print("─" * 60)
    print(f"Overall: {narrow}/{len(all_stats)} layers NARROW | "
          f"{spread}/{len(all_stats)} SPREAD | "
          f"{hetero}/{len(all_stats)} HETEROGENEOUS")
    if narrow >= len(all_stats) - 1:
        print("Verdict: current Padé@I setup is well-matched to your data. "
              "Whitening / Fréchet reference unlikely to yield measurable gains.")
    elif spread >= 1:
        print("Verdict: at least one layer operates in a SPREAD regime. "
              "A per-sample or per-layer reference point (Log-Euclidean mean) "
              "may provide measurable accuracy gains. Consider as future work.")
    elif hetero >= 1:
        print("Verdict: HETEROGENEOUS between-sample variance detected. "
              "A per-sample reference (not a global Fréchet) is the right "
              "direction if you pursue whitening.")
    else:
        print("Verdict: intermediate regime. Whitening gains likely small; "
              "the paper's current Padé@I choice is defensible.")


if __name__ == "__main__":
    main()
