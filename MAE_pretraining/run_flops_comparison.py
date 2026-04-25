"""
FLOP comparison runner: counts FLOPs for our model and external baselines
(BIOT, LaBraM, STEEGformer) on a matched input shape.

Usage:
    python -m MAE_pretraining.run_flops_comparison

Each model is built via the downstream builder so its forward signature is
correctly matched. We instrument with the patched flops_counter module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager

from MAE_pretraining.flops_counter import (
    FLOPsCounter,
    _add_attention_flops_hook,
    _add_bmm_flops_hook,
    _add_matmul_flops_hook,
    _add_linalg_solve_hook,
    _add_linalg_eigh_hook,
    _add_fft_hooks,
    _add_einsum_hook,
    _format_number,
)


# ─────────────────────────────────────────────────────────────────────
# Standard input shape for fair comparison
# ─────────────────────────────────────────────────────────────────────
# bci_comp_2a-style input: 22 channels, 4 seconds at 128 Hz
NUM_CHANNELS = 22
SEQ_LENGTH = 512
BATCH_SIZE = 1


@torch.no_grad()
def count_model_flops(model, forward_fn, name="model"):
    """
    Counts FLOPs using the patched counter.

    Args:
        model: the model instance (already on cpu)
        forward_fn: callable that takes the model and runs one forward pass
                    e.g., lambda m: m(eeg, channel_list)
        name: human-readable name
    """
    model.eval()

    counter = FLOPsCounter()
    counter.register_hooks(model)

    original_sdpa = _add_attention_flops_hook(counter, model)
    original_bmm = _add_bmm_flops_hook(counter, model)
    original_matmul = _add_matmul_flops_hook(counter)
    original_solve = _add_linalg_solve_hook(counter)
    original_eigh = _add_linalg_eigh_hook(counter)
    original_fft, original_ifft, original_rfft, original_irfft = _add_fft_hooks(counter)
    original_einsum = _add_einsum_hook(counter)

    try:
        forward_fn(model)
    except Exception as e:
        print(f"  [warn] {name}: forward error ({type(e).__name__}: {e}), "
              f"FLOPs may be incomplete")
    finally:
        counter.remove_hooks()
        F.scaled_dot_product_attention = original_sdpa
        torch.bmm = original_bmm
        torch.matmul = original_matmul
        torch.linalg.solve = original_solve
        torch.linalg.eigh = original_eigh
        torch.fft.fft = original_fft
        torch.fft.ifft = original_ifft
        torch.fft.rfft = original_rfft
        torch.fft.irfft = original_irfft
        torch.einsum = original_einsum

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "name": name,
        "flops": counter.flops,
        "gflops": counter.flops / 1e9,
        "params_M": total_params / 1e6,
        "layer_breakdown": dict(counter.layer_flops),
    }


def build_our_model(patch_size=16, use_hilbert_target=False, learn_mu_reference=True,
                    disable_bias=False):
    """Build our pretraining model directly (not the downstream wrapper)."""
    from MAE_pretraining.bert_parallel_approx_riemann import ApproxAdaptiveRiemannBert
    return ApproxAdaptiveRiemannBert(
        config=None,
        num_channels=NUM_CHANNELS,
        enc_dim=512,
        depth_e=8,
        mask_prob=0.5,
        patch_size=patch_size,
        use_corr_masking=False,
        value_bias_layers=0,
        learn_mu_reference=learn_mu_reference,
        use_rope=False,
        mask_strategy='random',
        use_hilbert_target=use_hilbert_target,
        disable_bias=disable_bias,
    )


def build_external_model(model_name):
    """Build an external baseline via the downstream builder."""
    from downstream.get_benchmark_foundation_model import MODEL_BUILDERS
    builder = MODEL_BUILDERS[model_name]
    model = builder(
        num_classes=4,
        checkpoint_path=None,
        num_channels=NUM_CHANNELS,
        data_length=SEQ_LENGTH,
        base_sfreq=128,
        training_mode='linear_probe',
        config_yaml=None,
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# Forward functions per model (signatures differ)
# ─────────────────────────────────────────────────────────────────────

def forward_ours(model):
    """Our model: model(eeg, channel_list) where eeg is (B, C, T)."""
    eeg = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    channel_list = torch.arange(NUM_CHANNELS, dtype=torch.long)
    return model(eeg, channel_list)


def forward_steegformer(model):
    """STEEGformer downstream: model(x, channel_list)."""
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    channel_list = torch.arange(NUM_CHANNELS, dtype=torch.long)
    return model(x, channel_list)


def forward_biot(model):
    """BIOT: typically takes (B, C, T)."""
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    return model(x)


def forward_labram(model):
    """LaBraM: forward(x, input_chans) where x is (B, N_electrodes, n_patches, patch_size)."""
    # LaBraM uses patch_size=200 typically; check downstream wrapper handling
    # Pass standard input — wrapper should handle reshaping
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    channel_list = torch.arange(NUM_CHANNELS, dtype=torch.long)
    try:
        # Try the (x, channel_list) signature first (downstream wrapper)
        return model(x, channel_list)
    except TypeError:
        # Fall back to (x, input_chans) raw signature
        input_chans = list(range(NUM_CHANNELS))
        return model(x, input_chans)


# ─────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────

def run_comparison():
    print(f"\n{'='*80}")
    print(f"  FLOP Comparison — Input: B={BATCH_SIZE}, C={NUM_CHANNELS}, T={SEQ_LENGTH}")
    print(f"{'='*80}\n")

    results = []

    # ── Our model variants ──
    configs = [
        ("Ours (C1, no μ, p=16)", lambda: build_our_model(patch_size=16, learn_mu_reference=False, use_hilbert_target=False), forward_ours),
        ("Ours (C1+μ, p=16)", lambda: build_our_model(patch_size=16, learn_mu_reference=True, use_hilbert_target=False), forward_ours),
        ("Ours (Run 5: C1+μ+Hilbert, p=32)", lambda: build_our_model(patch_size=32, learn_mu_reference=True, use_hilbert_target=True), forward_ours),
        ("Ours (baseline, no Riemannian)", lambda: build_our_model(patch_size=16, disable_bias=True), forward_ours),
    ]
    for name, builder, fwd_fn in configs:
        try:
            model = builder()
            r = count_model_flops(model, fwd_fn, name=name)
            results.append(r)
            del model
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")

    # ── External baselines ──
    external = [
        ("BIOT", "biot", forward_biot),
        ("LaBraM", "labram", forward_labram),
        ("STEEGformer", "steegformer", forward_steegformer),
    ]
    for display_name, model_key, fwd_fn in external:
        try:
            model = build_external_model(model_key)
            r = count_model_flops(model, fwd_fn, name=display_name)
            results.append(r)
            del model
        except Exception as e:
            print(f"  [skip] {display_name}: {type(e).__name__}: {e}")

    # ── Print summary table ──
    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<40s} {'GFLOPs':>10s} {'Params (M)':>12s}")
    print(f"  {'-'*40} {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['name']:<40s} {r['gflops']:>10.2f} {r['params_M']:>12.2f}")
    print(f"{'='*80}\n")

    # ── Print per-layer breakdown for each ──
    for r in results:
        print(f"\n  Layer breakdown — {r['name']} ({r['gflops']:.2f} GFLOPs total)")
        print(f"  {'-'*60}")
        for layer_type, flops in sorted(r['layer_breakdown'].items(), key=lambda x: -x[1]):
            pct = 100 * flops / max(r['flops'], 1)
            print(f"    {layer_type:<35s} {_format_number(flops, 'FLOPs'):>14s}  ({pct:5.1f}%)")
        print()

    return results


if __name__ == "__main__":
    run_comparison()
