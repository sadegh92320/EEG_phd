"""
Embedding vs Riemannian Bias Analysis
======================================

Decomposes the spatial attention logits into:
  1. Q·K / √d  — driven by channel embedding + learned features
  2. riem_bias  — Riemannian covariance bias (C1)

Measures the relative magnitude and influence of each component
on the final attention pattern.

If the Riemannian bias dominates, the channel embedding is less important.
If Q·K dominates, the covariance bias is just a small correction.

Also measures: how much does removing each component change the
attention distribution (KL divergence)?

Usage:
    from analysis.embedding_vs_bias import run_embedding_vs_bias
    results = run_embedding_vs_bias(model, dataloader, device, num_layers=8)
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_embedding_vs_bias(model, dataloader, device, num_layers=None,
                           max_batches=None):
    """
    Measure relative contribution of Q·K (channel embedding) vs
    Riemannian bias to the spatial attention pattern.

    For each layer, captures:
      - qk_score: Q·K / √d  (before adding riem_bias)
      - riem_bias: the Riemannian covariance bias
      - full_score: qk_score + riem_bias (before softmax)

    Metrics per layer:
      - Magnitude ratio: ||riem_bias||_F / ||qk_score||_F
      - Attention contribution: KL(full_attn || qk_only_attn)
        (how much does removing riem_bias change attention?)
      - Attention contribution: KL(full_attn || bias_only_attn)
        (how much does removing Q·K change attention?)
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

    # We need to hook into the attention module to capture Q·K and riem_bias
    # separately. The spatial attention computes:
    #   score = Q·K^T / √d + riem_bias
    #   attn = softmax(score)
    #
    # We'll monkey-patch the forward to intercept these intermediate values.
    # But we learned that monkey-patching is fragile. Instead, we hook the
    # riemannian_bias module to get riem_bias, and hook the full attention
    # to capture the Q·K part.
    #
    # Strategy: hook riemannian_bias output for the bias, then hook the
    # attention module to capture score components using a custom wrapper.

    # Storage for current batch
    bias_storage = {}     # layer → riem_bias tensor
    qk_storage = {}       # layer → qk_score tensor
    score_storage = {}    # layer → full score (before softmax)

    # We need to intercept lines 1602-1603 in the attention forward.
    # The cleanest way: temporarily wrap the softmax call to capture inputs.
    # Or: we can compute Q·K ourselves from the hook on the attention module.
    #
    # Actually, simplest: hook riemannian_bias for the bias (already works),
    # then hook the full layer to get input/output, and reconstruct Q·K
    # from the attention module's stored QKV.
    #
    # Even simpler: just patch the 3 lines to store intermediates.

    # Let's use a direct approach: register a hook on the attention module
    # that captures the pre-softmax score components.

    original_forwards = {}
    attn_modules = {}

    for i in range(n_layers):
        layer = encoder[i]
        attn_mod = layer.attn
        attn_modules[i] = attn_mod
        original_forwards[i] = attn_mod.forward

        def make_patched_forward(layer_idx, orig_forward, attn_ref):
            def patched_forward(*args, **kwargs):
                # We'll intercept the spatial attention score computation
                # by temporarily patching the inner computation.

                # Store the riem_bias via existing hook (below)
                # For Q·K, we need to capture inside the forward.

                # Approach: run original forward but also store intermediates
                # We know the structure: after riem_bias is computed and QKV
                # is split, lines 1602-1603 compute score.
                #
                # Instead of deep surgery, let's use a simpler trick:
                # run the forward twice — once normal, once without riem_bias
                # (by zeroing it). The difference tells us the contribution.
                #
                # Actually that changes softmax nonlinearity. Better:
                # just capture riem_bias and total score pre-softmax.

                # Let's store a flag so the riem_bias hook also stores values
                result = orig_forward(*args, **kwargs)
                return result
            return patched_forward

    # Simpler approach: just hook riemannian_bias for the bias magnitude,
    # and run a second forward with riem_bias zeroed to measure the
    # counterfactual attention.

    # Actually, the cleanest approach for measuring relative importance:
    # 1. Hook riemannian_bias to get bias magnitude
    # 2. Hook the spatial attention score computation
    #
    # For (2), we can't easily hook a specific line. But we CAN:
    # - Run model normally → get output
    # - Temporarily set riem_bias output to zero → run again → get output
    # - Compare attention patterns
    #
    # BUT that requires two forward passes per batch. Let's do it more
    # efficiently by hooking the right modules.

    # PLAN: For each layer's attention:
    # - Hook riemannian_bias to capture riem_bias (B*N, H, C, C)
    # - Replace the score computation to also store qk_score
    #
    # Since the score line is: score = (q_s @ k_s^T) / √d + riem_bias
    # We can capture both by storing riem_bias from hook, then computing
    # score - riem_bias = qk_score after the fact.
    #
    # But we don't have access to score... unless we hook it.
    #
    # Let's just patch the attention forward minimally:

    # Use **kwargs passthrough to avoid signature mismatches
    original_forwards = {}

    for i in range(n_layers):
        attn_mod = encoder[i].attn
        original_forwards[i] = attn_mod.forward

        def make_instrumented_forward(layer_idx, orig_fwd, attn_ref):
            def instrumented_forward(*args, **kwargs):
                """
                Wraps original forward, captures spatial Q·K and riem_bias
                components, then delegates to original for actual output.
                """
                from einops import rearrange

                # Extract x_norm and num_chan from args
                x_norm = args[0]
                num_chan = args[1] if len(args) > 1 else kwargs.get('num_chan')
                # Use residual if provided, else x_norm (downstream has no residual)
                residual = kwargs.get('residual', None)
                channel_idx = kwargs.get('channel_idx', None)
                if len(args) > 2 and channel_idx is None:
                    # might be positional
                    pass
                mask = kwargs.get('mask', None)

                B, L, D = x_norm.shape
                C = num_chan
                N = L // C
                H = attn_ref.num_heads
                H2 = H // 2
                d = D // H

                # Spatial reshape — use residual for Riemannian bias if available
                x_for_spd = residual if residual is not None else x_norm
                x_space = rearrange(x_for_spd, 'b (n c) d -> (b n) c d', c=C)
                if mask is not None:
                    mask_space = rearrange(mask, 'b (n c) -> (b n) c', c=C)
                    x_space = x_space * (~mask_space).unsqueeze(-1).float()
                else:
                    mask_space = None

                # Riemannian bias
                riem_bias, L_n = attn_ref.riemannian_bias(
                    x_space, channel_idx, mask_space=mask_space,
                )

                # QKV from normalized input
                qkv = attn_ref.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                q_s = q[:, H2:]
                k_s = k[:, H2:]

                # Spatial Q·K score
                q_s_r = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=C)
                k_s_r = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=C)

                with torch.amp.autocast('cuda', enabled=False), \
                     torch.amp.autocast('cpu', enabled=False):
                    qk_score = (q_s_r.float() @ k_s_r.float().transpose(-2, -1)) / (d ** 0.5)

                # Store for analysis
                bias_storage[layer_idx] = riem_bias.detach().cpu()
                qk_storage[layer_idx] = qk_score.detach().cpu()
                score_storage[layer_idx] = (qk_score + riem_bias.float()).detach().cpu()

                # Run the ORIGINAL forward for the actual output
                return orig_fwd(*args, **kwargs)

            return instrumented_forward

        attn_mod.forward = make_instrumented_forward(
            i, original_forwards[i], attn_mod
        )

    # ── Collect data ──
    # Running accumulators per layer
    sum_qk_norm = {l: 0.0 for l in range(n_layers)}
    sum_bias_norm = {l: 0.0 for l in range(n_layers)}
    sum_qk_mean = {l: 0.0 for l in range(n_layers)}
    sum_bias_mean = {l: 0.0 for l in range(n_layers)}
    sum_kl_no_bias = {l: 0.0 for l in range(n_layers)}    # KL(full || qk_only)
    sum_kl_no_qk = {l: 0.0 for l in range(n_layers)}      # KL(full || bias_only)
    sum_bias_fraction = {l: 0.0 for l in range(n_layers)}  # |bias| / (|bias| + |qk|)
    n_samples = {l: 0 for l in range(n_layers)}

    print(f"Running forward passes (max_batches={max_batches})...")

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
            except Exception as e:
                print(f"   Batch {batch_idx} failed: {e}")
                continue

            for l in range(n_layers):
                if l not in bias_storage:
                    continue

                rb = bias_storage[l].float()    # (B*N, H, C, C)
                qk = qk_storage[l].float()      # (B*N, H, C, C)
                full = score_storage[l].float()  # (B*N, H, C, C)

                BN = rb.shape[0]

                # Magnitude: Frobenius norm per sample, averaged
                rb_norm = rb.reshape(BN, -1).norm(dim=-1).mean().item()
                qk_norm = qk.reshape(BN, -1).norm(dim=-1).mean().item()

                sum_qk_norm[l] += qk_norm
                sum_bias_norm[l] += rb_norm

                # Mean absolute value
                sum_qk_mean[l] += qk.abs().mean().item()
                sum_bias_mean[l] += rb.abs().mean().item()

                # Fraction: how much of the total logit magnitude is from bias
                total_norm = qk_norm + rb_norm
                if total_norm > 0:
                    sum_bias_fraction[l] += rb_norm / total_norm

                # KL divergence: how much does removing each component change attention?
                # full_attn = softmax(qk + bias)
                # qk_only_attn = softmax(qk)
                # bias_only_attn = softmax(bias)
                full_attn = torch.softmax(full, dim=-1).clamp(min=1e-8)
                qk_attn = torch.softmax(qk, dim=-1).clamp(min=1e-8)
                bias_attn = torch.softmax(rb, dim=-1).clamp(min=1e-8)

                # KL(full || qk_only) — information lost by removing bias
                kl_no_bias = (full_attn * (full_attn.log() - qk_attn.log())).sum(dim=-1).mean().item()
                # KL(full || bias_only) — information lost by removing Q·K
                kl_no_qk = (full_attn * (full_attn.log() - bias_attn.log())).sum(dim=-1).mean().item()

                sum_kl_no_bias[l] += kl_no_bias
                sum_kl_no_qk[l] += kl_no_qk

                n_samples[l] += 1

            bias_storage.clear()
            qk_storage.clear()
            score_storage.clear()

            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1}")

    # Restore original forwards
    for i in range(n_layers):
        encoder[i].attn.forward = original_forwards[i]

    gc.collect()

    # ── Print results ──
    print()
    print("=" * 70)
    print("CHANNEL EMBEDDING (Q·K) vs RIEMANNIAN BIAS ANALYSIS")
    print("=" * 70)
    print()
    print(f"{'Layer':<7} {'||Q·K||':>9} {'||Bias||':>9} {'Bias%':>7} "
          f"{'KL-noBias':>10} {'KL-noQK':>10} {'Dominant':>10}")
    print("-" * 70)

    results = {}
    for l in range(n_layers):
        if n_samples[l] == 0:
            continue

        n = n_samples[l]
        qk_n = sum_qk_norm[l] / n
        bias_n = sum_bias_norm[l] / n
        bias_frac = sum_bias_fraction[l] / n * 100
        kl_nb = sum_kl_no_bias[l] / n
        kl_nq = sum_kl_no_qk[l] / n

        if kl_nb > kl_nq:
            dominant = "Riem Bias"
        elif kl_nq > kl_nb * 1.5:
            dominant = "Q·K (emb)"
        else:
            dominant = "Balanced"

        results[l] = {
            'qk_norm': qk_n,
            'bias_norm': bias_n,
            'bias_fraction_pct': bias_frac,
            'kl_removing_bias': kl_nb,
            'kl_removing_qk': kl_nq,
            'dominant': dominant,
        }

        print(f"{l:<7} {qk_n:>9.4f} {bias_n:>9.4f} {bias_frac:>6.1f}% "
              f"{kl_nb:>10.4f} {kl_nq:>10.4f} {dominant:>10}")

    print()
    print("INTERPRETATION:")
    print("   ||Q·K||, ||Bias||  = Frobenius norm of each logit component")
    print("   Bias%              = fraction of total logit magnitude from Riem bias")
    print("   KL-noBias          = KL(full_attn || qk_only) — cost of removing bias")
    print("   KL-noQK            = KL(full_attn || bias_only) — cost of removing Q·K")
    print("   Higher KL = that component contributes MORE to the final attention")
    print()
    print("   If Bias% increases through layers → model increasingly relies on")
    print("   Riemannian geometry over channel embedding for spatial routing")
    print()

    return results
