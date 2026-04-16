"""
Temporal Embedding Analysis
============================

Measures the contribution of temporal positional embeddings to:
1. Token representation magnitude (how visible is position in the token?)
2. Temporal attention structure (position-dependent vs content-driven?)
3. Counterfactual: attention with vs without temporal embedding

Metrics:
    - Magnitude ratio: ||temporal_embed|| / ||full_token||
    - Attention locality: correlation between attention weight and
      temporal distance (negative = local attention, zero = global)
    - Counterfactual KL: KL(normal_attn || no_temporal_attn)
      (how much does removing temporal embedding change attention?)

Usage:
    from analysis.temporal_embedding_analysis import run_temporal_analysis
    results = run_temporal_analysis(model, dataloader, device, num_layers=8)
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import sys, os
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_temporal_analysis(model, dataloader, device, num_layers=None,
                           max_batches=None):
    """
    Analyze temporal embedding contribution.
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

    # ══════════════════════════════════════════════════════════
    # Part 1: Magnitude analysis (no forward pass needed)
    # ══════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEMPORAL EMBEDDING ANALYSIS")
    print("=" * 70)

    # Get temporal embedding values for typical sequence length
    # The temporal embedding is sinusoidal with shape (1, max_len, D)
    temp_emb = model.temporal_embedding
    pe_buffer = temp_emb.pe  # (1, max_len, D)

    # For a typical 6s window at 128Hz with patch_size=16: N=48
    N_typical = 48
    pe_slice = pe_buffer[0, :N_typical, :]  # (N, D)
    pe_norm = pe_slice.norm(dim=-1).mean().item()

    print(f"\nPart 1: Embedding Magnitude")
    print("-" * 40)
    print(f"   Temporal embed norm (avg over {N_typical} positions): {pe_norm:.4f}")

    # ══════════════════════════════════════════════════════════
    # Part 2: Forward pass analysis — temporal attention patterns
    # ══════════════════════════════════════════════════════════

    # Hook temporal attention to capture Q·K scores
    # Temporal attention: q_t, k_t are (B*C, H, N, d)
    # We need to hook before F.scaled_dot_product_attention
    # Since that's a functional call, we need to intercept the attention module

    # Strategy: two forward passes per batch
    #   1. Normal forward → capture temporal attention patterns
    #   2. Forward with temporal embedding zeroed → compare

    # For pass 1, hook the temporal Q·K computation
    temporal_attn_storage = {}  # layer → (B*C, H, N, N) attention weights

    original_forwards = {}

    for i in range(n_layers):
        attn_mod = encoder[i].attn
        original_forwards[i] = attn_mod.forward

        def make_instrumented_forward(layer_idx, orig_fwd, attn_ref):
            def instrumented_forward(*args, **kwargs):
                x_norm = args[0]
                num_chan = args[1] if len(args) > 1 else kwargs.get('num_chan')

                B, L, D = x_norm.shape
                C = num_chan
                N = L // C
                H = attn_ref.num_heads
                H2 = H // 2
                d = D // H

                # QKV
                qkv = attn_ref.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Temporal branch Q, K
                q_t = q[:, :H2]
                k_t = k[:, :H2]
                q_t_r = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
                k_t_r = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)

                # Temporal attention scores (before softmax)
                with torch.amp.autocast('cuda', enabled=False), \
                     torch.amp.autocast('cpu', enabled=False):
                    t_score = (q_t_r.float() @ k_t_r.float().transpose(-2, -1)) / (d ** 0.5)
                    t_attn = t_score.softmax(dim=-1)

                temporal_attn_storage[layer_idx] = t_attn.detach().cpu()

                # Run original forward for actual output
                return orig_fwd(*args, **kwargs)
            return instrumented_forward

        attn_mod.forward = make_instrumented_forward(
            i, original_forwards[i], attn_mod
        )

    # ── Accumulators ──
    # Per-layer: locality score, magnitude ratios
    sum_locality = {l: 0.0 for l in range(n_layers)}
    sum_attn_entropy = {l: 0.0 for l in range(n_layers)}
    n_batches = {l: 0 for l in range(n_layers)}

    # For magnitude comparison: collect first batch's embeddings
    magnitude_collected = False
    patch_norm_val = 0.0
    chan_norm_val = 0.0
    temp_norm_val = 0.0

    # For counterfactual: we'll do a second pass with zeroed temporal embed
    sum_kl_no_temporal = {l: 0.0 for l in range(n_layers)}
    normal_attn_cache = {}

    print(f"\nPart 2: Forward pass analysis (max_batches={max_batches})...")

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

            B, C, T = eeg.shape

            # ── Collect magnitude info from first batch ──
            if not magnitude_collected:
                with torch.no_grad():
                    x_patch = model.patch(eeg)  # (B, N, C, D)
                    N = x_patch.shape[1]
                    x_flat = rearrange(x_patch, 'b n c d -> b (n c) d')
                    L = x_flat.shape[1]

                    # Channel embedding
                    cl = channel_ids
                    if cl.dim() == 1:
                        cl = cl.unsqueeze(0).expand(B, -1)
                    chan_id = cl.unsqueeze(1).repeat(1, N, 1).view(B, L)
                    chan_emb = model.channel_embedding(chan_id)

                    # Temporal embedding
                    seq_idx = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
                    temp_emb_val = model.temporal_embedding(seq_idx)

                    patch_norm_val = x_flat.norm(dim=-1).mean().item()
                    chan_norm_val = chan_emb.norm(dim=-1).mean().item()
                    temp_norm_val = temp_emb_val.norm(dim=-1).mean().item()

                    full_token = x_flat + chan_emb + temp_emb_val
                    full_norm_val = full_token.norm(dim=-1).mean().item()

                magnitude_collected = True

            # ── Pass 1: Normal forward ──
            try:
                _ = model(eeg, channel_ids)
            except Exception as e:
                print(f"   Batch {batch_idx} failed: {e}")
                continue

            # Save normal attention for counterfactual comparison
            for l in range(n_layers):
                if l in temporal_attn_storage:
                    normal_attn_cache[l] = temporal_attn_storage[l].clone()

            # Analyze temporal attention structure
            for l in range(n_layers):
                if l not in temporal_attn_storage:
                    continue

                t_attn = temporal_attn_storage[l]  # (B*C, H, N, N)
                BC, H, N, _ = t_attn.shape

                # Locality: correlation between attention weight and distance
                # For each row i, compute weighted average distance
                pos = torch.arange(N, dtype=torch.float)
                dist_matrix = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (N, N)

                # Average attention-weighted distance
                avg_dist = (t_attn * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                # Normalize by max possible distance
                max_dist = N - 1
                locality = 1.0 - (avg_dist.mean().item() / (max_dist / 2))
                # locality > 0.5 = attention is local, < 0.5 = attention is global
                sum_locality[l] += locality

                # Entropy of attention (low = peaked/local, high = uniform/global)
                entropy = -(t_attn * t_attn.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
                max_entropy = np.log(N)
                norm_entropy = entropy / max_entropy
                sum_attn_entropy[l] += norm_entropy

                n_batches[l] += 1

            temporal_attn_storage.clear()

            # ── Pass 2: Counterfactual — zero temporal embedding ──
            # Temporarily replace temporal embedding forward
            original_temp_forward = model.temporal_embedding.forward

            def zero_temporal(seq_indices):
                batch_size, seq_len = seq_indices.shape
                return torch.zeros(batch_size, seq_len,
                                   model.temporal_embedding.pe.shape[-1],
                                   device=seq_indices.device)

            model.temporal_embedding.forward = zero_temporal

            try:
                _ = model(eeg, channel_ids)
            except Exception:
                pass

            model.temporal_embedding.forward = original_temp_forward

            # Compare attention patterns
            for l in range(n_layers):
                if l in temporal_attn_storage and l in normal_attn_cache:
                    normal = normal_attn_cache[l].clamp(min=1e-8)
                    no_temp = temporal_attn_storage[l].clamp(min=1e-8)
                    kl = (normal * (normal.log() - no_temp.log())).sum(dim=-1).mean().item()
                    sum_kl_no_temporal[l] += kl

            temporal_attn_storage.clear()
            normal_attn_cache.clear()

            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1}")

    # Restore original forwards
    for i in range(n_layers):
        encoder[i].attn.forward = original_forwards[i]

    gc.collect()

    # ── Print results ──
    print()
    print("Part 1: Embedding Magnitudes (L2 norm)")
    print("-" * 50)
    print(f"   Patch embedding:    {patch_norm_val:.4f}")
    print(f"   Channel embedding:  {chan_norm_val:.4f}")
    print(f"   Temporal embedding: {temp_norm_val:.4f}")
    total_components = patch_norm_val + chan_norm_val + temp_norm_val
    if total_components > 0:
        print(f"   Temporal fraction:  {temp_norm_val / total_components * 100:.1f}%")
        print(f"   Channel fraction:   {chan_norm_val / total_components * 100:.1f}%")
        print(f"   Patch fraction:     {patch_norm_val / total_components * 100:.1f}%")

    print()
    print("Part 2: Temporal Attention Structure per Layer")
    print("-" * 70)
    print(f"{'Layer':<7} {'Locality':>10} {'Norm Entropy':>13} "
          f"{'KL(rm temp)':>12} {'Pattern':>12}")
    print("-" * 70)

    results = {}
    for l in range(n_layers):
        if n_batches[l] == 0:
            continue
        n = n_batches[l]
        loc = sum_locality[l] / n
        ent = sum_attn_entropy[l] / n
        kl = sum_kl_no_temporal[l] / n if n > 0 else 0.0

        if loc > 0.6:
            pattern = "Local"
        elif loc > 0.45:
            pattern = "Mixed"
        else:
            pattern = "Global"

        results[l] = {
            'locality': loc,
            'normalized_entropy': ent,
            'kl_removing_temporal': kl,
            'pattern': pattern,
        }

        print(f"{l:<7} {loc:>10.4f} {ent:>13.4f} {kl:>12.4f} {pattern:>12}")

    print()
    print("Part 3: Embedding Magnitude vs Riemannian Bias")
    print("-" * 50)
    print("   (Compare temporal/channel embedding norms with the")
    print("    Riemannian bias norms from the embedding_vs_bias analysis)")
    print()
    print("INTERPRETATION:")
    print("   Locality > 0.6 = temporal attention is position-dependent (local)")
    print("   Locality < 0.4 = temporal attention is content-driven (global)")
    print("   High KL(rm temp) = temporal embedding strongly shapes attention")
    print("   Low KL(rm temp)  = attention is mostly content-driven,")
    print("                      temporal position barely matters")
    print()
    print("   If temporal embedding fraction is small AND KL is low,")
    print("   the model relies on content (patch features) over position")
    print("   for temporal routing — which is expected for EEG where")
    print("   non-stationarity makes fixed positions less informative.")
    print()

    results['magnitudes'] = {
        'patch_norm': patch_norm_val,
        'channel_norm': chan_norm_val,
        'temporal_norm': temp_norm_val,
    }

    return results
