import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def merge_token(x, num_channels, k):
    """
    Temporal token merging via bipartite matching.

    Args:
        x: (B, L, D) token sequence where L = N * C
        num_channels: C — number of EEG channels
        k: number of token pairs to merge per channel

    Returns:
        x_merged: (B, L', D) with L' = (N - k) * C
    """
    B, L, D = x.shape

    # Reshape to (B, C, N, D) — group by channel, then time
    x = rearrange(x, "b (n c) d -> b c n d", c=num_channels)
    B, C, N, D = x.shape

    # Split into even (source) and odd (target) time steps
    # Even indices: 0, 2, 4, ...  Odd indices: 1, 3, 5, ...
    x_even = x[:, :, 0::2, :]   # (B, C, N_even, D)
    x_odd = x[:, :, 1::2, :]    # (B, C, N_odd, D)
    N_even = x_even.shape[2]
    N_odd = x_odd.shape[2]

    assert k <= N_even, f"k={k} exceeds number of source tokens N_even={N_even}"

    # --- Bipartite matching (no gradients through routing decisions) ---
    with torch.no_grad():
        # Cosine similarity: (B, C, N_even, N_odd)
        x_even_norm = F.normalize(x_even, dim=-1)
        x_odd_norm = F.normalize(x_odd, dim=-1)
        S = x_even_norm @ x_odd_norm.transpose(-1, -2)

        # For each even token, find its best odd match
        best_odd_per_even = S.argmax(dim=-1)           # (B, C, N_even)
        best_sim_per_even = S.gather(
            dim=-1, index=best_odd_per_even.unsqueeze(-1)
        ).squeeze(-1)                                   # (B, C, N_even)

        # Pick the top-k even tokens with highest match similarity
        _, top_even = best_sim_per_even.topk(k, dim=-1)  # (B, C, k)

        # Find which odd tokens those top-k evens matched
        matched_odd = best_odd_per_even.gather(dim=-1, index=top_even)  # (B, C, k)

        # --- Deduplicate odd matches to get a fixed-size unmerged odd set ---
        # Build a mask of which odd tokens are consumed by at least one merge
        odd_consumed = torch.zeros(B, C, N_odd, device=x.device, dtype=torch.bool)
        odd_consumed.scatter_(dim=2, index=matched_odd, value=True)
        n_consumed = odd_consumed.sum(dim=-1)  # (B, C) — number of unique odd tokens consumed

        # The problem: n_consumed varies per (b, c) because of duplicate matches.
        # We need a fixed count for tensor operations.
        # Solution: always consume exactly k odd tokens.
        # If duplicates exist (n_consumed < k), consume additional odd tokens
        # (those with highest similarity to any merged even) to reach exactly k.

        # For each odd token, compute its max similarity to any of the top-k even tokens
        # Gather columns of S corresponding to the top-k even tokens
        top_even_exp = top_even.unsqueeze(-1).expand(-1, -1, -1, N_odd)  # (B, C, k, N_odd)
        S_topk = S.gather(dim=2, index=top_even_exp)  # (B, C, k, N_odd)
        max_sim_to_merged = S_topk.max(dim=2).values   # (B, C, N_odd)

        # Already consumed odd tokens get highest priority (they must stay consumed)
        # Set unconsumed odd tokens' scores, then pick top-k overall
        # Give already-consumed tokens a large bonus so they're always selected
        priority = max_sim_to_merged + odd_consumed.float() * 1e6
        _, top_odd_to_consume = priority.topk(k, dim=-1)  # (B, C, k)

    # --- Now we have exactly k even tokens and k odd tokens to remove/merge ---
    # top_even: (B, C, k) — even tokens to merge
    # top_odd_to_consume: (B, C, k) — odd tokens to consume (merge or drop)

    # Gather tokens for merging (gradients flow through here)
    top_even_d = top_even.unsqueeze(-1).expand(-1, -1, -1, D)
    even_to_merge = x_even.gather(dim=2, index=top_even_d)  # (B, C, k, D)

    matched_odd_d = matched_odd.unsqueeze(-1).expand(-1, -1, -1, D)
    odd_to_merge = x_odd.gather(dim=2, index=matched_odd_d)  # (B, C, k, D)

    merged = (even_to_merge + odd_to_merge) / 2  # (B, C, k, D)

    # --- Collect unmerged tokens (fixed counts: N_even - k and N_odd - k) ---
    # Even: mask out top_even
    even_mask = torch.ones(B, C, N_even, device=x.device, dtype=torch.bool)
    even_mask.scatter_(dim=2, index=top_even, value=False)

    # Odd: mask out top_odd_to_consume (exactly k, guaranteed)
    odd_mask = torch.ones(B, C, N_odd, device=x.device, dtype=torch.bool)
    odd_mask.scatter_(dim=2, index=top_odd_to_consume, value=False)

    # Gather unmerged — now fixed size across (B, C)
    even_unmerged = x_even[even_mask].view(B, C, N_even - k, D)
    odd_unmerged = x_odd[odd_mask].view(B, C, N_odd - k, D)

    # Concatenate: (B, C, N_even - k + N_odd - k + k, D) = (B, C, N - k, D)
    x_out = torch.cat([even_unmerged, odd_unmerged, merged], dim=2)

    # Reshape back to (B, L', D) where L' = (N - k) * C
    x_out = rearrange(x_out, "b c n d -> b (n c) d")

    return x_out
