import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def merge_token(x, num_channels, k):
    """
    Covariance-aware temporal token merging via bipartite matching.

    Unlike standard ToME (cosine similarity in embedding space), this uses
    spatial covariance structure to decide which time steps are redundant.
    Two time steps with similar channel covariance patterns carry redundant
    spatial information and should be merged.

    Criterion: for each time step t, compute the channel covariance
    S_t = X_t @ X_t^T / D (a C*C SPD matrix), project to the tangent space
    at identity via Pade [1,1] log map: L_t = 2(S_t - I)(I + S_t)^{-1},
    then measure cosine similarity between tangent vectors.
    This ties token merging directly to the same geometric framework as
    the Riemannian spatial attention bias and temporal covariance bias.

    Merging is done at the time-step level (all channels for a time step are
    merged together), not per-channel independently.

    Args:
        x: (B, L, D) token sequence where L = N * C
        num_channels: C — number of EEG channels
        k: number of time-step pairs to merge

    Returns:
        x_merged: (B, L', D) with L' = (N - k) * C
    """
    B, L, D = x.shape
    C = num_channels

    # Reshape to (B, N, C, D) — time steps x channels x embedding
    x = rearrange(x, "b (n c) d -> b n c d", c=C)
    B, N, C, D = x.shape

    # Split into even (source) and odd (target) time steps
    x_even = x[:, 0::2, :, :]   # (B, N_even, C, D)
    x_odd = x[:, 1::2, :, :]    # (B, N_odd, C, D)
    N_even = x_even.shape[1]
    N_odd = x_odd.shape[1]

    assert k <= N_even, f"k={k} exceeds number of source time steps N_even={N_even}"

    # --- Covariance-based bipartite matching (no gradients through routing) ---
    with torch.no_grad():
        # Compute per-time-step channel covariance: S_t = X_t @ X_t^T / D
        # x_even: (B, N_even, C, D) -> cov_even: (B, N_even, C, C)
        cov_even = x_even.float() @ x_even.float().transpose(-1, -2) / D
        cov_odd = x_odd.float() @ x_odd.float().transpose(-1, -2) / D

        # SPD regularization
        eye = torch.eye(C, device=x.device, dtype=torch.float32)
        cov_even = cov_even + 1e-5 * eye
        cov_odd = cov_odd + 1e-5 * eye

        # Project to tangent space via Pade [1,1]: log(S) ~ 2(S-I)(I+S)^{-1}
        # Same log map used by spatial Riemannian bias and temporal cov bias
        X_even = cov_even - eye
        X_odd = cov_odd - eye
        tan_even = torch.linalg.solve(eye + cov_even, 2.0 * X_even)  # (B, N_even, C, C)
        tan_odd = torch.linalg.solve(eye + cov_odd, 2.0 * X_odd)    # (B, N_odd, C, C)

        # Vectorize upper triangle (symmetric -> vector)
        # This avoids redundant lower-triangle entries
        idx = torch.triu_indices(C, C, device=x.device)
        vec_even = tan_even[:, :, idx[0], idx[1]]  # (B, N_even, C*(C+1)/2)
        vec_odd = tan_odd[:, :, idx[0], idx[1]]    # (B, N_odd, C*(C+1)/2)

        # Cosine similarity between tangent vectors: (B, N_even, N_odd)
        vec_even_norm = F.normalize(vec_even, dim=-1)
        vec_odd_norm = F.normalize(vec_odd, dim=-1)
        S = vec_even_norm @ vec_odd_norm.transpose(-1, -2)  # (B, N_even, N_odd)

        # For each even time step, find its best odd match
        best_odd_per_even = S.argmax(dim=-1)           # (B, N_even)
        best_sim_per_even = S.gather(
            dim=-1, index=best_odd_per_even.unsqueeze(-1)
        ).squeeze(-1)                                   # (B, N_even)

        # Pick the top-k even time steps with highest match similarity
        _, top_even = best_sim_per_even.topk(k, dim=-1)  # (B, k)

        # Find which odd time steps those top-k evens matched
        matched_odd = best_odd_per_even.gather(dim=-1, index=top_even)  # (B, k)

        # --- Deduplicate odd matches to get a fixed-size unmerged odd set ---
        odd_consumed = torch.zeros(B, N_odd, device=x.device, dtype=torch.bool)
        odd_consumed.scatter_(dim=1, index=matched_odd, value=True)

        # If duplicates exist, consume additional odd time steps to reach exactly k
        top_even_exp = top_even.unsqueeze(-1).expand(-1, -1, N_odd)  # (B, k, N_odd)
        S_topk = S.gather(dim=1, index=top_even_exp)  # (B, k, N_odd)
        max_sim_to_merged = S_topk.max(dim=1).values   # (B, N_odd)

        priority = max_sim_to_merged + odd_consumed.float() * 1e6
        _, top_odd_to_consume = priority.topk(k, dim=-1)  # (B, k)

    # --- Merge: average the (C, D) channel embeddings of matched time steps ---
    # top_even: (B, k) — even time steps to merge
    # top_odd_to_consume: (B, k) — odd time steps to consume

    # Gather tokens for merging (gradients flow through here)
    top_even_d = top_even.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, D)  # (B, k, C, D)
    even_to_merge = x_even.gather(dim=1, index=top_even_d)  # (B, k, C, D)

    matched_odd_d = matched_odd.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, D)  # (B, k, C, D)
    odd_to_merge = x_odd.gather(dim=1, index=matched_odd_d)  # (B, k, C, D)

    merged = (even_to_merge + odd_to_merge) / 2  # (B, k, C, D)

    # --- Collect unmerged time steps ---
    even_mask = torch.ones(B, N_even, device=x.device, dtype=torch.bool)
    even_mask.scatter_(dim=1, index=top_even, value=False)

    odd_mask = torch.ones(B, N_odd, device=x.device, dtype=torch.bool)
    odd_mask.scatter_(dim=1, index=top_odd_to_consume, value=False)

    # Gather unmerged — expand mask to (B, N_*, C, D) for indexing
    even_unmerged = x_even[even_mask].view(B, N_even - k, C, D)
    odd_unmerged = x_odd[odd_mask].view(B, N_odd - k, C, D)

    # Concatenate: (B, N_even-k + N_odd-k + k, C, D) = (B, N-k, C, D)
    x_out = torch.cat([even_unmerged, odd_unmerged, merged], dim=1)

    # Reshape back to (B, L', D) where L' = (N - k) * C
    x_out = rearrange(x_out, "b n c d -> b (n c) d")

    return x_out
