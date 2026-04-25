import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ═════════════════════════════════════════════════════════════════════════════
# EEG-adapted Rotary Position Embedding (RoPE)
# ═════════════════════════════════════════════════════════════════════════════
#
# Standard RoPE rotates Q/K dimension pairs by position-dependent angles,
# encoding relative position without modifying token magnitude.
#
# EEG adaptation: learnable rotation frequencies initialized over the
# physiologically relevant range [0.5, 50] Hz (delta → gamma).
# Per-layer so each layer can learn its own temporal frequency preference.
# ═════════════════════════════════════════════════════════════════════════════

class EEGRoPE(nn.Module):
    """
    Learnable Rotary Position Embedding for EEG temporal attention.

    Each dimension pair (2i, 2i+1) in Q/K is rotated by angle = n × ω_i,
    where n is the token index and ω_i is a learnable rotation speed.

    Frequencies are initialized log-uniformly over [freq_min, freq_max] Hz,
    covering the physiologically relevant EEG frequency bands.

    Args:
        dim:       dimension of Q/K per head (must be even)
        freq_min:  minimum initialization frequency in Hz (default: 0.5 = delta)
        freq_max:  maximum initialization frequency in Hz (default: 50.0 = low gamma)
    """
    def __init__(self, dim, freq_min=0.5, freq_max=50.0, learnable=True):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        n_pairs = dim // 2
        self.learnable = learnable

        if learnable:
            # EEG-RoPE: learnable ω initialized log-uniform over EEG range.
            # Convert [freq_min, freq_max] Hz to radians/step assuming dt ≈ 0.125s
            # (128Hz, patch_size=16). This is just for initialization — the model
            # learns the actual values.
            dt_approx = 16.0 / 128.0  # 0.125 seconds per token step
            omega_min = 2.0 * math.pi * freq_min * dt_approx  # ~0.39 rad/step
            omega_max = 2.0 * math.pi * freq_max * dt_approx   # ~39.3 rad/step

            log_omega = torch.linspace(
                math.log(omega_min), math.log(omega_max), n_pairs
            )
            omega_init = torch.exp(log_omega)
            self.omega = nn.Parameter(omega_init)  # (n_pairs,)
        else:
            # Standard RoPE: fixed geometric series ω_i = 1/10000^(2i/d)
            # Same formula as in "Attention Is All You Need" / RoFormer.
            omega_init = 1.0 / (10000.0 ** (torch.arange(0, n_pairs).float() / n_pairs))
            self.register_buffer('omega', omega_init)  # (n_pairs,) — NOT learned

    def forward(self, q, k):
        """
        Apply rotary embedding to temporal Q and K.

        RoPE is applied to ALL positions (masked and visible alike).
        Position information is essential for reconstruction — the model
        must know WHERE to reconstruct. The shortcut prevention comes
        from the masking strategy (block masking), not from modifying RoPE.

        Note: We considered mask-aware RoPE (skipping rotation for masked
        positions) but this creates an absolute-position bias: unrotated
        Q attending to rotated K produces scores that depend on absolute
        position j rather than relative distance (j-i). This biases
        mask token attention toward early positions — strictly worse.

        Args:
            q: (B*C, H, N, d) temporal queries
            k: (B*C, H, N, d) temporal keys

        Returns:
            q_rot, k_rot: rotated Q and K, same shape
        """
        N = q.shape[2]
        device = q.device

        # Token indices: 0, 1, ..., N-1
        t = torch.arange(N, device=device, dtype=torch.float32)  # (N,)

        # Rotation angles: (N, n_pairs)
        angles = t.unsqueeze(-1) * self.omega.unsqueeze(0)  # (N, d//2)

        cos_a = angles.cos()  # (N, d//2)
        sin_a = angles.sin()  # (N, d//2)

        # Reshape for broadcast: (1, 1, N, d//2)
        cos_a = cos_a.unsqueeze(0).unsqueeze(0)
        sin_a = sin_a.unsqueeze(0).unsqueeze(0)

        q_rot = self._apply_rotary(q, cos_a, sin_a)
        k_rot = self._apply_rotary(k, cos_a, sin_a)

        return q_rot, k_rot

    @staticmethod
    def _apply_rotary(x, cos_a, sin_a):
        """
        Apply 2D rotation to consecutive dimension pairs.

        x:     (B, H, N, d)
        cos_a: (1, 1, N, d//2)
        sin_a: (1, 1, N, d//2)

        For each pair (x[..., 2i], x[..., 2i+1]):
            x_rot[..., 2i]   = x[..., 2i] * cos - x[..., 2i+1] * sin
            x_rot[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos
        """
        # Split into even/odd dimensions
        x_even = x[..., 0::2]  # (B, H, N, d//2)
        x_odd  = x[..., 1::2]  # (B, H, N, d//2)

        # Cast cos/sin to match x dtype (fp16/fp32)
        cos_a = cos_a.to(x.dtype)
        sin_a = sin_a.to(x.dtype)

        # Rotate
        out_even = x_even * cos_a - x_odd * sin_a
        out_odd  = x_even * sin_a + x_odd * cos_a

        # Interleave back: stack on last dim then reshape
        # (B, H, N, d//2, 2) → (B, H, N, d)
        out = torch.stack([out_even, out_odd], dim=-1)
        return out.reshape(x.shape)

    def get_frequencies_hz(self, dt=0.125):
        """
        Convert learned ω to Hz for interpretability.

        Args:
            dt: seconds per token step (patch_size / sample_rate)

        Returns:
            freqs_hz: (n_pairs,) tensor of learned frequencies in Hz
        """
        return self.omega.detach() / (2.0 * math.pi * dt)


# ─── fp32-safe eigendecomposition ────────────────────────────────────────────
# torch.linalg.eigh has NO fp16/bf16 CUDA kernel.  This wrapper guarantees
# float32 computation regardless of autocast / Lightning precision casting.
def safe_eigh(M):
    """torch.linalg.eigh that always runs in float32 and returns float32."""
    return torch.linalg.eigh(M.float())
# ─────────────────────────────────────────────────────────────────────────────


def pade_matrix_log(M, num_squarings=4, sqrt_iters=6, pade_order=6):
    """
    Matrix logarithm via inverse scaling-and-squaring + Taylor approximation.

    All operations are batched matmuls + linalg.solve — GPU-friendly.
    No eigendecomposition needed, so backward pass is always clean (no
    degenerate-eigenvalue NaN).

    Internally runs in float32 to avoid fp16 kernel gaps.

    Algorithm:
        1. Repeated matrix square root: compute A = M^{1/2^s} by applying
           Denman-Beavers iteration s times (each time converged with
           sqrt_iters iterations). After this, A ≈ I.
        2. Taylor series: log(I + X) ≈ X - X²/2 + X³/3 - ...
           Converges because ||A - I|| < 1 after sufficient squarings.
        3. Scale back: log(M) = 2^s * log(A)

    Args:
        M: (..., C, C) batch of SPD matrices
        num_squarings: number of square-root halvings (more = A closer to I)
        sqrt_iters: Denman-Beavers iterations per square root (6 is sufficient
                    for quadratic convergence on well-conditioned SPD matrices)
        pade_order: order of the Taylor approximation (higher = more accurate)
    Returns:
        (..., C, C) matrix logarithm of M
    """
    orig_dtype = M.dtype
    # Force float32 — torch.linalg.solve has no fp16/bf16 CUDA kernel
    with torch.amp.autocast('cuda', enabled=False), \
         torch.amp.autocast('cpu', enabled=False), \
         torch.amp.autocast('mps', enabled=False):
        M = M.float()
        C = M.shape[-1]
        I = torch.eye(C, device=M.device, dtype=M.dtype).expand_as(M)

        # ── Step 1: Repeated matrix square root via Denman-Beavers ──
        # Each outer iteration computes M^{1/2} of the current matrix,
        # so after num_squarings iterations we have M^{1/2^s}.
        A = M
        for _ in range(num_squarings):
            # Denman-Beavers: Y→A^{1/2} with sqrt_iters iterations
            Y = A
            for _ in range(sqrt_iters):
                Y_inv = torch.linalg.solve(Y, I)
                Y = 0.5 * (Y + Y_inv)
            A = Y  # A ← A^{1/2}

        # Now A ≈ M^{1/2^s}, which should be very close to I

        # ── Step 2: Taylor series log(I + X) for A ≈ I ──
        X = A - I
        # Clamp X to prevent divergence if A isn't close enough to I
        X = X.clamp(-0.9, 0.9)
        result = torch.zeros_like(X)
        X_power = X  # X^1
        for k in range(1, pade_order + 1):
            sign = 1.0 if k % 2 == 1 else -1.0
            result = result + (sign / k) * X_power
            if k < pade_order:
                X_power = X_power @ X  # X^{k+1}

        # ── Step 3: Undo the scaling: log(M) = 2^s * log(M^{1/2^s}) ──
        result = result * (2 ** num_squarings)

        # Final NaN guard — replace any NaN with 0 (identity in tangent space)
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    return result.to(orig_dtype)


class MultiHeadAttentionViT(nn.Module):
    """Multi head attention module, takes the embedding dim and number of head."""
    def __init__(self, embed_dim, num_heads = 3, proj_drop = 0.1, att_dropout = 0.1, qkv_bias = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim//num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias=qkv_bias)
        self.fc = nn.Linear(embed_dim,embed_dim)
        self.att_dropout = att_dropout
        self.proj_drop = nn.Dropout(proj_drop)


    def split_heads(self, X):
        return X.view(X.size(0), X.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)      

    def forward(self, x, mask_pad = None):
        #Compute Q, K and V and seperate segments per head
        B, N, D = x.shape

        #Extract the query key value vectors each with shape
        # B, num_head, num_patches, dim_head
        qkv = self.split_heads(self.qkv(x))
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        #Compute the attention score
        out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask_pad, dropout_p=self.att_dropout if self.training else 0, is_causal=False)

        out = out.transpose(1,2)
        out = out.reshape(out.size(0), out.size(1), self.h*self.dim_head)
        out = self.fc(out)
        out = self.proj_drop(out)
        return out


class DropPath(nn.Module):
    """Randomly drops the attention branch"""
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = float(drop_prob)
    
    def drop_path(self, x):
        #Deactivate if training
        if self.drop_prob == 0.0 or self.training == False:
            return x
        keep_prob = 1 - self.drop_prob
        
        #Dim of tensor B, 1, 1,...
        dim_tensor = (x.shape[0],) + ([1]) * (x.ndim - 1)

        #Create a tensor of size B,1,1... with values between 0 and 2
        rand_tensor = keep_prob + torch.rand(dim_tensor, dtype=x.dtype, device=x.device)
        #Convert values to binary
        drop_tensor = rand_tensor.floor()
        #Drop the chosen values and divide by the probability to keep the same original expectation
        return x.div(keep_prob) * drop_tensor
    
    def forward(self, x):
        return self.drop_path(x)
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features = None, act = nn.GELU, drop = 0):
        super().__init__()
        out_features = out_features or in_features
        self.act = act()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)
    

class TransformerLayerViT(nn.Module):
    def __init__(self, embed_dim, nhead, mlp_ratio = 4, qkv_bias = True,drop = 0, att_drop = 0, 
                 drop_path = 0, act = nn.GELU, norm = nn.LayerNorm):
        super().__init__()
        self.attn = MultiHeadAttentionViT(embed_dim=embed_dim, num_heads=nhead, proj_drop=drop,
                                               att_dropout=att_drop, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, mask_pad = None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask_pad))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
    
class TimeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3, dropout=0.1, att_dropout=0.1, is_causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.att_dropout = att_dropout

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)

    def forward(self, x, num_chan):
        B, L, D = x.shape
        assert L % num_chan == 0
        N = L // num_chan

        # (B, L, D) -> (B, N, C, D) -> (B*C, N, D)
        x = rearrange(x, "b (n c) d -> b n c d", c=num_chan)
        x = rearrange(x, "b n c d -> (b c) n d")

        qkv = self.split_heads(self.qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.att_dropout if self.training else 0.0,
            is_causal=self.is_causal
        )

        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), self.h * self.dim_head)
        x = self.fc(x)

        # (B*C, N, D) -> (B, N, C, D) -> (B, L, D)
        x = rearrange(x, "(b c) n d -> b n c d", b=B, c=num_chan)
        x = rearrange(x, "b n c d -> b (n c) d")

        return self.dropout(x)


# =============================================================================
# Riemannian Attention Bias for Spatial Attention
# =============================================================================
#
# Core idea: EEG channel relationships are fundamentally about covariance
# structure, which lives on the SPD (Symmetric Positive Definite) manifold.
# Standard dot-product attention treats channel interactions as Euclidean,
# ignoring this geometry. We inject the Riemannian structure as an additive
# bias to the spatial attention logits.
#
# Pipeline at each time step t:
#   1. Given channel embeddings X_t ∈ R^{C×D}, compute the sample covariance:
#        S_t = (1/D) X_t X_t^T + εI    (SPD matrix, C×C)
#   2. Project to the tangent space at the identity via matrix logarithm:
#        L_t = log(S_t)                 (symmetric matrix, C×C)
#   3. Use L_t as an attention bias with per-head learned scaling:
#        attention_logits = QK^T/√d + α_h · L_t
#
# Why this works:
# - The matrix log maps SPD matrices to the tangent space where Euclidean
#   operations become valid approximations of Riemannian operations
# - L_t(i,j) encodes the log-domain correlation between channels i and j,
#   respecting the manifold geometry (log-Euclidean metric)
# - Per-head scaling lets each head decide how much geometric prior to use
# - Initialized at α=0, so the model starts as standard attention and
#   gradually learns to incorporate the Riemannian bias during training
# =============================================================================


class SPDLogMap(nn.Module):
    """
    Projects SPD matrices to the tangent space at identity via matrix logarithm.
    Uses eigendecomposition: S = QΛQ^T → log(S) = Q log(Λ) Q^T

    This is the Log-Euclidean projection, which is computationally efficient
    and provides a good approximation of the full AIRM tangent space.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, S):
        """
        Args:
            S: (..., C, C) batch of SPD matrices
        Returns:
            (..., C, C) batch of symmetric matrices in tangent space
        """
        orig_dtype = S.dtype
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False), \
             torch.amp.autocast('mps', enabled=False):
            S = S.float()
            eigenvalues, eigenvectors = safe_eigh(S)
            eigenvalues = eigenvalues.clamp(min=self.eps)
            log_eigenvalues = torch.log(eigenvalues)
            result = eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)
        return result.to(orig_dtype)


class RiemannianAttentionBias(nn.Module):
    """
    Computes a geometry-aware attention bias from channel embeddings.

    At each time step, channel embeddings form a covariance matrix on the SPD
    manifold. We project this to the tangent space and use it as an additive
    bias to the attention logits, giving the model awareness of the Riemannian
    structure of inter-channel relationships.

    Args:
        num_heads: Number of attention heads (each gets its own learned scale)
        eps: Regularization for SPD matrix (ensures strict positive definiteness)
    """
    def __init__(self, num_heads, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.spd_log = SPDLogMap(eps=eps)

        # Per-head learnable scaling — initialized to 0 so the model starts
        # as standard Euclidean attention and learns to use the bias
        self.head_scales = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x):
        """
        Args:
            x: (B*N, C, D) channel embeddings at each time step
               B*N = batch_size * num_time_patches
               C = number of channels
               D = embedding dimension
        Returns:
            (B*N, num_heads, C, C) attention bias per head
        """
        BN, C, D = x.shape

        # Step 1: Compute sample covariance → SPD matrix in float32
        # x @ x^T can overflow fp16 when residual stream has large values
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            x_f32 = x.float()
            S = torch.bmm(x_f32, x_f32.transpose(-2, -1)) / D
            S = S + self.eps * torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)

        # Step 2: Project to tangent space via matrix log
        L = self.spd_log(S)  # (B*N, C, C)

        # Step 3: Per-head scaling
        # head_scales: (num_heads,) → (1, num_heads, 1, 1)
        scales = self.head_scales.view(1, self.num_heads, 1, 1)

        # L: (B*N, C, C) → (B*N, 1, C, C) → broadcast to (B*N, num_heads, C, C)
        bias = L.unsqueeze(1) * scales

        return bias


class RiemannianSpaceAttention(nn.Module):
    """
    Spatial attention with Riemannian bias.

    Drop-in replacement for SpaceAttention. Adds a Riemannian attention bias
    derived from the SPD manifold structure of channel covariances. The bias
    is computed from the channel embeddings BEFORE the QKV projection, so it
    captures the raw geometric structure of the input.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Output dropout probability
        att_dropout: Attention weight dropout probability
        eps: SPD regularization constant
    """
    def __init__(self, embed_dim, num_heads=3, dropout=0.1, att_dropout=0.1, eps=1e-5):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

        # Riemannian bias module
        self.riemannian_bias = RiemannianAttentionBias(num_heads=num_heads, eps=eps)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)

    def forward(self, x, num_chan):
        B, L, D = x.shape
        assert L % num_chan == 0
        N = L // num_chan

        # Reshape: (B, N*C, D) → (B*N, C, D)
        x = rearrange(x, "b (n c) d -> (b n) c d", c=num_chan)

        # Compute Riemannian bias from raw embeddings (before QKV projection)
        # This captures the geometric structure of the input, not the projected space
        riem_bias = self.riemannian_bias(x)  # (B*N, num_heads, C, C)

        # Standard QKV computation
        qkv = self.split_heads(self.qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores with Riemannian bias
        score = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        score = score + riem_bias  # Add geometric bias
        score = score.softmax(dim=-1)
        attn = F.dropout(score, p=self.att_dropout, training=self.training)

        out = attn @ v
        out = out.transpose(1, 2).reshape(out.size(0), out.size(2), self.h * self.dim_head)
        out = self.fc(out)

        # Reshape back: (B*N, C, D) → (B, N*C, D)
        out = rearrange(out, "(b n) c d -> b (n c) d", b=B, n=N)

        return self.dropout(out)


class RiemannianCrissCrossTransformer(nn.Module):
    """
    Criss-Cross Transformer with Riemannian spatial attention.

    Drop-in replacement for CrissCrossTransformer. Uses standard Euclidean
    attention for the temporal dimension (where Riemannian geometry doesn't
    apply) and Riemannian-biased attention for the spatial dimension (where
    channel covariance structure is geometrically meaningful).

    Args:
        embed_dim: Total embedding dimension
        nhead: Number of attention heads
        num_chan: Number of EEG channels (needed for reshaping)
        mlp_ratio: MLP hidden dimension ratio
        drop: Dropout probability
        att_drop: Attention dropout probability
        drop_path: Stochastic depth probability
        act: Activation function class
        norm: Normalization layer class
        spd_eps: Regularization for SPD computation
    """
    def __init__(self, embed_dim, nhead, mlp_ratio=4, drop=0.0, att_drop=0.0,
                 drop_path=0.0, act=nn.GELU, norm=nn.LayerNorm, spd_eps=1e-5):
        super().__init__()
       

        # Temporal attention stays Euclidean — temporal dynamics within a single
        # channel are sequential, not covariance-structured
        self.attn_time = TimeAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=drop,
            att_dropout=att_drop
        )

        # Spatial attention with Riemannian bias — this is where the manifold
        # structure of channel covariances gets injected
        self.attn_space = RiemannianSpaceAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=drop,
            att_dropout=att_drop,
            eps=spd_eps
        )

        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path3 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.norm3 = norm(embed_dim)

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan):
        x = x + self.drop_path1(self.attn_time(self.norm1(x), num_chan))
        x = x + self.drop_path2(self.attn_space(self.norm2(x), num_chan))
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x
    


# =============================================================================
# Improved Riemannian Attention: Adaptive Reference + Residual Stream Bias
# =============================================================================
#
# Two improvements over the original Riemannian attention bias:
#
# 1. ADAPTIVE LOG MAP: Instead of projecting to the tangent space at the
#    identity matrix I (Log-Euclidean), we learn a reference point R on the
#    SPD manifold. The log map at R is:
#        Log_R(S) = log(R^{-1/2} S R^{-1/2})
#    This captures *deviations* from the learned typical covariance, rather
#    than absolute covariance.
#
#    VARIABLE CHANNEL SUPPORT: The reference R_full is stored in the GLOBAL
#    channel space (144×144, covering all possible EEG channels). At forward
#    time, the batch provides channel indices and we extract the C×C submatrix
#    R = R_full[idx, :][:, idx]. This means:
#    - Different batches can have different numbers of channels
#    - R learns the covariance geometry across ALL channels globally
#    - Each batch just takes the relevant slice
#
#    R_full is parameterized via a Cholesky factor L where R = LL^T + εI,
#    guaranteeing R stays SPD. Initialized at L=I so R≈I at init.
#
# 2. RESIDUAL STREAM BIAS: The Riemannian bias is computed from the residual
#    stream (pre-LayerNorm) rather than the normalized input. LayerNorm
#    destroys per-channel variance information that is critical for the
#    covariance structure on the SPD manifold. By using the residual stream,
#    we preserve the full scale+correlation geometry.
#
# Additionally, a first-order approximation mode is available:
#    log(M) ≈ M - I    (valid when M is close to I)
# This avoids eigendecomposition entirely. The adaptive reference R makes
# this approximation more accurate by bringing M = R^{-1/2} S R^{-1/2}
# closer to I as R adapts to the data distribution.
# =============================================================================

# Total number of channels in the global mapping (channel_info.yaml)
TOTAL_GLOBAL_CHANNELS = 144


class AdaptiveLogMap(nn.Module):
    """
    Log map at a reference point on the SPD manifold, supporting
    variable channel counts across batches.

    Supports three tangent-space projection modes:
        'approx' — first-order Taylor: S - I  (fastest, least accurate)
        'pade'   — Padé [1,1]: 2(S-I)(I+S)^{-1}  (good accuracy near I)
        'eigh'   — full eigendecomposition  (exact, expensive)

    Args:
        total_channels: Size of the global channel space (default 144)
        eps: Regularization constant for numerical stability
        log_mode: 'approx', 'pade', or 'eigh'
        use_approx: DEPRECATED — kept for backward compat.
    """
    def __init__(self, total_channels=TOTAL_GLOBAL_CHANNELS, eps=1e-5,
                 log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None):
        super().__init__()
        self.eps = eps
        self.total_channels = total_channels

        # Backward compat: use_approx=True overrides log_mode
        if use_approx:
            self.log_mode = 'approx'
        else:
            self.log_mode = log_mode

    def forward(self, S, channel_idx=None):
        """
        Args:
            S: (batch, C, C) batch of SPD matrices
            channel_idx: (C,) global channel indices (unused, kept for API compat)
        Returns:
            (batch, C, C) tangent vectors
        """
        orig_dtype = S.dtype
        C = S.shape[-1]
        I = torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)

        # ── Tangent-space projection ──
        if self.log_mode == 'approx':
            # First-order: S - I
            return (S - I).to(orig_dtype)
        else:
            # Padé [1,1] approximant of matrix logarithm:
            # log(S) ≈ 2(S - I)(I + S)^{-1}
            # Must disable autocast — Cholesky has no fp16 CUDA kernel
            with torch.amp.autocast('cuda', enabled=False), \
                 torch.amp.autocast('cpu', enabled=False):
                S_f32 = S.float().contiguous()
                I_f32 = I.float()
                X = S_f32 - I_f32
                # (I + S)^{-1} · 2X — device-agnostic linalg.solve.
                T = torch.linalg.solve(I_f32 + S_f32, 2 * X)

            # ── NaN/Inf guard after Padé ──
            if torch.isnan(T).any() or torch.isinf(T).any():
                print(f"[AdaptiveLogMap] NaN/Inf AFTER Padé solve!")
                print(f"  S diag range: [{S.diagonal(dim1=-2,dim2=-1).min().item():.4f}, "
                      f"{S.diagonal(dim1=-2,dim2=-1).max().item():.4f}]")
                # Fall back to first-order (S - I) which can't produce NaN from SPD input
                T = X

            return T.to(orig_dtype)



class AdaptiveRiemannianAttentionBias(nn.Module):
    """
    Geometry-aware attention bias with adaptive manifold reference.

    Supports variable channel counts: the reference is stored globally (144×144)
    and the batch's channel indices select the relevant submatrix at runtime.

    Args:
        num_heads: Number of attention heads (each gets its own learned scale)
        total_channels: Size of the global channel space (default 144)
        eps: Regularization for SPD matrix
        log_mode: 'approx', 'pade', or 'eigh' (see AdaptiveLogMap)
        use_approx: DEPRECATED — kept for backward compat.
    """
    def __init__(self, num_heads, total_channels=TOTAL_GLOBAL_CHANNELS,
                 eps=1e-5, log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None,
                 learn_mu_reference=True, disable_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        # When disable_bias=True, skip the covariance + Padé path entirely
        # and return zeros. This is the real "baseline" ablation — no
        # Riemannian compute cost on GPU (or on MPS CPU-fallback path).
        # Used for fair efficiency comparisons and fast baseline runs.
        self.disable_bias = disable_bias
        self.adaptive_log = AdaptiveLogMap(
            total_channels=total_channels, eps=eps,
            log_mode=log_mode, use_approx=use_approx,
        )

        # Per-head learnable scaling — initialized to 0 so the model starts
        # as standard Euclidean attention and learns to use the bias
        self.head_scales = nn.Parameter(torch.zeros(num_heads))

        # Learnable tangent-space reference (C3: whitening without matrix sqrt).
        # Stored in the global channel space for cross-dataset sharing.
        # Subtracted from log(S) before bias computation, effectively shifting
        # the log-Euclidean reference from I toward the learned Fréchet-mean-like
        # center of the covariance distribution. Initialized at 0 → neutral start.
        # See Arsigny et al. 2007 for the log-Euclidean framework justification:
        # in tangent space at identity, Euclidean subtraction corresponds to
        # Riemannian recentering under the log-Euclidean metric.
        if learn_mu_reference:
            self.mu_log = nn.Parameter(
                torch.zeros(total_channels, total_channels)
            )
        else:
            self.register_parameter('mu_log', None)

    def forward(self, x, channel_idx, mask_space=None):
        """
        Args:
            x: (B*N, C, D) channel embeddings (from residual stream).
               Masked-channel rows should already be zeroed by caller.
            channel_idx: (C,) long tensor — global channel indices for this batch
            mask_space: (B*N, C) boolean — True = masked channel at this timestep.
                        When provided, masked rows/cols of S are overwritten with
                        identity structure so log(S) is zero on masked positions
                        (clean zero bias instead of Padé artifacts on fake zeros).
        Returns:
            bias: (B*N, num_heads, C, C) attention bias per head
            L: (B*N, C, C) tangent vector (log(S) minus optional learned μ).
                Exposed for the optional value-bias path.
        """
        BN, C, D = x.shape

        # ── Fast baseline path: skip all Riemannian computation ──
        # When disable_bias is True, return zero bias without computing the
        # covariance matrix or the Padé log-map. This is the true "no-C1"
        # ablation — no compute cost, no CPU fallback on MPS, no wall-time tax.
        # Returns zeros of expected shape so downstream code doesn't break.
        if self.disable_bias:
            zero_bias = torch.zeros(
                BN, self.num_heads, C, C, device=x.device, dtype=x.dtype
            )
            zero_L = torch.zeros(BN, C, C, device=x.device, dtype=x.dtype)
            return zero_bias, zero_L

        # Step 1: Compute sample covariance → SPD matrix
        # MUST be float32 — the residual stream x can have large values after
        # many layers, and x @ x^T overflows fp16 (max 65504) → inf → NaN.
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            x_f32 = x.float()
            S = torch.bmm(x_f32, x_f32.transpose(-2, -1)) / D
            eye = torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)
            S = S + self.eps * eye

            # Structural masking: overwrite masked rows/cols with identity.
            # Without this, masked entries in S are fake zeros → Padé log
            # produces garbage (-1 on diagonal, noise off-diagonal) at masked
            # positions. After this fix, S = [[S_vis, 0], [0, I]] block-wise,
            # so log(S) has clean zeros on masked rows/cols.
            if mask_space is not None:
                mf = mask_space.float()
                m_any = (mf.unsqueeze(-1) + mf.unsqueeze(-2)).clamp_(max=1.0)
                S = S * (1.0 - m_any) + eye * m_any

        # Step 2: Project to tangent space at identity via Padé [1,1]
        L = self.adaptive_log(S, channel_idx)  # (B*N, C, C)

        # Step 2b: Learnable tangent-space centering (C3). Subtract the
        # learnable reference μ[channel_idx] from L, shifting the effective
        # log-Euclidean reference point away from identity toward the learned
        # Fréchet-mean-like center. Symmetrized to preserve the symmetry of
        # tangent vectors. Under log-Euclidean geometry, this is the geodesic
        # mean shift — no matrix sqrt required.
        if self.mu_log is not None:
            mu_sub = self.mu_log[channel_idx][:, channel_idx]  # (C, C) submatrix
            mu_sub = 0.5 * (mu_sub + mu_sub.transpose(-2, -1))  # enforce symmetry
            L = L - mu_sub.unsqueeze(0)                        # broadcast over B*N

        # Step 3: Per-head scaling on the (possibly centered) tangent
        scales = self.head_scales.view(1, self.num_heads, 1, 1)
        bias = L.unsqueeze(1) * scales

        return bias, L


# ═════════════════════════════════════════════════════════════════════════════
# TemporalRiemannianAttentionBias — Run 6 (frequency-aware via Wiener-Khinchin)
# ═════════════════════════════════════════════════════════════════════════════
#
# Symmetric counterpart to AdaptiveRiemannianAttentionBias on the temporal axis.
# For each channel, computes Σ_temporal = X X^T / D where X is the patch-sequence
# embeddings (N, D). By Wiener-Khinchin theorem, this autocorrelation matrix
# encodes the same information as the power spectrum: the off-diagonal banding
# patterns reveal which periodicities (frequencies) are present.
#
# Used as additive bias on temporal attention scores, mirroring how spatial
# Σ is used on spatial attention. Layer-adaptive (computed from residual
# stream at each layer) — does not depend on raw signal, no FFT, no bands.
#
# Shape conventions:
#   Input: (B*C, N, D) per-channel patch-sequence embeddings
#   Σ_temporal: (B*C, N, N) symmetric SPD
#   Output bias: (B*C, num_heads, N, N) — added to temporal attention scores
#
# Variable N across datasets handled via μ_temporal of shape (max_patches, max_patches),
# indexed at [:N, :N] at runtime.
# ═════════════════════════════════════════════════════════════════════════════


class TemporalRiemannianAttentionBias(nn.Module):
    """
    Riemannian attention bias on the temporal axis.

    Symmetric to AdaptiveRiemannianAttentionBias but operates on patch-sequence
    embeddings per channel. Captures periodic temporal structure (frequency
    information via Wiener-Khinchin) as an additive bias on temporal attention.

    Args:
        num_heads: Number of temporal attention heads
        max_patches: Maximum N (patch count). μ_temporal sized for this.
                     Indexed at [:N, :N] for variable-N inputs.
        eps: Ridge regularization for SPD safety
        learn_mu_reference: Use learnable μ_temporal tangent-space reference
        disable_bias: When True, return zeros without computing covariance/Padé
                      (for fast baseline ablation, no compute cost).
    """
    def __init__(self, num_heads, max_patches=128, eps=1e-5,
                 learn_mu_reference=True, disable_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.max_patches = max_patches
        self.disable_bias = disable_bias

        # Per-head learnable scaling. Init at 0 so model starts identical to
        # no-temporal-bias baseline; learns to use the bias if helpful.
        self.head_scales = nn.Parameter(torch.zeros(num_heads))

        # Learnable tangent-space reference μ_temporal. Stored as max_patches ×
        # max_patches matrix; runtime indexes the [:N, :N] submatrix.
        # Init at 0 → neutral start (Padé expansion at identity).
        if learn_mu_reference:
            self.mu_log = nn.Parameter(torch.zeros(max_patches, max_patches))
        else:
            self.register_parameter('mu_log', None)

    def forward(self, x_temporal, mask_temporal=None):
        """
        Args:
            x_temporal: (B*C, N, D) per-channel patch-sequence embeddings.
                        Masked patches' rows should be zeroed by caller.
            mask_temporal: (B*C, N) boolean — True = masked patch.
                           When provided, masked rows/cols of S overwritten
                           with identity (clean zero bias on masked positions).
        Returns:
            bias: (B*C, num_heads, N, N) attention bias per head
            L: (B*C, N, N) tangent vector (log(S) − μ_temporal_sub)
        """
        BC, N, D = x_temporal.shape

        # Fast baseline path: skip Riemannian compute entirely
        if self.disable_bias:
            zero_bias = torch.zeros(
                BC, self.num_heads, N, N,
                device=x_temporal.device, dtype=x_temporal.dtype,
            )
            zero_L = torch.zeros(
                BC, N, N, device=x_temporal.device, dtype=x_temporal.dtype,
            )
            return zero_bias, zero_L

        # Step 1: compute Σ_temporal in fp32 (linalg.solve has no fp16 kernel)
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False), \
             torch.amp.autocast('mps', enabled=False):
            x_f32 = x_temporal.float()
            S = torch.bmm(x_f32, x_f32.transpose(-2, -1)) / D  # (B*C, N, N)
            eye = torch.eye(N, device=S.device, dtype=S.dtype).unsqueeze(0)
            S = S + self.eps * eye

            # Mask handling: overwrite masked rows/cols with identity, so log(S)
            # is zero on masked positions (no Padé garbage on fake-zero entries).
            if mask_temporal is not None:
                mf = mask_temporal.float()
                m_any = (mf.unsqueeze(-1) + mf.unsqueeze(-2)).clamp_(max=1.0)
                S = S * (1.0 - m_any) + eye * m_any

            # Step 2: Padé [1,1] log-map: log(S) ≈ 2(S − I)(I + S)^{-1}
            # via torch.linalg.solve to avoid explicit matrix inverse.
            Y = eye + S                                    # (B*C, N, N)
            X_num = 2.0 * (S - eye)                        # (B*C, N, N)
            L = torch.linalg.solve(Y, X_num)               # (B*C, N, N)
            # Symmetrize for numerical safety (tangent vectors should be sym)
            L = 0.5 * (L + L.transpose(-2, -1))

        # Step 3: subtract learnable μ_temporal[:N, :N] submatrix, symmetrized
        if self.mu_log is not None:
            assert N <= self.max_patches, (
                f"N={N} exceeds max_patches={self.max_patches}. "
                f"Increase max_temporal_patches in TemporalRiemannianAttentionBias."
            )
            mu_sub = self.mu_log[:N, :N]
            mu_sub = 0.5 * (mu_sub + mu_sub.transpose(-2, -1))
            L = L - mu_sub.unsqueeze(0)                    # broadcast over B*C

        # Step 4: per-head scaling
        scales = self.head_scales.view(1, self.num_heads, 1, 1)
        bias = L.unsqueeze(1) * scales                     # (B*C, num_heads, N, N)

        return bias, L


# ═════════════════════════════════════════════════════════════════════════════
# FilterBankBias — Run 4 FB-C1
# ═════════════════════════════════════════════════════════════════════════════
#
# Inject FBCSP-style band-specific spatial priors into the Riemannian attention
# bias. Classical EEG analysis (Ang et al. 2008 FBCSP, Barachant et al. 2012
# MDRM) separates signal into K physiological bands and computes one covariance
# matrix per band — we bring that prior into the transformer as a learnable
# additive bias on top of the adaptive per-layer C1 bias.
#
# Design (all constraints from the pressure test are enforced here):
#   (1) SincConv on RAW signal, once at the encoder level. Not per-layer:
#       raw-frequency semantics degrade past the first few transformer layers.
#   (2) MASK-AWARE covariance. During pretraining, the raw-signal samples
#       corresponding to masked patches are zeroed BEFORE SincConv. Each pair
#       (i, j) is normalized by the number of time samples where BOTH channels
#       are visible — prevents masked content from leaking into S_k.
#   (3) Ridge regularization λI on S_k before Padé log-map; guarantees SPD
#       even with heavy masking and high channel counts.
#   (4) SincConv cutoffs FROZEN at init by default. Learnable cutoffs have
#       a collapse mode (bands drift together) that wastes capacity.
#   (5) β gates — the per-head per-band scalars that multiply the FB bias —
#       are declared in the attention module (not here). They init to 0 so
#       a fresh --use_filter_bank run on an old C1+C3 checkpoint matches the
#       checkpoint's behavior bit-for-bit at epoch 0.
#
# Output: per-band (log(S_k) − μ_k), shape (B, K, C, C). Reused across all
# transformer layers via an `fb_log_S` kwarg threaded through forward().
# ═════════════════════════════════════════════════════════════════════════════


class SincConv1d(nn.Module):
    """
    Parametric bandpass SincConv layer.

    Each band is a windowed sinc bandpass: h(t) = 2f_hi sinc(2f_hi t)
                                                - 2f_lo sinc(2f_lo t),
    multiplied by a Hamming window. Parameters: (f_lo, f_hi) per band — with
    `learnable=False` they are registered as buffers (frozen at init) and
    applied via a standard depthwise conv1d.

    Args:
        sample_rate: Hz of the input signal.
        num_bands:   K bands in the filter bank.
        kernel_size: FIR taps. Must be odd. Longer → sharper transitions but
                     heavier boundary decay at low frequencies. Default 65
                     taps ≈ 0.5 s at 128 Hz — adequate for α/β/γ, marginal
                     for δ (0.5-4 Hz). Use 129 if δ fidelity matters.
        band_edges:  list of (f_lo, f_hi) in Hz. Length must equal num_bands.
                     Default: classical EEG bands (δ, θ, α, β, γ).
        learnable:   If True, cutoffs trained with the rest of the model. If
                     False (default), cutoffs are frozen buffers.
        min_band_hz: floor for (f_hi - f_lo) when learnable — prevents
                     collapse to zero-bandwidth.
    """
    DEFAULT_BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0)]

    def __init__(self, sample_rate, num_bands=5, kernel_size=65,
                 band_edges=None, learnable=False, min_band_hz=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        if band_edges is None:
            band_edges = self.DEFAULT_BANDS
        assert len(band_edges) == num_bands, \
            f"band_edges length {len(band_edges)} != num_bands {num_bands}"

        self.sample_rate = float(sample_rate)
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.learnable = learnable
        self.min_band_hz = float(min_band_hz)

        lo = torch.tensor([e[0] for e in band_edges], dtype=torch.float32)
        hi = torch.tensor([e[1] for e in band_edges], dtype=torch.float32)
        # Parameterize as (f_lo, band_width): keeps hi > lo by construction.
        band_hz = hi - lo

        if learnable:
            self.f_lo = nn.Parameter(lo)
            self.band_hz = nn.Parameter(band_hz)
        else:
            self.register_buffer('f_lo', lo)
            self.register_buffer('band_hz', band_hz)

        # Hamming window and symmetric time axis, registered as buffers.
        n = (kernel_size - 1) // 2
        # Symmetric time axis in SAMPLES, excluding 0 (center handled via limit).
        t = torch.arange(-n, n + 1, dtype=torch.float32)
        self.register_buffer('t_axis', t)
        window = 0.54 - 0.46 * torch.cos(
            2.0 * math.pi * torch.arange(kernel_size, dtype=torch.float32)
            / (kernel_size - 1)
        )
        self.register_buffer('window', window)

    def _build_filters(self):
        """Build the K bandpass filters from current (f_lo, band_hz)."""
        # Clamp to physically valid range: 0 < f_lo < f_hi < Nyquist.
        f_lo = self.f_lo.clamp(min=0.0, max=self.sample_rate / 2.0 - self.min_band_hz)
        band = self.band_hz.clamp(min=self.min_band_hz,
                                  max=self.sample_rate / 2.0)
        f_hi = (f_lo + band).clamp(max=self.sample_rate / 2.0)

        # Normalize by sample rate: cycles per sample.
        f_lo_norm = (f_lo / self.sample_rate).view(-1, 1)  # (K, 1)
        f_hi_norm = (f_hi / self.sample_rate).view(-1, 1)  # (K, 1)
        t = self.t_axis.view(1, -1)                        # (1, L)

        # Ideal bandpass FIR: h(t) = 2 f_hi sinc(2 f_hi t) − 2 f_lo sinc(2 f_lo t).
        # At t=0 the sinc limit is 1, so h(0) = 2 (f_hi − f_lo). Handle that
        # by replacing NaN / large values at the center with the analytic limit.
        two_pi_hi_t = 2 * math.pi * f_hi_norm * t
        two_pi_lo_t = 2 * math.pi * f_lo_norm * t

        # Safe division: where t==0, directly substitute the limit later.
        eps = 1e-12
        # sinc(2f t) in unnormalized form = sin(2π f t) / (π t). At t=0 the
        # analytic limit is 2f. `torch.where` broadcasts cond(1,L), A(K,1),
        # B(K,L) → (K, L) without any explicit expand_as (which would fail
        # because f_*_norm has shape (K, 1) ≠ t's shape (1, L)).
        sinc_hi = torch.where(
            t.abs() < 1e-9,
            2.0 * f_hi_norm,                               # (K, 1) broadcast
            torch.sin(two_pi_hi_t) / (math.pi * t + eps),  # (K, L)
        )
        sinc_lo = torch.where(
            t.abs() < 1e-9,
            2.0 * f_lo_norm,                               # (K, 1) broadcast
            torch.sin(two_pi_lo_t) / (math.pi * t + eps),  # (K, L)
        )
        filt = sinc_hi - sinc_lo                               # (K, L)
        filt = filt * self.window.unsqueeze(0)                 # Hamming window
        # Energy-normalize each band → each filter has unit L2 norm so the
        # covariance of a unit-variance white signal is comparable across bands.
        filt = filt / (filt.norm(dim=-1, keepdim=True) + 1e-8)
        return filt                                            # (K, L)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) raw signal.
        Returns:
            y: (B, K, C, T) band-filtered signal per band.
        """
        B, C, T = x.shape
        K = self.num_bands
        filt = self._build_filters()                           # (K, L)
        # Depthwise conv: treat the (B*C) axis as the channel dimension,
        # apply each band filter → (B*C, K, T).
        x_flat = x.reshape(B * C, 1, T)
        # Use groups=1 so each band is applied to the full signal independently.
        # Padding = (L-1)//2 keeps output length = T.
        pad = (self.kernel_size - 1) // 2
        weight = filt.unsqueeze(1)                             # (K, 1, L)
        y = F.conv1d(x_flat, weight, padding=pad)              # (B*C, K, T)
        y = y.reshape(B, C, K, T).permute(0, 2, 1, 3).contiguous()
        return y                                               # (B, K, C, T)


class FilterBankBias(nn.Module):
    """
    Filter-bank C1 bias — shared across all transformer layers.

    Given a raw EEG trial and a per-channel per-sample visibility mask, produces
    K centered log-covariance matrices (one per band) that serve as static
    spatial attention biases at every encoder layer. Each layer's attention
    then weights these K biases with its own learnable β_{k,h} scalars.

    Constraints enforced:
      • Mask-aware: zeros masked samples BEFORE SincConv, then divides the
        covariance accumulator by the per-pair unmasked count. Prevents
        information about masked patches from leaking into S_k.
      • Ridge λI for SPD safety.
      • Padé [1,1] log map, same as C1.
      • Learnable per-band tangent center μ_k, stored in global 144×144 space.

    Args:
        sample_rate:        Hz of the raw signal.
        num_bands:          K.
        kernel_size:        SincConv FIR taps (odd).
        band_edges:         list of (f_lo, f_hi) in Hz, length K.
        learnable_cutoffs:  freeze cutoffs at init (default) or train them.
        total_channels:     size of the global channel space (μ_k lives here).
        eps_ridge:          ridge λ factor; λ = eps_ridge · tr(S_k)/C.
        learn_mu:           attach learnable per-band μ_k (default True).
    """
    def __init__(self, sample_rate, num_bands=5, kernel_size=65,
                 band_edges=None, learnable_cutoffs=False,
                 total_channels=TOTAL_GLOBAL_CHANNELS,
                 eps_ridge=1e-4, learn_mu=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.total_channels = total_channels
        self.eps_ridge = eps_ridge

        self.sinc = SincConv1d(
            sample_rate=sample_rate,
            num_bands=num_bands,
            kernel_size=kernel_size,
            band_edges=band_edges,
            learnable=learnable_cutoffs,
        )

        # Per-band tangent-space centers in global channel space.
        # Shape: (K, total_channels, total_channels). Init 0 → no centering
        # at the start, matches current μ init.
        if learn_mu:
            self.mu_log_bank = nn.Parameter(
                torch.zeros(num_bands, total_channels, total_channels)
            )
        else:
            self.register_parameter('mu_log_bank', None)

    def forward(self, x_raw, vis_mask, channel_idx):
        """
        Args:
            x_raw:       (B, C, T) raw signal after channel/trial normalization.
            vis_mask:    (B, C, T) float in {0,1} — 1 = visible sample. Can be
                         None → treated as all-visible (downstream / no mask).
            channel_idx: (C,) long — global channel indices.
        Returns:
            fb_log_S: (B, K, C, C) float — per-band centered log-covariance.
                      Reused by every transformer layer's FB bias.
        """
        B, C, T = x_raw.shape
        K = self.num_bands
        device = x_raw.device

        if vis_mask is None:
            vis_mask = torch.ones(B, C, T, device=device, dtype=x_raw.dtype)
        else:
            vis_mask = vis_mask.to(dtype=x_raw.dtype)

        # ── Mask the raw signal BEFORE SincConv (leakage fix) ──
        # SincConv's receptive field bleeds across mask boundaries, but zeroed
        # masked samples cannot propagate their original content — only a
        # decayed mix of visible neighbors. No content leak. A mild spectral
        # artifact near boundaries remains (acceptable; see design doc).
        x_masked = x_raw * vis_mask                             # (B, C, T)

        # ── SincConv decomposition ──
        y = self.sinc(x_masked)                                 # (B, K, C, T)

        # ── Re-apply visibility mask on the band-filtered output ──
        # SincConv's receptive field smears visibility: a "valid" sample in
        # the middle of a masked block gets a (small) non-zero value from
        # neighboring visible context. Zero those out again so the covariance
        # accumulator sums only over truly-visible time samples.
        y = y * vis_mask.unsqueeze(1)                           # (B, K, C, T)

        # ── Mask-aware covariance per band ──
        # S_k[b, i, j] = Σ_t y_k[b, i, t] y_k[b, j, t] / N_pairs[b, i, j]
        # where N_pairs = Σ_t vis[b, i, t] vis[b, j, t].
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            y_f32 = y.float()                                   # (B, K, C, T)
            # Cross-channel accumulator per band.
            # einsum: (B, K, C, T) × (B, K, C, T) → (B, K, C, C)
            cov_sum = torch.einsum('bkit,bkjt->bkij', y_f32, y_f32)

            vis_f32 = vis_mask.float()                          # (B, C, T)
            # Pairwise visible count: same across K bands.
            n_pairs = torch.einsum('bit,bjt->bij', vis_f32, vis_f32)  # (B, C, C)
            n_pairs = n_pairs.clamp_min(1.0).unsqueeze(1)       # (B, 1, C, C)

            S = cov_sum / n_pairs                               # (B, K, C, C)

            # ── Trace-normalize S to unit mean-diag per band ──
            # Padé [1,1] approximates log(S) accurately only when eigenvalues
            # of S lie near 1. Raw-signal band covariances can span orders of
            # magnitude across trials/channels, which pushes Padé far off the
            # true log and lets β amplify the error into Inf during training.
            # Rescaling by tr(S)/C anchors S near I regardless of signal scale.
            diag_trace = S.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # (B, K, 1)
            S = S / diag_trace.clamp_min(1e-6).unsqueeze(-1)    # (B, K, C, C), mean diag = 1
            eye = torch.eye(C, device=S.device, dtype=S.dtype).view(1, 1, C, C)

            # Ridge for SPD safety (absolute now, not relative — S is unit-trace).
            S = S + self.eps_ridge * eye

            # Padé [1,1] log-map, batched across (B, K): log(S) ≈ 2(S-I)(I+S)^{-1}
            # Collapse (B, K) to a single batch axis for torch.linalg.solve.
            S_flat = S.reshape(B * K, C, C).contiguous()
            I_flat = eye.expand(B * K, -1, -1, -1).reshape(B * K, C, C).contiguous()
            X = S_flat - I_flat
            logS_flat = torch.linalg.solve(I_flat + S_flat, 2.0 * X)
            logS = logS_flat.reshape(B, K, C, C)

            # NaN/Inf guard: fall back to first-order (S - I) per band.
            bad = torch.isnan(logS).any(dim=(-1, -2)) | torch.isinf(logS).any(dim=(-1, -2))
            if bad.any():
                fallback = (S - eye)
                logS = torch.where(bad.unsqueeze(-1).unsqueeze(-1), fallback, logS)

            # ── Defensive magnitude cap ──
            # Padé log of a unit-trace matrix is bounded by ~±2 analytically;
            # clamp to ±5 as a generous safety net so β cannot turn a single
            # outlier band into an attention-destroying Inf during fp16 cast.
            logS = logS.clamp(-5.0, 5.0)

        # ── Subtract per-band learnable center μ_k, symmetrized ──
        if self.mu_log_bank is not None:
            # Index μ_k along the channel dims: (K, total, total) → (K, C, C).
            mu_k = self.mu_log_bank[:, channel_idx][:, :, channel_idx]  # (K, C, C)
            mu_k = 0.5 * (mu_k + mu_k.transpose(-2, -1))
            logS = logS - mu_k.unsqueeze(0)                     # broadcast over B

        return logS.to(x_raw.dtype)                             # (B, K, C, C)


class AdaptiveRiemannianParallelAttention(nn.Module):
    """
    Parallel spatial-temporal attention with Riemannian spatial bias (C1).

    H/2 temporal heads + H/2 spatial heads, shared QKV.
    Spatial heads get additive Riemannian bias: score += α_h · log(S).
    Temporal heads are standard (optional EEG-RoPE on Q_t/K_t).

    Optional fine-tune extras:
      use_branch_gate=True  → per-layer learnable scalars scaling each
                              branch's output before the final concat+fc.
                              Init 1.0 → identical to ungated at epoch 0;
                              trains per-task spatial/temporal mixing.
    """
    def __init__(self, embed_dim, num_heads=8, total_channels=TOTAL_GLOBAL_CHANNELS,
                 dropout=0.1, att_dropout=0.1, spd_eps=1e-5,
                 log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None,
                 use_value_bias=True,
                 learn_mu_reference=True,
                 use_rope=False, rope_freq_min=0.5, rope_freq_max=50.0,
                 rope_learnable=True,
                 use_branch_gate=False,
                 use_filter_bank=False, fb_num_bands=5, fb_beta_init=0.0,
                 disable_bias=False,
                 use_temporal_bias=False, max_temporal_patches=128):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for parallel split"
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.heads_per_branch = num_heads // 2
        self.dim_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.use_value_bias = use_value_bias
        # Baseline ablation flag: when True, skip Riemannian bias computation
        # entirely. Propagated to AdaptiveRiemannianAttentionBias below.
        self.disable_bias = disable_bias
        # Temporal Riemannian bias (Run 6): adds Σ_temporal bias on temporal
        # attention, mirroring spatial Σ. Captures frequency content via
        # Wiener-Khinchin (autocorrelation = inverse FFT of power spectrum).
        # Layer-adaptive — computed from residual stream at each layer.
        self.use_temporal_bias = use_temporal_bias

        # Filter-bank bias (Run 4 FB-C1): per-layer (H_spatial × K) scalars that
        # weight the static raw-signal band biases precomputed at the encoder
        # level. fb_beta_init controls the starting value:
        #   • 0.0 (default) → bit-exact match with non-FB ckpt at load (probe).
        #   • ~0.2 (= 1/K with K=5) → uniform init for fresh pretraining; FB
        #     enters the bias from step 1 so gradient flows through β
        #     immediately rather than waiting for it to be discovered.
        self.use_filter_bank = use_filter_bank
        self.fb_num_bands = fb_num_bands
        if self.use_filter_bank:
            # Shape: (H_spatial, K). Each spatial head gets K band weights.
            self.fb_beta = nn.Parameter(
                torch.full((self.heads_per_branch, fb_num_bands), float(fb_beta_init))
            )
        self.use_rope = use_rope

        # Branch gate: per-layer learnable scalars that scale each branch's
        # output before the final concat+projection. At init both = 1.0 so
        # behavior is identical to ungated model. During fine-tune the gates
        # learn per-task mixing — expected pattern is spatial-heavy on
        # narrow-band tasks (MI), balanced on broader-spectrum tasks (emotion).
        self.use_branch_gate = use_branch_gate
        if self.use_branch_gate:
            self.branch_gate_s = nn.Parameter(torch.ones(1))
            self.branch_gate_t = nn.Parameter(torch.ones(1))

        # Shared QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

        # RoPE for temporal heads
        # learnable=True  → EEG-RoPE (frequencies init on EEG bands, learned)
        # learnable=False → Standard RoPE (fixed geometric series like LLMs)
        if use_rope:
            self.temporal_rope = EEGRoPE(
                dim=self.dim_head,
                freq_min=rope_freq_min,
                freq_max=rope_freq_max,
                learnable=rope_learnable,
            )

        # Adaptive Riemannian bias for spatial heads
        self.riemannian_bias = AdaptiveRiemannianAttentionBias(
            num_heads=self.heads_per_branch,
            total_channels=total_channels,
            eps=spd_eps,
            log_mode=log_mode,
            use_approx=use_approx,
            use_frechet=use_frechet,
            frechet_R_inv_sqrt=frechet_R_inv_sqrt,
            learn_mu_reference=learn_mu_reference,
            disable_bias=disable_bias,
        )

        # Temporal Riemannian bias for temporal heads (Run 6)
        if self.use_temporal_bias:
            self.temporal_riemannian_bias = TemporalRiemannianAttentionBias(
                num_heads=self.heads_per_branch,
                max_patches=max_temporal_patches,
                eps=spd_eps,
                learn_mu_reference=learn_mu_reference,
                disable_bias=disable_bias,
            )
        else:
            self.temporal_riemannian_bias = None

        # Geometric value mixing: V' = V + β_h · (L @ V)
        # Injects covariance structure into attention values — complementary
        # to the score bias (which controls routing, not feature mixing).
        # Initialized to 0 so model starts as standard attention.
        if use_value_bias:
            self.value_beta = nn.Parameter(torch.zeros(self.heads_per_branch))

    def forward(self, x_norm, num_chan, residual=None, channel_idx=None,
                mask=None, fb_log_S=None):
        """
        Args:
            x_norm:   (B, L, D) normalized input (post-LayerNorm)
            num_chan: number of EEG channels C in this batch
            residual: (B, L, D) raw residual stream (pre-LayerNorm)
            channel_idx: (C,) long tensor — global channel indices
            mask:     (B, L) boolean — True = masked token (pretraining only)
            fb_log_S: (B, K, C, C) float — per-band centered log(S_k) from
                      the encoder-level FilterBankBias. Reused across all
                      layers. Ignored unless use_filter_bank=True.
        Returns:
            out: (B, L, D) attention output
        """
        B, L, D = x_norm.shape
        assert L % num_chan == 0
        N = L // num_chan
        C = num_chan
        H = self.num_heads
        H2 = self.heads_per_branch
        d = self.dim_head

        # ── Compute Riemannian bias from residual stream ──
        bias_source = residual if residual is not None else x_norm
        x_space = rearrange(bias_source, 'b (n c) d -> (b n) c d', c=C)

        mask_space = None
        if mask is not None:
            mask_space = rearrange(mask, 'b (n c) -> (b n) c', c=C)
            x_space = x_space * (~mask_space).unsqueeze(-1).float()

        # Riemannian bias (with optional learnable tangent-space centering)
        riem_bias, L_n = self.riemannian_bias(
            x_space, channel_idx, mask_space=mask_space,
        )

        # ── Filter-bank bias (Run 4 FB-C1) ──
        # β_{h,k} · (log(S_k) − μ_k) summed over K bands, added to the
        # existing α·log(S) + C3 bias. Broadcast the static (B, K, C, C)
        # bias across the N temporal positions to match (B*N, H_s, C, C).
        if self.use_filter_bank and fb_log_S is not None:
            # einsum: β (H_s, K) × fb (B, K, C, C) → (B, H_s, C, C)
            fb_bias_BHCC = torch.einsum('hk,bkij->bhij', self.fb_beta, fb_log_S)
            # Broadcast over N temporal positions → (B*N, H_s, C, C).
            # We expand to (B, N, H_s, C, C) then collapse to match riem_bias.
            fb_bias_expand = fb_bias_BHCC.unsqueeze(1).expand(B, N, H2, C, C)
            fb_bias_flat = fb_bias_expand.reshape(B * N, H2, C, C)
            riem_bias = riem_bias + fb_bias_flat.to(riem_bias.dtype)

        # ── Shared QKV ──
        qkv = self.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_t, q_s = q[:, :H2], q[:, H2:]
        k_t, k_s = k[:, :H2], k[:, H2:]
        v_t, v_s = v[:, :H2], v[:, H2:]

        # ── Temporal attention branch ──
        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)
        v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=C)

        # EEG-RoPE: inject relative temporal position via rotation.
        # Applied BEFORE attention so temporal scores reflect relative offsets.
        # RoPE rotates ALL positions (including masked tokens) — the model
        # needs position info for reconstruction. Shortcut prevention comes
        # from the masking STRATEGY, not from modifying RoPE.
        if self.use_rope:
            q_t, k_t = self.temporal_rope(q_t, k_t)

        # ── Temporal Riemannian bias (Run 6) ──
        # Compute Σ_temporal per channel from patch-sequence embeddings.
        # By Wiener-Khinchin, this autocorrelation matrix encodes frequency
        # content (off-diagonal banding patterns reveal periodicities).
        if self.use_temporal_bias:
            # Per-channel patch-sequence embeddings: (B*C, N, D)
            x_temporal = rearrange(bias_source, 'b (n c) d -> (b c) n d', c=C)
            mask_temporal = None
            if mask is not None:
                mask_temporal = rearrange(mask, 'b (n c) -> (b c) n', c=C)
                x_temporal = x_temporal * (~mask_temporal).unsqueeze(-1).float()

            temp_bias, _temp_L = self.temporal_riemannian_bias(
                x_temporal, mask_temporal=mask_temporal,
            )
            # temp_bias shape: (B*C, num_heads, N, N) — additive attention bias

            # Manual N×N temporal attention with additive bias.
            # SDPA with float attn_mask falls back from Flash Attention to
            # slower memory-efficient/math backends, so we go manual.
            #
            # CRITICAL: keep q@k in autocast dtype (fp16/bf16) — wrapping the
            # whole block in autocast(enabled=False) was forcing fp32 matmul
            # on (B*C, H, N, N), the dominant cost (~5x heavier than spatial
            # because N=48 vs C≈22). Only the softmax goes to fp32 for
            # numerical stability; the bias is downcast to score dtype before
            # adding (head_scales init at 0 → bias bounded → fp16-safe).
            score_t = (q_t @ k_t.transpose(-2, -1)) / (d ** 0.5)
            score_t = score_t + temp_bias.to(score_t.dtype)
            with torch.amp.autocast('cuda', enabled=False), \
                 torch.amp.autocast('cpu', enabled=False), \
                 torch.amp.autocast('mps', enabled=False):
                score_t = score_t.float().softmax(dim=-1)
            score_t = F.dropout(score_t, p=self.att_dropout, training=self.training)
            out_t = score_t.to(v_t.dtype) @ v_t
        else:
            # Standard N×N temporal self-attention (no temporal bias)
            out_t = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=self.att_dropout if self.training else 0.0,
            )

        out_t = rearrange(out_t, '(b c) h n d -> b h (n c) d', b=B, c=C)

        # ── Spatial attention (Riemannian-biased) ──
        q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=C)
        k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=C)
        v_s = rearrange(v_s, 'b h (n c) d -> (b n) h c d', c=C)

        # ── Geometric value mixing: V' = V + β_h · (L @ V) ──
        # Score bias (α·L on attention logits) controls WHERE to attend.
        # Value bias (β·L@V) controls WHAT information flows — complementary.
        # Runs in native dtype (fp16) to save VRAM; no softmax, fp32 not needed.
        #
        # LEAK PREVENTION: L @ V mixes all channels' values. If masked channels
        # keep their mask-token V vectors, L routes mask-token info into
        # unmasked channels — a subtle leak (model sees "which channels are
        # masked" through the value path). Zero out masked V before mixing.
        if self.use_value_bias:
            beta_h = self.value_beta.view(1, H2, 1, 1)
            L_exp = L_n.unsqueeze(1).to(v_s.dtype)   # (B*N, 1, C, C)
            v_for_geo = v_s
            if mask is not None:
                mask_v = rearrange(mask, 'b (n c) -> (b n) c', c=C)
                v_for_geo = v_s * (~mask_v).unsqueeze(1).unsqueeze(-1).float()
            v_geo = L_exp @ v_for_geo                  # (B*N, H2, C, d)
            v_s = v_s + beta_h * v_geo

        # Manual attention with Riemannian bias.
        # SDPA with float attn_mask falls back from Flash Attention to slower
        # backends, so we go manual. Same dtype discipline as temporal:
        # q@k stays in autocast dtype (fp16/bf16), only softmax goes to fp32.
        # The bias is bounded by head_scales (init 0) so fp16 add is safe.
        score = (q_s @ k_s.transpose(-2, -1)) / (d ** 0.5)
        score = score + riem_bias.to(score.dtype)
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            score = score.float().softmax(dim=-1)
        score = F.dropout(score, p=self.att_dropout, training=self.training)
        out_s = score.to(v_s.dtype) @ v_s

        out_s = rearrange(out_s, '(b n) h c d -> b h (n c) d', b=B, n=N)

        # ── Branch gate (optional): scale each branch before concat ──
        # Multiplicatively scales out_s and out_t before the final projection.
        # Mathematically equivalent to scaling the corresponding column blocks
        # of self.fc, so the learned gates directly answer "how much does each
        # branch contribute to this layer's output." Gated behind
        # use_branch_gate; init 1.0 keeps behavior identical to ungated model.
        #
        # LINEAR-PROBE AUTOGRAD GUARD: when the encoder is frozen (linear probe
        # with only the gate trainable), the gate's backward only needs the
        # *values* of out_s / out_t — not their autograd graphs. Tracing the
        # graph back through frozen encoder ops (torch.linalg.solve in the Padé
        # log, SDPA backward, advanced indexing on mu_log) hits known MPS
        # bugs where kernels return uninitialized memory. Detach when the
        # encoder's QKV weight is frozen — mathematically identical in linear
        # probe (no grad flows there anyway), sidesteps the MPS crash, and
        # leaves full fine-tune untouched (qkv.weight.requires_grad=True →
        # no detach, normal gradient flow).
        if self.use_branch_gate:
            if not self.qkv.weight.requires_grad:
                out_s = out_s.detach()
                out_t = out_t.detach()
            out_s = out_s * self.branch_gate_s
            out_t = out_t * self.branch_gate_t

        # ── Output projection ──
        # Concatenate temporal + spatial heads, then project.
        out = torch.cat([out_t, out_s], dim=1)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.fc(out)
        return self.dropout(out)


class AdaptiveRiemannianParallelTransformer(nn.Module):
    """
    Parallel head-split transformer with adaptive Riemannian spatial attention.

    Supports variable channel counts: the learned SPD reference lives in the
    global 144-channel space. Each batch provides its channel indices, and the
    relevant C×C submatrix is extracted at runtime.

    Args:
        embed_dim: Total embedding dimension
        nhead: Number of attention heads (must be even)
        total_channels: Size of global channel space (default 144)
        mlp_ratio: MLP hidden dimension ratio
        drop: Dropout probability
        att_drop: Attention dropout probability
        drop_path: Stochastic depth probability
        act: Activation function class
        norm: Normalization layer class
        spd_eps: SPD regularization constant
        log_mode: 'approx', 'pade', or 'eigh' (see AdaptiveLogMap)
        use_approx: DEPRECATED — kept for backward compat.
        use_frechet: If True, pre-whiten S with frozen Fréchet mean
        frechet_R_inv_sqrt: (C, C) tensor — precomputed R^{-1/2}
    """
    def __init__(self, embed_dim, nhead=8, total_channels=TOTAL_GLOBAL_CHANNELS,
                 mlp_ratio=4, drop=0.0, att_drop=0.0, drop_path=0.0, act=nn.GELU,
                 norm=nn.LayerNorm, spd_eps=1e-5, log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None,
                 use_value_bias=True,
                 learn_mu_reference=True,
                 use_rope=False, rope_freq_min=0.5, rope_freq_max=50.0,
                 rope_learnable=True,
                 use_branch_gate=False,
                 use_filter_bank=False, fb_num_bands=5, fb_beta_init=0.0,
                 disable_bias=False,
                 use_temporal_bias=False, max_temporal_patches=128):
        super().__init__()

        self.attn = AdaptiveRiemannianParallelAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            total_channels=total_channels,
            dropout=drop,
            att_dropout=att_drop,
            spd_eps=spd_eps,
            log_mode=log_mode,
            use_approx=use_approx,
            use_frechet=use_frechet,
            frechet_R_inv_sqrt=frechet_R_inv_sqrt,
            use_value_bias=use_value_bias,
            learn_mu_reference=learn_mu_reference,
            use_rope=use_rope,
            rope_freq_min=rope_freq_min,
            rope_freq_max=rope_freq_max,
            rope_learnable=rope_learnable,
            use_branch_gate=use_branch_gate,
            use_filter_bank=use_filter_bank,
            fb_num_bands=fb_num_bands,
            fb_beta_init=fb_beta_init,
            disable_bias=disable_bias,
            use_temporal_bias=use_temporal_bias,
            max_temporal_patches=max_temporal_patches,
        )

        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan, channel_idx=None, mask=None, fb_log_S=None):
        """
        Args:
            x: (B, L, D) input tensor where L = N * num_chan
            num_chan: number of channels C in this batch
            channel_idx: (C,) long tensor — global channel indices
            mask:     (B, L) boolean — True = masked token (pretraining only).
            fb_log_S: (B, K, C, C) — per-band centered log(S_k) from the
                      encoder-level FilterBankBias. Threaded through from
                      the pretraining / downstream forward. None = FB disabled.
        """
        # Pass normalized input for QKV, raw residual for Riemannian bias.
        x = x + self.drop_path1(
            self.attn(self.norm1(x), num_chan, residual=x,
                      channel_idx=channel_idx, mask=mask, fb_log_S=fb_log_S)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
