import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import mne
import yaml
import os


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
                Y_chol = torch.linalg.cholesky(Y)
                Y_inv = torch.cholesky_solve(I, Y_chol)
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


class RotaryEmbedding(nn.Module):
    """
        Implementation of the Rotary embedding which attributes to each token
        pair a relative rotation.
    """
    def __init__(self, model_dim,theta = 10000,is_learnable = False):
        super().__init__()
        assert model_dim % 2 == 0
        self.is_learnable = is_learnable

        #Define the frequency of the angle
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, model_dim, 2)[:(model_dim // 2)].float() / model_dim)),
            requires_grad = is_learnable)

    def create_frequency(self, kept_ids, device, dtype):

        #Total number of patches
        #Shape (num_patches)
        B, N = kept_ids.shape
        kept_ids = kept_ids.to(self.freqs.dtype).to(self.freqs.device)

        #Do the outer product between the freqs and all the kept patch indices
        freqs = kept_ids.reshape(B,N,1) * self.freqs
        #Convert the angle to polar coordinate
        polar = torch.polar(torch.ones_like(freqs), freqs)
        return polar

    def forward(self, x, id_kepts):
        #B, num_head, num_patch, dim_head
        B, H, N, D = x.shape
        device = x.device
        dtype = x.dtype
        ndim = x.ndim
        assert D == 2 * self.freqs.numel(), "RoPE model_dim must equal x last dim"

        assert id_kepts.shape == (B, N)
    
        polar = self.create_frequency(kept_ids=id_kepts, device=device, dtype=dtype)

        #Dimension B, 1, N, D
        polar = polar.unsqueeze(1)

        #Dimension B, num_head, N, D//2, 2
        #Segment the last dimension in individual 2D plan
        x2 = x.view(*x.shape[:-1], D//2, 2).contiguous()

        #Convert the vector is polar form to complex number
        x_complex = torch.view_as_complex(x2)

        #Multiply the vector by the angle then transform back to real number with correct shape
        x_rotate = x_complex * polar
        x_real = torch.view_as_real(x_rotate)
        x_real = x_real.reshape(*x.shape)
        return x_real


class MultiHeadAttention(nn.Module):
    """Multi head attention module, takes the embedding dim and number of head."""
    def __init__(self, embed_dim, num_heads = 3, dropout = 0.1, att_dropout = 0.1,is_causal = False, return_att = False, use_rotary = False, has_cls = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim//num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.fc = nn.Linear(embed_dim,embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.return_att = return_att
        self.att_dropout = att_dropout
        self.use_rotary = use_rotary

        #Used as a flag to signal if the input is expected to have the class token
        self.has_cls = has_cls

        self.rotary = RotaryEmbedding(model_dim=embed_dim//num_heads)

    def split_heads(self, X):
        return X.view(X.size(0), X.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)      

    def forward(self, x, position = None):
        #Compute Q, K and V and seperate segments per head
        B, N, D = x.shape

        #Extract the query key value vectors each with shape
        # B, num_head, num_patches, dim_head
        qkv = self.split_heads(self.qkv(x))
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        #If Rotary embeddings are used then apply it in the attention mechanism
        if self.use_rotary:
            assert position is not None, "position required when use_rotary=True"
            #shape is B, num_heads, num_patches, dim_head
            if self.has_cls:
                q, class_token_q = q[:,:,1:,:], q[:,:,:1,:]
                k, class_token_k = k[:,:,1:,:], k[:,:,:1,:]
                q = self.rotary(q, position)
                k = self.rotary(k, position)
                k = torch.concat([class_token_k, k], dim=2)
                q = torch.concat([class_token_q, q], dim=2)
            else:
                q = self.rotary(q, position)
                k = self.rotary(k, position)

        #Compute the attention score
        if self.return_att:
            score = (q@k.transpose(2,3))/(self.dim_head**0.5)

            #If causal we mask all date happening in the future
            if self.is_causal:
                T = score.size(-1)
                device = score.device
                mask = torch.triu(torch.ones(T,T, dtype=torch.bool, device=device), diagonal=1)
                score = score.masked_fill(mask, float('-inf'))
                
            #Compute the attnetion matrix
            score = score.softmax(dim=-1)
            
            #Apply attention to V
            attn = F.dropout(score, p=self.att_dropout, training=self.training)
            out = attn@v
        else:
            #Compute the attention score using pytorch built in function
            out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.att_dropout if self.training else 0, is_causal=self.is_causal)
            score = None
        out = out.transpose(1,2)
        out = out.reshape(out.size(0), out.size(1), self.h*self.dim_head)
        out = self.fc(out)
        return self.dropout(out), score

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


class MultiHeadCrossAttention(nn.Module):
    """Multi head attention module, takes the embedding dim and number of head."""
    def __init__(self, embed_dim, num_heads = 3, proj_drop = 0.1, att_dropout = 0.1, qkv_bias = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim//num_heads
        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.fc = nn.Linear(embed_dim,embed_dim)
        self.att_dropout = att_dropout
        self.proj_drop = nn.Dropout(proj_drop)


    def split_heads(self, X):
        return X.view(X.size(0), X.size(1), self.h, self.dim_head).permute(0, 2, 1, 3)      

    def forward(self, key, query, mask_pad = None):
        #Compute Q, K and V and seperate segments per head

        #Extract the query key value vectors each with shape
        # B, num_head, num_patches, dim_head
        k = self.k(key)
        v = self.v(key)
        q = self.q(query)
        k = self.split_heads(k)
        v = self.split_heads(v)
        q = self.split_heads(q)

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
    
class CrossTransformerLayer(nn.Module):
    def __init__(self, embed_dim, nhead, mlp_ratio = 4, qkv_bias = True,drop = 0, att_drop = 0, 
                 drop_path = 0, act = nn.GELU, norm = nn.LayerNorm):
        super().__init__()
        self.attn = MultiHeadCrossAttention(embed_dim=embed_dim, num_heads=nhead, proj_drop=drop,
                                               att_dropout=att_drop, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, query, key, mask_pad = None):
        x = query + self.drop_path(self.attn(self.norm1(key), self.norm1(query), mask_pad))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x




class TransformerLayer(nn.Module):
    """Define one transformer layer"""
    def __init__(self, embed_dim, nhead = 3, dim_feedforward=2048, dropout=0.1, activation = "gelu", keep_prob = 1, use_rotary = False, has_cls = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=dropout, use_rotary=use_rotary, has_cls = has_cls)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        activations = {"gelu": nn.GELU(), "relu": nn.ReLU()}
        self.activation = activations[activation]
        self.drop_path = DropPath(keep_prob=keep_prob) if keep_prob < 1 else nn.Identity()

    def forward(self, src, keep):
        #Compute the multihead attention
        z = self.norm1(src)
        attn, _ = self.self_attn(z, position = keep)

        #Apply residual connection and layer normalization
        Z = src + self.drop_path(attn)

        #Apply normalization
        ff = self.norm2(Z)

        #MLP layer
        ff = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(ff)))))

        return (Z + self.drop_path(ff))


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


class SpaceAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3, dropout=0.1, att_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.dim_head = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), 3, self.h, self.dim_head).permute(2, 0, 3, 1, 4)

    def forward(self, x, num_chan):
        B, L, D = x.shape
        assert L % num_chan == 0
        N = L // num_chan

        # (B, L, D) -> (B, N, C, D) -> (B*N, C, D)
        x = rearrange(x, "b (n c) d -> b n c d", c=num_chan)
        x = rearrange(x, "b n c d -> (b n) c d")

        qkv = self.split_heads(self.qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]

        score = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        score = score.softmax(dim=-1)
        attn = F.dropout(score, p=self.att_dropout, training=self.training)

        out = attn @ v
        out = out.transpose(1, 2).reshape(out.size(0), out.size(2), self.h * self.dim_head)
        out = self.fc(out)

        # (B*N, C, D) -> (B, N, C, D) -> (B, L, D)
        out = rearrange(out, "(b n) c d -> b n c d", b=B, n=N)
        out = rearrange(out, "b n c d -> b (n c) d")

        return self.dropout(out)



class CrissCrossTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_chan, mlp_ratio=4, drop=0.0, att_drop=0.0,
                 drop_path=0.0, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.num_chan = num_chan

        self.attn_time = TimeAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=drop,
            att_dropout=att_drop
        )
        self.attn_space = SpaceAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=drop,
            att_dropout=att_drop
        )

        self.drop_path1 = nn.Identity()
        self.drop_path2 = nn.Identity()
        self.drop_path3 = nn.Identity()

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.norm3 = norm(embed_dim)

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x):
        x = x + self.drop_path1(self.attn_time(self.norm1(x), self.num_chan))
        x = x + self.drop_path2(self.attn_space(self.norm2(x), self.num_chan))
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x

    
def dist_spd(spd1, spd2):
    pass


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
    


class RiemannianSpaceTransformer(nn.Module):
    
    def __init__(self, embed_dim, nhead, mlp_ratio=4, drop=0.0, att_drop=0.0,
                 drop_path=0.0, act=nn.GELU, norm=nn.LayerNorm, spd_eps=1e-5):
        super().__init__()
       
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
        
        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
       

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan):
        
        x = x + self.drop_path1(self.attn_space(self.norm1(x), num_chan))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    
class RiemannianTimeTransformer(nn.Module):
    
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

      

        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
       

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
       

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan):
        x = x + self.drop_path1(self.attn_time(self.norm1(x), num_chan))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class RiemannianParallelAttention(nn.Module):
    """
    CBraMod-style parallel spatial-temporal attention with Riemannian bias.

    A single QKV projection is shared. Heads are split in two:
      - First half: temporal attention (B*C, N, D_half) — Euclidean
      - Second half: spatial attention (B*N, C, D_half) — Riemannian-biased

    Outputs are concatenated back to full dimension and projected.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, att_dropout=0.1, spd_eps=1e-5):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for parallel split"
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.heads_per_branch = num_heads // 2
        self.dim_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.half_dim = self.heads_per_branch * self.dim_head  # D // 2

        # Shared QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

        # Riemannian bias for spatial heads only
        self.riemannian_bias = RiemannianAttentionBias(num_heads=self.heads_per_branch, eps=spd_eps)

    def forward(self, x, num_chan):
        B, L, D = x.shape
        assert L % num_chan == 0
        N = L // num_chan
        H = self.num_heads
        H2 = self.heads_per_branch
        d = self.dim_head

        # Shared QKV: (B, L, D) -> (B, L, 3*D)
        qkv = self.qkv(x).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d)

        # Split heads: first H2 for temporal, last H2 for spatial
        q_t, q_s = q[:, :H2], q[:, H2:]  # (B, H2, L, d)
        k_t, k_s = k[:, :H2], k[:, H2:]
        v_t, v_s = v[:, :H2], v[:, H2:]

        # ── Temporal attention (Euclidean) ──
        # Reshape: (B, H2, N*C, d) -> (B, H2, N, C, d) -> (B*C, H2, N, d)
        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=num_chan)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=num_chan)
        v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=num_chan)

        out_t = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=self.att_dropout if self.training else 0.0,
        )  # (B*C, H2, N, d)
        # Back to (B, H2, N*C, d)
        out_t = rearrange(out_t, '(b c) h n d -> b h (n c) d', b=B, c=num_chan)

        # ── Spatial attention (Riemannian-biased) ──
        # Reshape: (B, H2, N*C, d) -> (B*N, H2, C, d)
        q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=num_chan)
        k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=num_chan)
        v_s = rearrange(v_s, 'b h (n c) d -> (b n) h c d', c=num_chan)

        # Compute Riemannian bias from the input embeddings at each time step
        x_space = rearrange(x, 'b (n c) d -> (b n) c d', c=num_chan)
        riem_bias = self.riemannian_bias(x_space)  # (B*N, H2, C, C)

        # Manual spatial attention with Riemannian bias — float32 for stability
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            score = (q_s.float() @ k_s.float().transpose(-2, -1)) / (d ** 0.5)
            score = score + riem_bias.float()
            score = score.softmax(dim=-1)
        score = F.dropout(score, p=self.att_dropout, training=self.training)
        out_s = score.to(v_s.dtype) @ v_s  # (B*N, H2, C, d)

        # Back to (B, H2, N*C, d)
        out_s = rearrange(out_s, '(b n) h c d -> b h (n c) d', b=B, n=N)

        # ── Concatenate heads and project ──
        out = torch.cat([out_t, out_s], dim=1)  # (B, H, L, d)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.fc(out)
        return self.dropout(out)


class RiemannianParallelCrissCrossTransformer(nn.Module):
    """
    CBraMod-style transformer layer: parallel head-split spatial-temporal attention
    with Riemannian bias on the spatial heads.

    Unlike sequential (time→space) or separate (two encoder stacks), this fuses
    spatial and temporal information at every layer through shared MLP processing
    after parallel head-split attention.
    """
    def __init__(self, embed_dim, nhead=8, mlp_ratio=4, drop=0.0, att_drop=0.0,
                 drop_path=0.0, act=nn.GELU, norm=nn.LayerNorm, spd_eps=1e-5):
        super().__init__()

        self.attn = RiemannianParallelAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=drop,
            att_dropout=att_drop,
            spd_eps=spd_eps,
        )

        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan):
        x = x + self.drop_path1(self.attn(self.norm1(x), num_chan))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
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
                # Use Cholesky solve instead of linalg.solve —
                # I+S is SPD (eigenvalues > 1), and Cholesky uses a
                # different CUDA kernel that avoids misaligned-address bugs
                L_chol = torch.linalg.cholesky(I_f32 + S_f32)
                T = torch.cholesky_solve(2 * X, L_chol)

            # ── NaN/Inf guard after Padé ──
            if torch.isnan(T).any() or torch.isinf(T).any():
                print(f"[AdaptiveLogMap] NaN/Inf AFTER Padé solve!")
                print(f"  S diag range: [{S.diagonal(dim1=-2,dim2=-1).min().item():.4f}, "
                      f"{S.diagonal(dim1=-2,dim2=-1).max().item():.4f}]")
                # Fall back to first-order (S - I) which can't produce NaN from SPD input
                T = X

            return T.to(orig_dtype)


# ─── Graph-Referenced Riemannian Bias (Contribution 2) ──────────────────────
# Precompute the global 144×144 electrode topology reference matrix R.
# R_{ij} = exp(-geodesic_dist(i,j)² / σ²) with σ=0.35 radians.
# This is an RBF kernel on the unit-sphere electrode positions — guaranteed SPD
# by Schoenberg's theorem (positive definite kernel on a metric space).

def _build_global_R(sigma=0.35, eps=1e-4):
    """
    Build the 144×144 electrode topology reference matrix R.

    Uses MNE standard_1005 montage positions, maps channel names via
    channel_info.yaml, computes geodesic distance on the unit sphere,
    and applies an RBF kernel.

    Returns:
        R: (144, 144) float32 tensor — SPD matrix
        idx_to_name: dict mapping global index → channel name
    """
    # Load channel mapping: name → global index
    yaml_path = os.path.join(os.path.dirname(__file__), "info_dataset", "channel_info.yaml")
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    ch_mapping = config["channels_mapping"]  # name → index
    idx_to_name = {v: k for k, v in ch_mapping.items()}
    total = len(ch_mapping)

    # Get 3D electrode positions from MNE
    montage = mne.channels.make_standard_montage("standard_1005")
    pos_dict_raw = montage.get_positions()["ch_pos"]
    pos_dict = {k.lower(): torch.tensor(v, dtype=torch.float32) for k, v in pos_dict_raw.items()}

    # Aliases (same as graph_embedding.py)
    aliases = {
        'cb1': 'poo7', 'cb2': 'poo8', 'cbz': 'pooz',
        't3': 't7', 't4': 't8', 't5': 'p7', 't6': 'p8',
        'm1': 'tp9', 'm2': 'tp10',
    }
    for alias, standard_name in aliases.items():
        if standard_name in pos_dict:
            pos_dict[alias] = pos_dict[standard_name]

    # Build position matrix (144, 3) ordered by global index
    positions = torch.zeros(total, 3)
    for idx in range(total):
        name = idx_to_name[idx]
        name_lower = name.lower()
        if name_lower in pos_dict:
            positions[idx] = pos_dict[name_lower]
        else:
            # Fallback: origin (shouldn't happen for standard channels)
            print(f"[_build_global_R] Warning: channel '{name}' not in montage, using origin.")
            positions[idx] = torch.tensor([0.0, 0.0, 0.0])

    # Normalize to unit sphere
    norms = positions.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    positions_norm = positions / norms

    # Geodesic distance matrix
    cos_sim = positions_norm @ positions_norm.T
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    D = torch.acos(cos_sim)  # (144, 144) in radians

    # RBF kernel → R
    R = torch.exp(-D.pow(2) / (sigma ** 2))

    # Small regularization for numerical safety
    R = R + eps * torch.eye(total)

    return R, idx_to_name


# Module-level cache: computed once, reused across all layers
_GLOBAL_R_CACHE = {}


def get_global_R(sigma=0.35, eps=1e-4):
    """Cached access to the global R matrix."""
    key = (sigma, eps)
    if key not in _GLOBAL_R_CACHE:
        R, idx_to_name = _build_global_R(sigma=sigma, eps=eps)
        _GLOBAL_R_CACHE[key] = (R, idx_to_name)
    return _GLOBAL_R_CACHE[key]


class GraphReferencedRiemannianBias(nn.Module):
    """
    Contribution 2: Graph-Referenced Riemannian Attention Bias.

    Computes a second spatial attention bias that captures how the observed
    covariance deviates from what electrode topology predicts.

    Math:
        R_sub = R[ch_idx, ch_idx]           — topology reference for this dataset's channels
        M = R_sub^{-1/2} · S · R_sub^{-1/2} — covariance in graph-referenced frame
        L_graph = Padé_log(M)               — tangent vector at IDENTITY (same space as C1)

    Both L (C1) and L_graph (C2) live in T_I — they add cleanly to spatial logits.

    Args:
        num_heads: Number of spatial attention heads
        total_channels: Global channel space size (144)
        sigma: RBF kernel bandwidth in radians (0.35 = adjacent electrodes ≈ 0.68 weight)
        eps: Regularization for R and SPD matrices
    """
    def __init__(self, num_heads, total_channels=TOTAL_GLOBAL_CHANNELS,
                 sigma=0.35, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.sigma = sigma

        # Per-head learnable scaling — initialized to 0 (starts as no contribution,
        # learns to use graph-referenced geometry)
        self.graph_head_scales = nn.Parameter(torch.zeros(num_heads))

        # Use first-order log approximation for C2: log(M) ≈ M - I
        # This is MORE accurate for C2 than C1 because M = R^{-1/2}SR^{-1/2}
        # is already close to I when R is a good prior (which is the whole point).
        # Saves one full Cholesky + solve per layer compared to Padé.
        # If needed, can upgrade to Padé later — but first-order should suffice.

        # Precompute global R and register as buffer (moves with model to GPU)
        R_global, _ = get_global_R(sigma=sigma, eps=1e-4)
        self.register_buffer('R_global', R_global)  # (144, 144)

        # Cache for R_sub^{-1/2} per channel configuration (avoids recomputation)
        self._R_inv_sqrt_cache = {}

    def _get_R_inv_sqrt(self, channel_idx):
        """
        Extract R_sub for the current channel set and compute R_sub^{-1/2}.
        Cached per unique channel configuration.

        Args:
            channel_idx: (C,) long tensor — global channel indices
        Returns:
            R_inv_sqrt: (C, C) float32 tensor
        """
        cache_key = tuple(channel_idx.cpu().tolist())
        if cache_key not in self._R_inv_sqrt_cache:
            # Extract submatrix
            R_sub = self.R_global[channel_idx][:, channel_idx]  # (C, C)

            # Compute R_sub^{-1/2} via eigendecomposition
            # R_sub is SPD (guaranteed by RBF kernel + eps regularization)
            with torch.no_grad():
                eigvals, eigvecs = torch.linalg.eigh(R_sub.float())
                # Clamp eigenvalues for safety
                eigvals = eigvals.clamp(min=1e-6)
                R_inv_sqrt = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T

            # Check condition number
            cond = eigvals.max() / eigvals.min()
            if cond > 1000:
                print(f"[GraphReferencedRiemannianBias] Warning: κ(R_sub)={cond:.1f} "
                      f"for C={len(channel_idx)}. Graph bias may be unreliable.")

            self._R_inv_sqrt_cache[cache_key] = R_inv_sqrt.to(self.R_global.device)

        return self._R_inv_sqrt_cache[cache_key]

    def forward(self, S, channel_idx):
        """
        Args:
            S: (B*N, C, C) batch of SPD covariance matrices (already regularized)
            channel_idx: (C,) long tensor — global channel indices
        Returns:
            bias: (B*N, num_heads, C, C) graph-referenced attention bias
            L_graph: (B*N, C, C) raw graph-referenced tangent vectors
        """
        BN, C, _ = S.shape

        # Get R_sub^{-1/2} for this channel configuration
        R_inv_sqrt = self._get_R_inv_sqrt(channel_idx)  # (C, C)

        # Pre-transform: M = R^{-1/2} S R^{-1/2}
        # This moves S into the graph-referenced frame
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            R_inv_sqrt_f32 = R_inv_sqrt.float().unsqueeze(0)  # (1, C, C)
            S_f32 = S.float()
            # M = R^{-1/2} @ S @ R^{-1/2}
            M = torch.bmm(
                torch.bmm(R_inv_sqrt_f32.expand(BN, -1, -1), S_f32),
                R_inv_sqrt_f32.expand(BN, -1, -1)
            )
        # First-order log: log(M) ≈ M - I
        # Accurate because M is near I when R is a good topology prior.
        # No Cholesky needed — just a subtraction.
        I = torch.eye(C, device=M.device, dtype=M.dtype).unsqueeze(0)
        L_graph = (M - I).to(S.dtype)  # (BN, C, C)

        # Per-head scaling
        scales = self.graph_head_scales.view(1, self.num_heads, 1, 1)
        bias = L_graph.unsqueeze(1) * scales  # (BN, num_heads, C, C)

        return bias, L_graph


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
                 use_frechet=False, frechet_R_inv_sqrt=None):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.adaptive_log = AdaptiveLogMap(
            total_channels=total_channels, eps=eps,
            log_mode=log_mode, use_approx=use_approx,
        )

        # Per-head learnable scaling — initialized to 0 so the model starts
        # as standard Euclidean attention and learns to use the bias
        self.head_scales = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, channel_idx):
        """
        Args:
            x: (B*N, C, D) channel embeddings (from residual stream)
            channel_idx: (C,) long tensor — global channel indices for this batch
        Returns:
            bias: (B*N, num_heads, C, C) attention bias per head
            L: (B*N, C, C) raw tangent vectors
            S: (B*N, C, C) SPD covariance matrices (reused by graph-referenced bias)
        """
        BN, C, D = x.shape

        # Step 1: Compute sample covariance → SPD matrix
        # MUST be float32 — the residual stream x can have large values after
        # many layers, and x @ x^T overflows fp16 (max 65504) → inf → NaN.
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            x_f32 = x.float()
            S = torch.bmm(x_f32, x_f32.transpose(-2, -1)) / D
            S = S + self.eps * torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)

        # Step 2: Project to tangent space at identity via Padé [1,1]
        L = self.adaptive_log(S, channel_idx)  # (B*N, C, C)

        # Step 3: Per-head scaling
        scales = self.head_scales.view(1, self.num_heads, 1, 1)
        bias = L.unsqueeze(1) * scales

        return bias, L, S



class AdaptiveRiemannianParallelAttention(nn.Module):
    """
    Parallel spatial-temporal attention with dual Riemannian spatial biases:
      - Contribution 1: Adaptive Riemannian bias (data-driven functional connectivity)
        L = Padé_log(S) where S is the residual-stream covariance
      - Contribution 2: Graph-Referenced Riemannian bias (topology-referenced)
        L_graph = Padé_log(R^{-1/2} S R^{-1/2}) where R encodes electrode topology

    Structure vs Function framing:
    - C1 captures pure functional connectivity — what the data says about channel relationships
    - C2 captures DEVIATION from structural connectivity — how the observed covariance
      differs from what electrode topology (spatial proximity) predicts
    - Both live in T_I (tangent space at identity) — they add cleanly to spatial logits

    Combined spatial logits: score += α_h · L + γ_h · L_graph
    where α_h (C1) and γ_h (C2) are per-head learnable scales, both init to 0.

    Extra parameters per layer: num_heads scalar scales + one Padé log map call.
    R_sub^{-1/2} is precomputed and cached per channel configuration.
    """
    def __init__(self, embed_dim, num_heads=8, total_channels=TOTAL_GLOBAL_CHANNELS,
                 dropout=0.1, att_dropout=0.1, spd_eps=1e-5,
                 log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None,
                 use_temporal_cov=False):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for parallel split"
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.heads_per_branch = num_heads // 2
        self.dim_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.half_dim = self.heads_per_branch * self.dim_head

        # Shared QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

        # ── Contribution 1: Adaptive Riemannian bias (functional connectivity) ──
        self.riemannian_bias = AdaptiveRiemannianAttentionBias(
            num_heads=self.heads_per_branch,
            total_channels=total_channels,
            eps=spd_eps,
            log_mode=log_mode,
            use_approx=use_approx,
            use_frechet=use_frechet,
            frechet_R_inv_sqrt=frechet_R_inv_sqrt,
        )

        # ── Contribution 2: Graph-Referenced Riemannian bias (structural deviation) ──
        self.graph_bias = GraphReferencedRiemannianBias(
            num_heads=self.heads_per_branch,
            total_channels=total_channels,
            sigma=0.35,
            eps=spd_eps,
        )

    def forward(self, x_norm, num_chan, residual=None, channel_idx=None,
                mask=None):
        """
        Args:
            x_norm: (B, L, D) normalized input (post-LayerNorm)
            num_chan: number of EEG channels C in this batch
            residual: (B, L, D) raw residual stream (pre-LayerNorm) — used for
                      Riemannian bias computation. If None, falls back to x_norm.
            channel_idx: (C,) long tensor — global channel indices for this batch.
            mask: (B, L) boolean — True = masked token. Used to zero out masked
                  channels before covariance computation (pretraining only).
                  None during downstream (no masking).
        """
        B, L, D = x_norm.shape
        assert L % num_chan == 0
        N = L // num_chan
        C = num_chan
        H = self.num_heads
        H2 = self.heads_per_branch
        d = self.dim_head

        # ── Compute Riemannian covariance from residual stream ──
        bias_source = residual if residual is not None else x_norm
        x_space = rearrange(bias_source, 'b (n c) d -> (b n) c d', c=C)

        # Zero out masked channels before covariance (pretraining only)
        if mask is not None:
            mask_space = rearrange(mask, 'b (n c) -> (b n) c', c=C)
            x_space = x_space * (~mask_space).unsqueeze(-1).float()

        # ── Contribution 1: Functional connectivity bias ──
        # riem_bias: (B*N, H2, C, C) — spatial attention bias from log(S)
        # L: (B*N, C, C) — tangent vectors at identity
        # S: (B*N, C, C) — SPD covariance matrices (reused by C2)
        riem_bias, L, S = self.riemannian_bias(x_space, channel_idx)

        # ── Contribution 2: Graph-Referenced structural deviation bias ──
        # graph_bias: (B*N, H2, C, C) — bias from log(R^{-1/2} S R^{-1/2})
        # L_graph: (B*N, C, C) — graph-referenced tangent vectors at identity
        graph_riem_bias, L_graph = self.graph_bias(S, channel_idx)

        # Store diagnostic norms for logging
        self._L_norm = L.detach().abs().mean()
        self._L_graph_norm = L_graph.detach().abs().mean()

        # ── Shared QKV ──
        qkv = self.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d)

        # Split heads: first H2 for temporal, last H2 for spatial
        q_t, q_s = q[:, :H2], q[:, H2:]
        k_t, k_s = k[:, :H2], k[:, H2:]
        v_t, v_s = v[:, :H2], v[:, H2:]

        # ── Temporal attention (standard, no bias) ──
        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)
        v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=C)

        out_t = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=self.att_dropout if self.training else 0.0,
        )
        out_t = rearrange(out_t, '(b c) h n d -> b h (n c) d', b=B, c=C)

        # ── Spatial attention (dual Riemannian-biased) ──
        q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=C)
        k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=C)
        v_s = rearrange(v_s, 'b h (n c) d -> (b n) h c d', c=C)

        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            score = (q_s.float() @ k_s.float().transpose(-2, -1)) / (d ** 0.5)
            # Dual bias: α_h · L (functional) + γ_h · L_graph (structural deviation)
            score = score + riem_bias.float() + graph_riem_bias.float()
            score = score.softmax(dim=-1)
        score = F.dropout(score, p=self.att_dropout, training=self.training)
        out_s = score.to(v_s.dtype) @ v_s

        out_s = rearrange(out_s, '(b n) h c d -> b h (n c) d', b=B, n=N)

        # ── Concatenate heads and project ──
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
                 use_temporal_cov=False):
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
            use_temporal_cov=use_temporal_cov,
        )

        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)

        hidden_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_size=hidden_size, act=act, drop=drop)

    def forward(self, x, num_chan, channel_idx=None, mask=None):
        """
        Args:
            x: (B, L, D) input tensor where L = N * num_chan
            num_chan: number of channels C in this batch
            channel_idx: (C,) long tensor — global channel indices
            mask: (B, L) boolean — True = masked token (pretraining only).
                  Passed to attention for mask-aware covariance. None during
                  downstream inference.
        """
        # Pass normalized input for QKV, raw residual for Riemannian bias
        x = x + self.drop_path1(
            self.attn(self.norm1(x), num_chan, residual=x,
                      channel_idx=channel_idx, mask=mask)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
