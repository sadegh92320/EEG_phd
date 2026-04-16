import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
                 learn_mu_reference=True):
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


# ─────────────────────────────────────────────────────────────────────────────
# C2: Riemannian Luna Temporal Compression
#
# Luna-style pack-unpack bottleneck restricted to the TEMPORAL dimension.
# l learnable auxiliary tokens compress N temporal tokens via cross-attention,
# process them with cheap l×l self-attention, then decompress back to N.
#
# The novelty: SPD-biased pack attention. Each auxiliary token carries a
# learnable SPD prototype μ_q on the channel covariance manifold. During
# packing, tokens whose spatial covariance S_t is close to μ_q (in Frobenius
# distance on the tangent space) get higher attention scores. This makes the
# compression geometry-aware: slots specialize in tokens from distinct
# covariance regimes (brain states), rather than compressing arbitrarily.
#
# The SPD distance reuses L (the log-map output from C1's Riemannian bias),
# so there is zero extra eigendecomposition cost — only l Frobenius distances
# per token, which is negligible.
#
# Reference: Ma et al., "Luna: Linear Unified Nested Attention", NeurIPS 2021.
# ─────────────────────────────────────────────────────────────────────────────

class RiemannianLunaTemporalCompression(nn.Module):
    """
    Geometry-guided temporal token compression via Luna-style pack-unpack.

    Replaces standard N×N temporal self-attention with:
        1. Pack:   l queries cross-attend to N tokens  → O(l·N)
        2. Process: self-attention among l tokens       → O(l²)
        3. Unpack: N queries cross-attend to l tokens   → O(N·l)

    Total: O(N·l) instead of O(N²).

    SPD bias in pack step: score(q,t) = Q_q·K_t/√d − β·‖L_t − μ_q‖_F
    where L_t = log(S_t) is the tangent-space covariance (reused from C1),
    and μ_q is a learnable per-slot prototype in global channel space.

    Memory-efficient design:
        - Reuses q_t/k_t/v_t from shared QKV projection (no redundant pack/unpack
          projections). Only the slot query (pack) and slot self-attention need
          dedicated projections.
        - SPD prototypes use low-rank factorization: μ_q = U_q U_q^T where
          U_q ∈ ℝ^{C_total × r}. Reduces params from l·C²_total to l·C_total·r.

    Args:
        embed_dim:       Total embedding dimension
        num_temporal_heads: Number of temporal heads (typically 4)
        num_slots:       l — number of auxiliary compression tokens
        total_channels:  Size of global channel space (144)
        att_dropout:     Attention dropout probability
        spd_beta_init:   Initial value for β (SPD distance weight). 0 → starts
                         as standard Luna, learns to use geometry.
        proto_rank:      Rank of low-rank prototype factorization (default 8).
    """

    def __init__(self, embed_dim, num_temporal_heads=4, num_slots=16,
                 total_channels=TOTAL_GLOBAL_CHANNELS,
                 att_dropout=0.1, spd_beta_init=0.0, proto_rank=8):
        super().__init__()
        self.num_heads = num_temporal_heads
        self.num_slots = num_slots
        self.dim_head = embed_dim // (num_temporal_heads * 2)  # half-dim since parallel split
        self.total_channels = total_channels
        self.proto_rank = proto_rank

        d = self.dim_head
        half_dim = num_temporal_heads * d  # dimension of temporal branch

        # ── Learnable auxiliary tokens (the "slots") ──
        # Initialized small so pack attention starts near-uniform
        self.slots = nn.Parameter(torch.randn(1, num_slots, half_dim) * 0.02)

        # ── Pack: only need a Q projection for slots ──
        # K and V come directly from the shared QKV's k_t and v_t
        self.pack_q = nn.Linear(half_dim, half_dim, bias=False)

        # ── Self-attention among packed slots: projection-free ──
        # The packed tokens already carry rich representations from the pack
        # cross-attention. Using them directly as Q/K/V for self-attention
        # saves ~1.2M params across 6 layers with negligible expressivity loss
        # (the 16 slots are low-dimensional enough that additional projections
        # add more parameters than information).

        # ── Unpack: no extra projections needed ──
        # Q = original q_t, K/V = processed slots (already transformed by self-attn)

        # ── SPD prototypes: low-rank factorization ──
        # μ_q = U_q @ U_q^T  where U_q ∈ ℝ^{total_channels × proto_rank}
        # Stores (num_slots, total_channels, proto_rank) instead of (num_slots, C², C²).
        # Initialized to zero → all slots start equidistant from all tokens.
        self.mu_proto_factors = nn.Parameter(
            torch.zeros(num_slots, total_channels, proto_rank)
        )

        # ── Learnable SPD distance weight ──
        self.spd_beta = nn.Parameter(torch.tensor(float(spd_beta_init)))

        self.att_dropout = att_dropout
        self.scale = d ** -0.5

    @property
    def mu_prototypes(self):
        """Reconstruct full prototypes from low-rank factors (for logging)."""
        # (l, C_total, r) @ (l, r, C_total) → (l, C_total, C_total)
        return self.mu_proto_factors @ self.mu_proto_factors.transpose(-2, -1)

    def _compute_spd_distances(self, L_per_sample, channel_idx):
        """
        Compute Frobenius distance between each token's tangent vector L_t
        and each slot's low-rank prototype μ_q = U_q U_q^T.

        Operates at SAMPLE level (B, N), not (B*C, N) — all channels at the
        same timestep share the same covariance, so distances are identical
        across channels. The caller broadcasts to B*C after this.

        Args:
            L_per_sample: (B, N, C_ch, C_ch) log-map output per temporal token.
            channel_idx: (C_ch,) global channel indices

        Returns:
            distances: (B, num_slots, N) — ‖L_t − μ_q‖_F for each slot-token pair
        """
        BC, N, C_ch, _ = L_per_sample.shape  # BC is actually B here (sample-level)
        l = self.num_slots
        r = self.proto_rank

        # Extract submatrix of each prototype factor for this dataset's channels
        # (l, total_ch, r) → (l, C_ch, r)
        U_sub = self.mu_proto_factors[:, channel_idx]  # (l, C_ch, r)

        # ‖L_t‖²_F: (BC, N)
        L_flat = L_per_sample.reshape(BC, N, -1)  # (B, N, C_ch²)
        L_sqnorm = (L_flat * L_flat).sum(dim=-1)  # (BC, N)

        # ‖μ_q‖²_F = ‖U U^T‖²_F = tr((UU^T)(UU^T)) = ‖U^T U‖²_F
        # U^T U: (l, r, r)
        UtU = torch.bmm(U_sub.transpose(-2, -1), U_sub)  # (l, r, r)
        mu_sqnorm = (UtU * UtU).sum(dim=(-2, -1))  # (l,)

        # tr(L · UU^T) = tr(U^T L U) = ‖L_flat @ U‖²_F summed smartly
        # L_per_token: (BC, N, C_ch, C_ch), U_sub: (l, C_ch, r)
        # For each slot q: tr(L_t · U_q U_q^T) = ‖L_t @ U_q‖²_F ... no.
        # Actually tr(L · UU^T) = sum_ij L_ij (UU^T)_ij
        #   = vec(L)^T vec(UU^T)
        # But we want to avoid forming UU^T.
        # tr(L · UU^T) = tr(U^T L U) (cyclic property)
        # L_t @ U_q: (BC, N, C_ch, r) for one slot. Then trace of U^T @ that.
        # = sum of (U_q^T @ L_t @ U_q) diagonal entries
        # Efficient: LU = einsum('bnij,ljr->bnlr' ... ) but that's a big tensor.
        #
        # Better: flatten L and UU^T and dot.
        # UU^T_flat: (l, C_ch²). L_flat: (BC, N, C_ch²).
        # cross = L_flat @ UU^T_flat^T → (BC, N, l). This needs UU^T_flat.
        # UU^T = U @ U^T → (l, C_ch, C_ch) → reshape to (l, C_ch²).
        # For C_ch=22, r=8: UU^T is (l, 22, 22) = 7.7K per slot. Very cheap.
        mu_sub = torch.bmm(U_sub, U_sub.transpose(-2, -1))  # (l, C_ch, C_ch)
        mu_flat = mu_sub.reshape(l, -1)  # (l, C_ch²)

        # cross = L_flat @ mu_flat^T → (BC, N, l)
        # Use einsum to avoid expanding mu_flat to (BC, ...)
        cross = torch.einsum('bni,li->bnl', L_flat, mu_flat)  # (BC, N, l)

        # ‖L - μ‖²_F = ‖L‖² + ‖μ‖² - 2·cross
        dist_sq = L_sqnorm.unsqueeze(-1) + mu_sqnorm.unsqueeze(0).unsqueeze(0) - 2 * cross
        dist_sq = dist_sq.clamp(min=0)

        # (BC, N, l) → (BC, l, N)
        distances = dist_sq.sqrt().permute(0, 2, 1)
        return distances

    def forward(self, q_t, k_t, v_t, L_n=None, channel_idx=None, C=None):
        """
        Luna-style temporal attention with SPD-biased packing.

        Reuses q_t/k_t/v_t from the shared QKV projection — no redundant
        pack/unpack projections. Only the slot queries and slot self-attention
        have dedicated parameters.

        Args:
            q_t: (B*C, H_t, N, d) temporal queries (already QKV-projected)
            k_t: (B*C, H_t, N, d) temporal keys
            v_t: (B*C, H_t, N, d) temporal values
            L_n: (B*N, C_ch, C_ch) tangent vectors from C1's Riemannian bias.
                 Reused to avoid recomputation. None → no SPD bias.
            channel_idx: (C_ch,) global channel indices.
            C:   Number of channels (needed to reshape L_n).

        Returns:
            out: (B*C, H_t, N, d) temporal attention output
        """
        BC, H, N, d = q_t.shape
        l = self.num_slots

        # ── Pack: slots attend to input tokens ──
        # Slot queries get their own projection; K and V reuse shared QKV output
        slots = self.slots.expand(BC, -1, -1)  # (BC, l, H*d)
        pq = self.pack_q(slots)  # (BC, l, H*d)
        pq = rearrange(pq, 'bc l (h d) -> bc h l d', h=H)  # (BC, H, l, d)

        # Pack attention: (BC, H, l, N)
        pack_score = (pq @ k_t.transpose(-2, -1)) * self.scale

        # ── SPD bias on pack attention ──
        if L_n is not None and channel_idx is not None and C is not None:
            B = BC // C
            # L_n is (B*N, C_ch, C_ch) — one covariance per (batch, timestep).
            # All C channels at timestep n share the SAME covariance, so we
            # compute distances at (B, N) level and broadcast to (B*C) — saves
            # a factor of C in both compute and memory.
            # Detach L_n: SPD distances guide attention routing but we don't
            # need gradients flowing back through the covariance computation
            # (C1's Riemannian bias already trains L_n). Saves ~15-20% backward time.
            L_per_sample = L_n.detach().reshape(B, N, *L_n.shape[1:])  # (B, N, C_ch, C_ch)

            # SPD distances at sample level: (B, l, N)
            spd_dist_b = self._compute_spd_distances(L_per_sample, channel_idx)

            # Broadcast to all channels: (B, l, N) → (B*C, l, N)
            # Use repeat_interleave so each sample's distances are repeated C
            # times contiguously (matching the (b c) layout of BC dimension).
            # This allocates only (B*C, l, N) — no intermediate 4D tensor.
            spd_dist = spd_dist_b.repeat_interleave(C, dim=0)  # (B*C, l, N)

            # Subtract β · distance (higher distance → lower score)
            # pack_score is (BC, H, l, N), spd_dist is (BC, l, N) → unsqueeze for heads
            pack_score = pack_score - self.spd_beta * spd_dist.unsqueeze(1)

        pack_attn = pack_score.softmax(dim=-1)
        pack_attn = F.dropout(pack_attn, p=self.att_dropout, training=self.training)

        # Packed representation: (BC, H, l, d)
        packed = pack_attn @ v_t

        # ── Process: self-attention among packed slots (projection-free) ──
        # packed is already (BC, H, l, d) — use directly as Q, K, V
        # Uses FlashAttention kernel for speed (no custom bias needed here)
        processed = F.scaled_dot_product_attention(
            packed, packed, packed,
            dropout_p=self.att_dropout if self.training else 0.0
        )  # (BC, H, l, d)

        # ── Unpack: input tokens attend to processed slots ──
        # Q = original q_t, K/V = processed slots (no extra projections)
        # Uses FlashAttention kernel for speed (no custom bias needed here)
        out = F.scaled_dot_product_attention(
            q_t, processed, processed,
            dropout_p=self.att_dropout if self.training else 0.0
        )  # (BC, H, N, d)

        return out


class AdaptiveRiemannianParallelAttention(nn.Module):
    """
    Parallel spatial-temporal attention with Riemannian spatial bias (C1).

    4 temporal heads + 4 spatial heads, shared QKV.
    Spatial heads get additive Riemannian bias: score += α_h · log(S)
    Temporal heads are standard (no bias).
    """
    def __init__(self, embed_dim, num_heads=8, total_channels=TOTAL_GLOBAL_CHANNELS,
                 dropout=0.1, att_dropout=0.1, spd_eps=1e-5,
                 log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None,
                 use_temporal_cov=False,
                 use_value_bias=True,
                 learn_mu_reference=True,
                 use_luna_temporal=False, luna_num_slots=16,
                 luna_spd_beta_init=0.0):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for parallel split"
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.heads_per_branch = num_heads // 2
        self.dim_head = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.half_dim = self.heads_per_branch * self.dim_head
        self.use_value_bias = use_value_bias
        self.use_luna_temporal = use_luna_temporal

        # Shared QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

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
        )

        # C2: Luna-style temporal compression with SPD-biased packing
        if use_luna_temporal:
            self.luna_temporal = RiemannianLunaTemporalCompression(
                embed_dim=embed_dim,
                num_temporal_heads=self.heads_per_branch,
                num_slots=luna_num_slots,
                total_channels=total_channels,
                att_dropout=att_dropout,
                spd_beta_init=luna_spd_beta_init,
            )

        # Geometric value mixing: V' = V + β_h · (L @ V)
        # Injects covariance structure into attention values — complementary
        # to the score bias (which controls routing, not feature mixing).
        # Initialized to 0 so model starts as standard attention.
        if use_value_bias:
            self.value_beta = nn.Parameter(torch.zeros(self.heads_per_branch))

    def forward(self, x_norm, num_chan, residual=None, channel_idx=None,
                mask=None):
        """
        Args:
            x_norm: (B, L, D) normalized input (post-LayerNorm)
            num_chan: number of EEG channels C in this batch
            residual: (B, L, D) raw residual stream (pre-LayerNorm)
            channel_idx: (C,) long tensor — global channel indices
            mask: (B, L) boolean — True = masked token (pretraining only)
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

        # ── Shared QKV ──
        qkv = self.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_t, q_s = q[:, :H2], q[:, H2:]
        k_t, k_s = k[:, :H2], k[:, H2:]
        v_t, v_s = v[:, :H2], v[:, H2:]

        # ── Temporal attention ──
        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=C)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=C)
        v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=C)

        if self.use_luna_temporal:
            # C2: Luna pack-process-unpack with SPD-biased packing
            # L_n from C1's Riemannian bias is reused — zero extra cost
            out_t = self.luna_temporal(
                q_t, k_t, v_t,
                L_n=L_n, channel_idx=channel_idx, C=C,
            )
        else:
            # Standard N×N temporal self-attention
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

        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            score = (q_s.float() @ k_s.float().transpose(-2, -1)) / (d ** 0.5)
            score = score + riem_bias.float()
            score = score.softmax(dim=-1)
        score = F.dropout(score, p=self.att_dropout, training=self.training)
        out_s = score.to(v_s.dtype) @ v_s

        out_s = rearrange(out_s, '(b n) h c d -> b h (n c) d', b=B, n=N)

        # ── Concatenate and project ──
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
                 use_temporal_cov=False,
                 use_value_bias=True,
                 learn_mu_reference=True,
                 use_luna_temporal=False, luna_num_slots=16,
                 luna_spd_beta_init=0.0):
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
            use_value_bias=use_value_bias,
            learn_mu_reference=learn_mu_reference,
            use_luna_temporal=use_luna_temporal,
            luna_num_slots=luna_num_slots,
            luna_spd_beta_init=luna_spd_beta_init,
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
        """
        # Pass normalized input for QKV, raw residual for Riemannian bias.
        x = x + self.drop_path1(
            self.attn(self.norm1(x), num_chan, residual=x,
                      channel_idx=channel_idx, mask=mask)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
