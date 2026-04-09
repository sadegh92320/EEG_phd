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

    Optional Fréchet mean pre-whitening (use_frechet=True):
        Instead of projecting at the identity, project at the offline
        Fréchet mean R of the training covariance distribution:
            S̃ = R^{-1/2} S R^{-1/2}     (whiten — brings S̃ close to I)
            log(S̃) ≈ Padé(S̃)             (now accurate because S̃ ≈ I)

        This is geometrically principled: R is the intrinsic center of
        the data on the SPD manifold, so whitening minimizes the distance
        ||S̃ - I|| across the training set, maximizing Padé accuracy.

        The frozen R^{-1/2} is loaded from a precomputed file and stored
        as a buffer (no gradients, moves with .to(device) automatically).

    Args:
        total_channels: Size of the global channel space (default 144)
        eps: Regularization constant for numerical stability
        log_mode: 'approx', 'pade', or 'eigh'
        use_approx: DEPRECATED — kept for backward compat.
        use_frechet: If True, pre-whiten S with frozen Fréchet mean
        frechet_R_inv_sqrt: (C, C) tensor — precomputed R^{-1/2}.
                            Required if use_frechet=True. Register as buffer.
    """
    def __init__(self, total_channels=TOTAL_GLOBAL_CHANNELS, eps=1e-5,
                 log_mode='eigh', use_approx=False,
                 use_frechet=False, frechet_R_inv_sqrt=None):
        super().__init__()
        self.eps = eps
        self.total_channels = total_channels
        self.use_frechet = use_frechet

        # Backward compat: use_approx=True overrides log_mode
        if use_approx:
            self.log_mode = 'approx'
        else:
            self.log_mode = log_mode

        # Frozen Fréchet mean reference (no gradients, moves with device)
        if use_frechet and frechet_R_inv_sqrt is not None:
            self.register_buffer('R_inv_sqrt', frechet_R_inv_sqrt)
        else:
            self.R_inv_sqrt = None

    def forward(self, S, channel_idx=None, ema_ref=None):
        """
        Args:
            S: (batch, C, C) batch of SPD matrices
            channel_idx: (C,) global channel indices — used to extract
                         the correct submatrix of R^{-1/2} when use_frechet=True
            ema_ref: unused, kept for API compatibility
        Returns:
            (batch, C, C) tangent vectors
        """
        orig_dtype = S.dtype
        C = S.shape[-1]
        I = torch.eye(C, device=S.device, dtype=S.dtype).unsqueeze(0)

        # ── Optional Fréchet pre-whitening ──
        # S̃ = R^{-1/2} S R^{-1/2}  (centered at Fréchet mean, close to I)
        S_before = S  # keep original in case whitening fails
        if self.use_frechet and self.R_inv_sqrt is not None:
            # Extract the C×C submatrix for this batch's channels
            if channel_idx is not None and self.R_inv_sqrt.shape[0] > C:
                R_sub = self.R_inv_sqrt[channel_idx][:, channel_idx]  # (C, C)
            else:
                R_sub = self.R_inv_sqrt[:C, :C]  # fallback: take first C
            R_sub = R_sub.to(S.dtype).unsqueeze(0)  # (1, C, C)
            S = R_sub @ S @ R_sub.transpose(-1, -2)
            # Re-symmetrize (numerical safety after two matmuls)
            S = 0.5 * (S + S.transpose(-1, -2))

            # ── NaN/Inf guard after whitening ──
            if torch.isnan(S).any() or torch.isinf(S).any():
                print(f"[AdaptiveLogMap] NaN/Inf AFTER Fréchet whitening!")
                print(f"  S_whitened range: [{S[~torch.isnan(S)].min().item():.4f}, "
                      f"{S[~torch.isnan(S)].max().item():.4f}]")
                print(f"  R_sub range: [{R_sub.min().item():.4f}, {R_sub.max().item():.4f}]")
                print(f"  S_original diag: [{S_before.diagonal(dim1=-2,dim2=-1).min().item():.4f}, "
                      f"{S_before.diagonal(dim1=-2,dim2=-1).max().item():.4f}]")
                # Fall back to un-whitened S to avoid crash
                S = S_before

        # ── Tangent-space projection ──
        if self.log_mode == 'approx':
            # First-order: S - I
            return (S - I).to(orig_dtype)
        else:
            # Padé [1,1] approximant of matrix logarithm:
            # log(S) ≈ 2(S - I)(I + S)^{-1}
            X = S - I
            T = torch.linalg.solve(I + S, 2 * X)

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
        use_frechet: If True, pre-whiten S with frozen Fréchet mean
        frechet_R_inv_sqrt: (C, C) tensor — precomputed R^{-1/2}
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
            use_frechet=use_frechet,
            frechet_R_inv_sqrt=frechet_R_inv_sqrt,
        )

        # Per-head learnable scaling — initialized to 0 so the model starts
        # as standard Euclidean attention and learns to use the bias
        self.head_scales = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x, channel_idx, return_covariance=False, ema_ref=None):
        """
        Args:
            x: (B*N, C, D) channel embeddings (from residual stream)
            channel_idx: (C,) long tensor — global channel indices for this batch
            return_covariance: if True, also return the raw covariance S
            ema_ref: optional (C, C) population covariance from EMAGeometricGraph
        Returns:
            bias: (B*N, num_heads, C, C) attention bias per head
            S (optional): (B*N, C, C) raw covariance (only if return_covariance=True)
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

        # Step 2: Project to tangent space at reference (EMA population or identity)
        L = self.adaptive_log(S, channel_idx, ema_ref=ema_ref)  # (B*N, C, C)

        # Step 3: Per-head scaling
        scales = self.head_scales.view(1, self.num_heads, 1, 1)
        bias = L.unsqueeze(1) * scales

        if return_covariance:
            return bias, S
        return bias


class TemporalCovarianceAttentionBias(nn.Module):
    """
    Contribution 2: Temporal dynamics of spatial covariance.

    Captures HOW the brain's spatial structure evolves over time by computing
    per-timestep channel covariance matrices and their pairwise Frobenius
    distances. This (N, N) distance matrix is a data-dependent temporal
    signal: "timesteps t and s have similar/different spatial configurations."

    Implementation: additive bias on temporal attention logits (same pattern
    as the Riemannian spatial bias), with learned per-head scales initialized
    near zero for safe warm-up.

    Unified geometric framework:
        - Spatial heads: Riemannian bias from per-sample channel covariance
        - Temporal heads: bias from dynamics of that same covariance over time
    """
    def __init__(self, num_heads):
        super().__init__()
        # Learned per-head scale, initialized near zero so at start
        # of training the model behaves identically to baseline
        self.head_scales = nn.Parameter(torch.full((num_heads,), 0.01))

    @staticmethod
    @torch.no_grad()
    def compute_temporal_cov_dist(x_patches, num_chan):
        """
        Compute pairwise Frobenius distance between per-timestep covariances.

        Args:
            x_patches: (B, N, C, D) patch embeddings BEFORE masking
            num_chan: C

        Returns:
            dist: (B, N, N) pairwise Frobenius distance matrix (float32)
        """
        # All in float32 to avoid fp16 overflow in covariance computation
        x = x_patches.float()  # (B, N, C, D)
        B, N, C, D = x.shape

        # Per-timestep covariance: S_t = X_t X_t^T / D
        cov = torch.matmul(x, x.transpose(-1, -2)) / D  # (B, N, C, C)

        # Pairwise Frobenius distance: ||S_t - S_s||_F
        # Efficient: ||A-B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2*tr(A^T B)
        cov_flat = cov.reshape(B, N, C * C)  # (B, N, C*C)
        norms_sq = (cov_flat ** 2).sum(dim=-1)  # (B, N)
        cross = torch.bmm(cov_flat, cov_flat.transpose(-1, -2))  # (B, N, N)
        dist_sq = norms_sq.unsqueeze(-1) + norms_sq.unsqueeze(-2) - 2 * cross
        dist_sq = dist_sq.clamp(min=0)  # numerical safety
        dist = torch.sqrt(dist_sq + 1e-8)  # (B, N, N)

        return dist

    def forward(self, temporal_cov_dist):
        """
        Compute per-head additive bias from temporal covariance distance.

        Args:
            temporal_cov_dist: (B, N, N) pairwise Frobenius distance

        Returns:
            bias: (B, H, N, N) additive bias for temporal attention logits
        """
        # dist: (B, N, N) -> (B, 1, N, N) * (1, H, 1, 1) -> (B, H, N, N)
        scales = self.head_scales.view(1, -1, 1, 1)
        # Negate: closer covariance = higher attention, farther = lower
        bias = -temporal_cov_dist.unsqueeze(1) * scales
        return bias


class AdaptiveRiemannianParallelAttention(nn.Module):
    """
    Parallel spatial-temporal attention with adaptive Riemannian bias.

    Supports variable channel counts across batches. The Riemannian reference
    lives in the global 144-channel space; each batch's channel indices select
    the relevant submatrix.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads (must be even — split 50/50)
        total_channels: Size of global channel space (default 144)
        dropout: Output dropout probability
        att_dropout: Attention weight dropout probability
        spd_eps: SPD regularization constant
        log_mode: 'approx', 'pade', or 'eigh' (see AdaptiveLogMap)
        use_approx: DEPRECATED — kept for backward compat.
        use_frechet: If True, pre-whiten S with frozen Fréchet mean
        frechet_R_inv_sqrt: (C, C) tensor — precomputed R^{-1/2}
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
        self.use_temporal_cov = use_temporal_cov

        # Shared QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = att_dropout

        # Adaptive Riemannian bias for spatial heads (global channel space)
        self.riemannian_bias = AdaptiveRiemannianAttentionBias(
            num_heads=self.heads_per_branch,
            total_channels=total_channels,
            eps=spd_eps,
            log_mode=log_mode,
            use_approx=use_approx,
            use_frechet=use_frechet,
            frechet_R_inv_sqrt=frechet_R_inv_sqrt,
        )

        # Temporal covariance bias for temporal heads (Contribution 2)
        if use_temporal_cov:
            self.temporal_cov_bias = TemporalCovarianceAttentionBias(
                num_heads=self.heads_per_branch,
            )

    def forward(self, x_norm, num_chan, residual=None, channel_idx=None,
                ema_ref=None, temporal_cov_dist=None):
        """
        Args:
            x_norm: (B, L, D) normalized input (post-LayerNorm) — used for QKV
            num_chan: number of EEG channels C in this batch
            residual: (B, L, D) raw residual stream (pre-LayerNorm) — used for
                      Riemannian bias. If None, falls back to x_norm.
            channel_idx: (C,) long tensor — global channel indices for this batch.
                         Required for the adaptive reference submatrix extraction.
            ema_ref: optional (C, C) population covariance from EMAGeometricGraph.
                     Used as the tangent space reference point in approx mode.
            temporal_cov_dist: optional (B, N, N) pairwise Frobenius distance
                     between per-timestep covariance matrices. When provided and
                     use_temporal_cov=True, added as bias to temporal attention.
        """
        B, L, D = x_norm.shape
        assert L % num_chan == 0
        N = L // num_chan
        H = self.num_heads
        H2 = self.heads_per_branch
        d = self.dim_head

        # Shared QKV from normalized input
        qkv = self.qkv(x_norm).reshape(B, L, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Split heads: first H2 for temporal, last H2 for spatial
        q_t, q_s = q[:, :H2], q[:, H2:]
        k_t, k_s = k[:, :H2], k[:, H2:]
        v_t, v_s = v[:, :H2], v[:, H2:]

        # ── Temporal attention ──
        q_t = rearrange(q_t, 'b h (n c) d -> (b c) h n d', c=num_chan)
        k_t = rearrange(k_t, 'b h (n c) d -> (b c) h n d', c=num_chan)
        v_t = rearrange(v_t, 'b h (n c) d -> (b c) h n d', c=num_chan)

        if self.use_temporal_cov and temporal_cov_dist is not None:
            # Manual temporal attention with covariance bias (same pattern as spatial)
            # Compute per-head additive bias: (B, H2, N, N)
            tcov_bias = self.temporal_cov_bias(temporal_cov_dist)  # (B, H2, N, N)
            # Expand across channels: (B, H2, N, N) -> (B*C, H2, N, N)
            tcov_bias = tcov_bias.unsqueeze(1).expand(-1, num_chan, -1, -1, -1)
            tcov_bias = tcov_bias.reshape(B * num_chan, H2, N, N)

            # Manual attention in float32 (same as spatial branch)
            with torch.amp.autocast('cuda', enabled=False), \
                 torch.amp.autocast('cpu', enabled=False):
                score_t = (q_t.float() @ k_t.float().transpose(-2, -1)) / (d ** 0.5)
                score_t = score_t + tcov_bias.float()
                score_t = score_t.softmax(dim=-1)
            score_t = F.dropout(score_t, p=self.att_dropout, training=self.training)
            out_t = score_t.to(v_t.dtype) @ v_t
        else:
            # Standard flash SDPA (no bias needed)
            out_t = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=self.att_dropout if self.training else 0.0,
            )
        out_t = rearrange(out_t, '(b c) h n d -> b h (n c) d', b=B, c=num_chan)

        # ── Spatial attention (Riemannian-biased) ──
        q_s = rearrange(q_s, 'b h (n c) d -> (b n) h c d', c=num_chan)
        k_s = rearrange(k_s, 'b h (n c) d -> (b n) h c d', c=num_chan)
        v_s = rearrange(v_s, 'b h (n c) d -> (b n) h c d', c=num_chan)

        # Compute Riemannian bias from RESIDUAL STREAM (pre-LayerNorm)
        # ema_ref changes the tangent space reference from identity to population average
        bias_source = residual if residual is not None else x_norm
        x_space = rearrange(bias_source, 'b (n c) d -> (b n) c d', c=num_chan)
        riem_bias = self.riemannian_bias(x_space, channel_idx, ema_ref=ema_ref)  # (B*N, H2, C, C)

        # Manual spatial attention with Riemannian bias
        # Compute score+softmax in float32 to prevent fp16 overflow
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            score = (q_s.float() @ k_s.float().transpose(-2, -1)) / (d ** 0.5)
            score = score + riem_bias.float()
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

    def forward(self, x, num_chan, channel_idx=None, ema_ref=None,
                temporal_cov_dist=None):
        """
        Args:
            x: (B, L, D) input tensor where L = N * num_chan
            num_chan: number of channels C in this batch
            channel_idx: (C,) long tensor — global channel indices
            ema_ref: optional (C, C) population covariance from EMAGeometricGraph.
                     Used as the tangent space reference point in approx mode.
                     If None, falls back to identity (S - I).
            temporal_cov_dist: optional (B, N, N) temporal covariance distance.
                     Passed to temporal attention heads as additive bias.
        """
        # Pass normalized input for QKV, raw residual for Riemannian bias
        x = x + self.drop_path1(
            self.attn(self.norm1(x), num_chan, residual=x,
                      channel_idx=channel_idx, ema_ref=ema_ref,
                      temporal_cov_dist=temporal_cov_dist)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ════════════════════════════════════════════════════════════════
# EMA Geometric Graph — population-level channel connectivity
# ════════════════════════════════════════════════════════════════

class EMAGeometricGraph(nn.Module):
    """
    Maintains a running EMA of channel-channel covariance across the entire
    training corpus. Lives in the global 144-channel space — each batch updates
    the submatrix corresponding to its channel set.

    The EMA graph captures population-level spatial structure that individual
    per-sample covariances cannot: averaged over thousands of segments, subjects,
    and datasets, it converges to the true population covariance pattern.

    Used as the REFERENCE POINT for the adaptive log map tangent space projection:
        tangent = S_sample - G_ema_sub  (instead of S_sample - I)

    This means the Riemannian bias measures per-sample DEVIATION from population
    average, not absolute covariance. More informative: identity treats all
    channels as uncorrelated, but EEG channels are always correlated.

    Args:
        total_channels: Size of global channel space (default 144)
        momentum: EMA decay factor (higher = slower update, more stable)
        num_heads: DEPRECATED — kept for backward compat, no longer used.
    """
    def __init__(self, total_channels=TOTAL_GLOBAL_CHANNELS, momentum=0.99,
                 num_heads=4):
        super().__init__()
        self.total_channels = total_channels
        self.momentum = momentum

        # Running EMA of channel covariance — initialized to identity
        # (uninformative prior: all channels equally connected)
        # At init, get_ref returns identity → S - I (same as before EMA warms up)
        self.register_buffer(
            'G_ema', torch.eye(total_channels, dtype=torch.float32)
        )
        # Track how many updates each channel pair has seen
        self.register_buffer(
            'update_count', torch.zeros(total_channels, dtype=torch.long)
        )

    @torch.no_grad()
    def update(self, S_batch, channel_idx):
        """
        Update the EMA graph with a batch of covariance matrices.

        Args:
            S_batch: (B*N, C, C) or (B, C, C) per-sample covariance matrices
            channel_idx: (C,) long tensor — global channel indices for this batch
        """
        if not self.training:
            return

        # Average across batch dimension → single (C, C) population estimate
        S_mean = S_batch.float().mean(dim=0)  # (C, C)

        # Update the relevant submatrix of G_ema
        idx = channel_idx.long()
        m = self.momentum
        sub = self.G_ema[idx.unsqueeze(1), idx.unsqueeze(0)]  # (C, C)
        self.G_ema[idx.unsqueeze(1), idx.unsqueeze(0)] = m * sub + (1 - m) * S_mean
        self.update_count[idx] += 1

    def get_ref(self, channel_idx):
        """
        Extract the EMA submatrix for the current channel set as the tangent
        space reference point.

        Returns the raw (C, C) population covariance — no scaling, no reshaping.
        The per-sample Riemannian bias's own head_scales handle magnitude.

        At init (before any updates), G_ema = I → tangent = S - I (standard).
        As training progresses, G_ema converges to population average →
        tangent = S - G_pop (measures per-sample deviation).

        Args:
            channel_idx: (C,) long tensor — global channel indices
        Returns:
            (C, C) population covariance submatrix
        """
        idx = channel_idx.long()
        return self.G_ema[idx.unsqueeze(1), idx.unsqueeze(0)]  # (C, C)