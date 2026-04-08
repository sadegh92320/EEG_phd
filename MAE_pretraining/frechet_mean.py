"""
Fréchet Mean Computation for SPD Matrices
==========================================

Computes the Riemannian Fréchet mean of a collection of SPD matrices.
The Fréchet mean R* minimizes:

    R* = argmin_R  Σ_i  ||log(R^{-1/2} S_i R^{-1/2})||²_F

This is the intrinsic "center of mass" on the SPD manifold.

Usage:
    python -m MAE_pretraining.frechet_mean \
        --config path/to/config.yaml \
        --output frechet_mean.pt \
        --max_batches 200

The output file contains:
    - 'R':          (C, C) Fréchet mean matrix
    - 'R_inv_sqrt': (C, C) precomputed R^{-1/2} for online whitening
    - 'R_sqrt':     (C, C) precomputed R^{1/2} (inverse of above)
    - 'channel_count': C (number of channels in this dataset group)
"""

import torch
import numpy as np
from scipy.linalg import sqrtm, inv
import lightning.pytorch as pl


def _matrix_sqrt_inv_np(M):
    """Compute M^{-1/2} via eigendecomposition (numpy, CPU)."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical safety
    M_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return M_inv_sqrt


def _matrix_sqrt_np(M):
    """Compute M^{1/2} via eigendecomposition (numpy, CPU)."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    M_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return M_sqrt


def compute_frechet_mean(covariances, max_iters=100, tol=1e-8, verbose=True,
                         warm_start=None):
    """
    Compute the Riemannian Fréchet mean of SPD matrices via iterative algorithm.

    Uses the Karcher flow / gradient descent on the SPD manifold:
        1. Initialize R = warm_start if provided, else arithmetic mean
        2. Repeat:
           a. Compute tangent vectors: T_i = log(R^{-1/2} S_i R^{-1/2})
           b. Average tangent vector: T_avg = mean(T_i)
           c. Update: R = R^{1/2} @ exp(T_avg) @ R^{1/2}
           d. Converge when ||T_avg||_F < tol

    Args:
        covariances: list of (C, C) numpy arrays — SPD matrices
        max_iters:   maximum Karcher flow iterations
        tol:         convergence tolerance on ||T_avg||_F
        verbose:     print progress
        warm_start:  (C, C) numpy array — previous Fréchet mean to start from.
                     When provided, Karcher typically converges in 2-5 iterations
                     instead of 10-30, since the encoder changes slowly per epoch.

    Returns:
        R: (C, C) numpy array — Fréchet mean
    """
    N = len(covariances)
    C = covariances[0].shape[0]

    # Stack into (N, C, C) array for vectorized operations
    covs = np.stack(covariances, axis=0)  # (N, C, C)

    # Subsample if too many covariances (>5000 is overkill for C×C estimation)
    MAX_COVS = 5000
    if N > MAX_COVS:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, MAX_COVS, replace=False)
        covs = covs[idx]
        N = MAX_COVS
        if verbose:
            print(f"  Subsampled to {N} covariances (more than enough for {C}×{C})")

    if warm_start is not None and warm_start.shape == (C, C):
        R = warm_start.copy()
        if verbose:
            print(f"  Warm-starting Karcher from previous Fréchet mean")
    else:
        # Initialize with arithmetic mean (reasonable for SPD near each other)
        R = np.mean(covs, axis=0)

    # Ensure R is SPD
    R = 0.5 * (R + R.T)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-8:
        R += (1e-8 - eigvals.min()) * np.eye(C)

    for it in range(max_iters):
        R_inv_sqrt = _matrix_sqrt_inv_np(R)
        R_sqrt = _matrix_sqrt_np(R)

        # Vectorized: whiten all covariances at once
        # M_all = R^{-1/2} @ S_i @ R^{-1/2} for all i
        M_all = R_inv_sqrt @ covs @ R_inv_sqrt  # (N, C, C) broadcast
        M_all = 0.5 * (M_all + M_all.transpose(0, 2, 1))  # symmetrize

        # Batched eigendecomposition
        eigvals_all, eigvecs_all = np.linalg.eigh(M_all)  # (N, C), (N, C, C)
        eigvals_all = np.maximum(eigvals_all, 1e-10)
        log_eigvals = np.log(eigvals_all)  # (N, C)

        # Reconstruct log matrices: V @ diag(log(λ)) @ V^T for all i
        # log_M_all[i] = eigvecs_all[i] @ diag(log_eigvals[i]) @ eigvecs_all[i].T
        log_M_all = eigvecs_all * log_eigvals[:, np.newaxis, :]  # (N, C, C) broadcast
        log_M_all = log_M_all @ eigvecs_all.transpose(0, 2, 1)  # (N, C, C)

        T_avg = log_M_all.mean(axis=0)  # (C, C)

        # Check convergence
        step_size = np.linalg.norm(T_avg, 'fro')
        if verbose and (it % 10 == 0 or step_size < tol):
            print(f"  Karcher iter {it:3d}: ||T_avg||_F = {step_size:.6e}")

        if step_size < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break

        # Early stopping: if diverging, revert to best iterate
        if it == 0:
            best_step = step_size
            best_R = R.copy()
        elif step_size < best_step:
            best_step = step_size
            best_R = R.copy()
        elif step_size > best_step * 2.0:
            if verbose:
                print(f"  Diverging at iteration {it} (step={step_size:.4e} > "
                      f"2×best={best_step:.4e}). Reverting to best iterate.")
            return best_R

        # Exponential map: update R = R^{1/2} @ exp(T_avg) @ R^{1/2}
        eigvals_t, eigvecs_t = np.linalg.eigh(T_avg)
        exp_T = eigvecs_t @ np.diag(np.exp(eigvals_t)) @ eigvecs_t.T
        R = R_sqrt @ exp_T @ R_sqrt

        # Symmetrize
        R = 0.5 * (R + R.T)

    return R


def compute_frechet_mean_from_dataloader(dataloader, eps=1e-5, max_batches=200,
                                          max_iters=100, verbose=True):
    """
    Compute Fréchet mean from a pretraining dataloader.

    Collects sample covariances from raw EEG batches, then computes the mean.

    Args:
        dataloader:  yields (eeg, channel_list) where eeg is (B, C, T)
        eps:         regularization added to covariance
        max_batches: maximum number of batches to use (for speed)
        max_iters:   Karcher flow iterations
        verbose:     print progress

    Returns:
        dict with keys: 'R', 'R_inv_sqrt', 'R_sqrt', 'channel_count'
    """
    if verbose:
        print("Collecting sample covariances from dataloader...")

    all_covs = []
    C_seen = None

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        eeg, channel_list = batch
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.numpy()

        B, C, T = eeg.shape

        if C_seen is None:
            C_seen = C
        elif C != C_seen:
            # Skip batches with different channel counts
            continue

        for i in range(B):
            x = eeg[i]  # (C, T)
            x = x - x.mean(axis=-1, keepdims=True)
            cov = (x @ x.T) / T + eps * np.eye(C)
            all_covs.append(cov)

        if verbose and batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: collected {len(all_covs)} covariances "
                  f"(C={C_seen})")

    if verbose:
        print(f"Total: {len(all_covs)} covariances, C={C_seen}")
        print(f"\nComputing Fréchet mean ({max_iters} max iterations)...")

    R = compute_frechet_mean(all_covs, max_iters=max_iters, verbose=verbose)
    R_inv_sqrt = _matrix_sqrt_inv_np(R)
    R_sqrt = _matrix_sqrt_np(R)

    result = {
        'R': torch.from_numpy(R).float(),
        'R_inv_sqrt': torch.from_numpy(R_inv_sqrt).float(),
        'R_sqrt': torch.from_numpy(R_sqrt).float(),
        'channel_count': C_seen,
    }

    if verbose:
        eigvals = np.linalg.eigvalsh(R)
        print(f"\nFréchet mean eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
        print(f"Condition number: {eigvals.max() / eigvals.min():.2f}")

    return result


def compute_frechet_mean_from_model(model, dataloader, enc_dim=512,
                                     eps=1e-2, max_batches=30,
                                     max_iters=100, verbose=True,
                                     warm_start=None, total_channels=144):
    """
    Compute Fréchet mean in the GLOBAL channel space from embedding-space
    covariances that the Riemannian attention actually sees during training.

    Instead of computing a C×C Fréchet mean for one channel count, this
    function accumulates covariances in the full `total_channels × total_channels`
    space. Each dataset's C×C covariance is scattered into the correct positions
    using the global channel indices from channel_list. Positions not seen by
    any dataset get identity (eps*I), so the resulting R_inv_sqrt can always
    be submatrix-indexed by any channel_idx at runtime.

    This handles mixed-channel training (e.g., 22-ch BCI2a, 32-ch FACED,
    62-ch datasets) without shape mismatches.

    Args:
        model:          ApproxAdaptiveRiemannBert instance (on CPU, eval mode)
        dataloader:     yields (eeg, channel_list) where eeg is (B, C, T)
        enc_dim:        embedding dimension D (for normalization)
        eps:            regularization added to covariance diagonal
        max_batches:    maximum number of batches to use
        max_iters:      Karcher flow iterations
        verbose:        print progress
        warm_start:     (total_channels, total_channels) numpy array — previous R
        total_channels: size of the global channel space (default 144)

    Returns:
        dict with keys: 'R', 'R_inv_sqrt', 'R_sqrt', 'channel_count'
    """
    if verbose:
        print(f"Collecting EMBEDDING-SPACE covariances in global {total_channels}-ch space...")

    model.eval()

    # Accumulate sum of covariances and count per (i,j) pair in global space
    G_sum = np.zeros((total_channels, total_channels))
    G_count = np.zeros((total_channels, total_channels))

    n_covs = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            eeg, channel_list = batch
            B, C, T = eeg.shape

            # Run through patch embedding only (same as encoder_forward up to patching)
            x = model.patch(eeg)        # (B, N, C, D)
            N = x.shape[1]
            D = x.shape[3]
            x = x.reshape(B, N, C, D)

            # Add channel + temporal embeddings (same as encoder_forward)
            device = eeg.device
            ch = torch.tensor(channel_list, dtype=torch.long, device=device) \
                if not isinstance(channel_list, torch.Tensor) \
                else channel_list.to(device)
            if ch.dim() == 1:
                ch = ch.unsqueeze(0).expand(B, -1)
            L = N * C
            x_flat = x.reshape(B, L, D)
            chan_id = ch.unsqueeze(1).repeat(1, N, 1).view(B, L)
            x_flat = x_flat + model.channel_embedding_e(chan_id)
            seq_idx = torch.arange(N, device=device, dtype=torch.long)
            eeg_seq = seq_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C).view(B, L)
            x_flat = x_flat + model.temporal_embedding_e(eeg_seq)

            # Reshape to (B*N, C, D)
            x_emb = x_flat.reshape(B, N, C, D).reshape(B * N, C, D)

            # Get global channel indices for this batch
            ch_idx = ch[0].cpu().numpy()  # (C,) global indices

            # Compute mean covariance for this batch (faster than per-sample)
            x_np = x_emb.float().numpy()  # (B*N, C, D)
            # Batch covariance: mean of (x @ x^T / D) over all B*N samples
            # einsum: (B*N, C, D) x (B*N, C, D) -> (C, C)
            cov_mean = np.einsum('bcd,bed->ce', x_np, x_np) / (x_np.shape[0] * D)

            # Regularize to ensure SPD before taking log
            cov_mean = 0.5 * (cov_mean + cov_mean.T)
            cov_mean += eps * np.eye(C)

            # Log-Euclidean mean (Arsigny et al. 2007):
            # R_LE = exp( mean( log(S_i) ) )
            # Accumulate log(S) in the global channel space
            eigvals_c, eigvecs_c = np.linalg.eigh(cov_mean)
            eigvals_c = np.maximum(eigvals_c, 1e-10)
            log_cov = eigvecs_c @ np.diag(np.log(eigvals_c)) @ eigvecs_c.T

            # Scatter log(S) into global space
            idx = np.ix_(ch_idx, ch_idx)
            G_sum[idx] += log_cov * x_np.shape[0]  # weight by num samples
            G_count[idx] += x_np.shape[0]
            n_covs += x_np.shape[0]

            if verbose and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: C={C}, {n_covs} total samples, "
                      f"channels seen: {(G_count.diagonal() > 0).sum()}/{total_channels}")

    # ── Log-Euclidean mean in global space ──
    # R_LE = exp( weighted_mean( log(S_i) ) )
    # For unseen channel pairs, log(I) = 0, so they map back to identity via exp(0) = 1
    channels_seen = int((G_count.diagonal() > 0).sum())

    # Average the accumulated log-covariances
    log_G_mean = np.zeros((total_channels, total_channels))
    mask_seen = G_count > 0
    log_G_mean[mask_seen] = G_sum[mask_seen] / G_count[mask_seen]
    # Unseen positions stay at 0 → exp(0) = identity

    # Symmetrize
    log_G_mean = 0.5 * (log_G_mean + log_G_mean.T)

    if verbose:
        print(f"Total: {n_covs} covariance samples across {channels_seen}/{total_channels} channels")
        print(f"\nComputing log-Euclidean mean (Arsigny et al. 2007)...")

    # Exponentiate back to SPD manifold: R = exp(mean(log(S)))
    eigvals_g, eigvecs_g = np.linalg.eigh(log_G_mean)
    R = eigvecs_g @ np.diag(np.exp(eigvals_g)) @ eigvecs_g.T

    # Symmetrize and ensure SPD
    R = 0.5 * (R + R.T)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-6:
        R += (1e-6 - eigvals.min()) * np.eye(total_channels)

    R_inv_sqrt = _matrix_sqrt_inv_np(R)
    R_sqrt = _matrix_sqrt_np(R)

    result = {
        'R': torch.from_numpy(R).float(),
        'R_inv_sqrt': torch.from_numpy(R_inv_sqrt).float(),
        'R_sqrt': torch.from_numpy(R_sqrt).float(),
        'channel_count': total_channels,
    }

    if verbose:
        eigvals = np.linalg.eigvalsh(R)
        cond = eigvals.max() / eigvals.min()
        print(f"\nLog-Euclidean mean R eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
        print(f"Condition number: {cond:.2f}")
        print(f"Channels with data: {channels_seen}/{total_channels}")
        if cond > 1000:
            print(f"  WARNING: High condition number ({cond:.0f}). "
                  f"Consider increasing eps.")

    return result


class FrechetRefreshCallback(pl.Callback):
    """
    Lightning callback that recomputes the Fréchet mean R every `refresh_every`
    epochs using the model's CURRENT learned embeddings.

    Why: R is initially computed from random-init embeddings. As training
    progresses, the patch projection changes and the embedding-space covariance
    distribution shifts. Periodically refreshing R keeps the pre-whitening
    accurate throughout training.

    How: At the start of every `refresh_every`-th epoch, we:
      1. Put the model in eval mode (no dropout / stochastic depth)
      2. Run `compute_frechet_mean_from_model` on the train dataloader
      3. Inject the new R_inv_sqrt into every encoder layer's adaptive_log
      4. Resume training

    Cost: ~30-60s per refresh (one forward pass through max_batches, no grads).
    With refresh_every=10 and max_epochs=200, that's 20 refreshes = ~15 minutes
    total, negligible compared to training.

    Args:
        train_dataloader: the training dataloader (same one used for training)
        refresh_every:    recompute R every N epochs (default 10)
        max_batches:      batches to collect per refresh (default 100, less than
                          init since we just need to track drift, not full accuracy)
        enc_dim:          embedding dimension (default 512)
        verbose:          print refresh logs
    """

    def __init__(self, train_dataloader, refresh_every=10, max_batches=100,
                 enc_dim=512, verbose=True):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.refresh_every = refresh_every
        self.max_batches = max_batches
        self.enc_dim = enc_dim
        self.verbose = verbose
        self._prev_R = None  # warm-start cache

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Skip epoch 0 (R was just computed at init)
        if epoch == 0 or epoch % self.refresh_every != 0:
            return

        # Check that model actually uses Fréchet
        first_log_map = pl_module.encoder[0].attn.riemannian_bias.adaptive_log
        if not first_log_map.use_frechet:
            return

        if self.verbose:
            print(f"\n[Fréchet Refresh] Epoch {epoch}: recomputing R from current embeddings...")

        # Temporarily move model to CPU for the computation
        device = next(pl_module.parameters()).device
        pl_module.cpu()
        pl_module.eval()

        frechet_result = compute_frechet_mean_from_model(
            pl_module, self.train_dataloader, enc_dim=self.enc_dim,
            max_batches=self.max_batches, verbose=self.verbose,
        )
        new_R_inv_sqrt = frechet_result['R_inv_sqrt']

        # Cache for next warm-start
        self._prev_R = frechet_result['R'].numpy()

        # Inject into every layer
        for layer in pl_module.encoder:
            log_map = layer.attn.riemannian_bias.adaptive_log
            log_map.R_inv_sqrt.copy_(new_R_inv_sqrt)

        # Move back to training device and mode
        pl_module.to(device)
        pl_module.train()

        if self.verbose:
            R_np = frechet_result['R'].numpy()
            eigvals = np.linalg.eigvalsh(R_np)
            print(f"[Fréchet Refresh] Done. Condition number: "
                  f"{eigvals.max() / eigvals.min():.2f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Fréchet mean of EEG covariances")
    parser.add_argument("--output", type=str, default="frechet_mean.pt",
                        help="Output path for the Fréchet mean")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Max batches to collect from dataloader")
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Max Karcher flow iterations")
    args = parser.parse_args()

    print("=" * 60)
    print("FRÉCHET MEAN COMPUTATION")
    print("=" * 60)
    print()
    print("To use this, import and call compute_frechet_mean_from_dataloader:")
    print()
    print("    from MAE_pretraining.frechet_mean import compute_frechet_mean_from_dataloader")
    print("    from MAE_pretraining.load_data import get_dataloader")
    print()
    print("    config = {...}  # your config")
    print("    train_loader, _ = get_dataloader(config)")
    print("    result = compute_frechet_mean_from_dataloader(train_loader)")
    print("    torch.save(result, 'frechet_mean.pt')")
    print()
    print("Then pass the path to your model:")
    print("    model = ApproxAdaptiveRiemannBert(frechet_path='frechet_mean.pt')")
