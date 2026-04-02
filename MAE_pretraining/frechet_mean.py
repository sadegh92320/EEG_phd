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


def compute_frechet_mean(covariances, max_iters=100, tol=1e-8, verbose=True):
    """
    Compute the Riemannian Fréchet mean of SPD matrices via iterative algorithm.

    Uses the Karcher flow / gradient descent on the SPD manifold:
        1. Initialize R = arithmetic mean (a decent starting point)
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

    Returns:
        R: (C, C) numpy array — Fréchet mean
    """
    N = len(covariances)
    C = covariances[0].shape[0]

    # Initialize with arithmetic mean (reasonable for SPD near each other)
    R = np.mean(covariances, axis=0)

    # Ensure R is SPD
    R = 0.5 * (R + R.T)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-8:
        R += (1e-8 - eigvals.min()) * np.eye(C)

    for it in range(max_iters):
        R_inv_sqrt = _matrix_sqrt_inv_np(R)
        R_sqrt = _matrix_sqrt_np(R)

        # Compute average tangent vector
        T_avg = np.zeros((C, C))
        for S in covariances:
            # Whitened matrix: R^{-1/2} S R^{-1/2}
            M = R_inv_sqrt @ S @ R_inv_sqrt
            # Symmetrize (numerical safety)
            M = 0.5 * (M + M.T)
            # Log map at identity
            eigvals_m, eigvecs_m = np.linalg.eigh(M)
            eigvals_m = np.maximum(eigvals_m, 1e-10)
            log_M = eigvecs_m @ np.diag(np.log(eigvals_m)) @ eigvecs_m.T
            T_avg += log_M

        T_avg /= N

        # Check convergence
        step_size = np.linalg.norm(T_avg, 'fro')
        if verbose and (it % 10 == 0 or step_size < tol):
            print(f"  Karcher iter {it:3d}: ||T_avg||_F = {step_size:.6e}")

        if step_size < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break

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
                                     eps=1e-5, max_batches=200,
                                     max_iters=100, verbose=True):
    """
    Compute Fréchet mean from the EMBEDDING-SPACE covariances that the
    Riemannian attention actually sees during training.

    IMPORTANT: The Riemannian attention branch computes S = x @ x^T / D
    where x is (B*N, C, D) — the embedded patch representations, NOT raw EEG.
    The Fréchet mean must come from the same space, otherwise R^{-1/2} S R^{-1/2}
    won't bring S near I and Padé will produce garbage / NaN.

    This function:
    1. Runs batches through the model's patch embedding + positional encoding
       (no masking, no transformer layers — just the input pipeline)
    2. Reshapes to (B*N, C, D) — same as what the attention sees
    3. Computes per-timestep covariance S = x @ x^T / D + eps*I
    4. Collects all S matrices and computes their Fréchet mean

    Args:
        model:       ApproxAdaptiveRiemannBert instance (on CPU, eval mode)
        dataloader:  yields (eeg, channel_list) where eeg is (B, C, T)
        enc_dim:     embedding dimension D (for normalization)
        eps:         regularization added to covariance
        max_batches: maximum number of batches to use
        max_iters:   Karcher flow iterations
        verbose:     print progress

    Returns:
        dict with keys: 'R', 'R_inv_sqrt', 'R_sqrt', 'channel_count'
    """
    if verbose:
        print("Collecting EMBEDDING-SPACE covariances from model...")

    model.eval()
    all_covs = []
    C_seen = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            eeg, channel_list = batch
            B, C, T = eeg.shape

            if C_seen is None:
                C_seen = C
            elif C != C_seen:
                continue

            # Run through patch embedding only (same as encoder_forward up to patching)
            x = model.patch(eeg)        # (B, N, C, D)
            N = x.shape[1]
            D = x.shape[3]
            x = x.reshape(B, N, C, D)   # explicit shape

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

            # Reshape to (B*N, C, D) — exactly what Riemannian attention receives
            x_emb = x_flat.reshape(B, N, C, D).reshape(B * N, C, D)

            # Compute per-timestep covariance: S = x @ x^T / D
            x_np = x_emb.float().numpy()
            for i in range(x_np.shape[0]):
                xi = x_np[i]  # (C, D)
                cov = (xi @ xi.T) / D + eps * np.eye(C)
                # Ensure SPD
                cov = 0.5 * (cov + cov.T)
                eigvals = np.linalg.eigvalsh(cov)
                if eigvals.min() < 1e-8:
                    cov += (1e-8 - eigvals.min()) * np.eye(C)
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
        cond = eigvals.max() / eigvals.min()
        print(f"\nFréchet mean eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
        print(f"Condition number: {cond:.2f}")
        if cond > 1000:
            print(f"  WARNING: High condition number ({cond:.0f}). "
                  f"R_inv_sqrt may have large entries → risk of overflow. "
                  f"Consider increasing eps or checking data normalization.")

    return result


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
