"""
Diagnostic: log(S_t) Structure for C3-Mixture Justification
=============================================================

Runs a pretrained C1 checkpoint on BCI-2a test data (or any downstream dataset
with class labels) and captures log(S_t) at every encoder layer via forward
hooks on AdaptiveRiemannianAttentionBias. Outputs two analyses that, together,
tell us whether a mixture-of-prototypes tangent-space centering (C3-Mixture)
can plausibly beat the current single-μ C3.

Analyses
--------
(1) Within-trial vs between-trial variation of log(S_t).
    For each layer ℓ and each trial i with N patches:
       μ_i^ℓ         = mean_t  log(S_{i,t}^ℓ)
       σ_intra^ℓ     = mean_{i,t} ‖log(S_{i,t}^ℓ) − μ_i^ℓ‖_F
       σ_inter^ℓ     = mean_i    ‖μ_i^ℓ − μ_•^ℓ‖_F           where μ_•^ℓ = mean_i μ_i^ℓ
       ratio_ℓ       = σ_intra / σ_inter

    Interpretation:
      ratio ≤ 0.3  → brain state ≈ constant within a trial. One prototype per
                      trial (or the global μ) is already near-optimal; mixture
                      cannot add much.
      ratio ~ 0.5–1.0 → within-trial shifts are comparable to between-trial
                      differences. Per-patch state assignment (C3-Mixture) is
                      justified.
      ratio > 1.0  → log(S) is dominated by patch-local noise, between-trial
                      structure is weak. Neither C3 nor C3-Mixture will be
                      a strong prior — investigate whether covariance is even
                      stable over a single 4 s trial.

(2) Multimodality of trial-mean log(S).
    Vectorize the upper triangle of each trial-mean log(S), run GMM with
    K ∈ {1, 2, 4, 8, 16} under diagonal covariance, record BIC. Also
    project to 2D via PCA, colored by class label (if available).

    Interpretation:
      BIC optimum at K=1 → trial-mean log(S) is unimodal. C3-Mixture collapses
                           to C3 in the best case. Kill the contribution.
      BIC optimum at K>1 → trial-mean log(S) is genuinely multimodal. If the
                           PCA figure also shows class-coherent clusters, the
                           prototype story has legs.

The script prints both results per layer and writes two figures
(bic_<layer>.png, pca_<layer>.png) under analysis/figures/logS_structure/.

Usage
-----
    python analysis/diagnostic_logS_structure.py \
        --checkpoint lightning_logs/version_XX/checkpoints/mae-epoch=YY-val_mse=ZZ.ckpt \
        --data_path downstream/data/bci_comp_2a \
        --config_path MAE_pretraining/info_dataset/bci_comp_2a.yaml \
        --layers 0,3,6,7 \
        --max_trials 200

Assumptions
-----------
  - Checkpoint is a C1-style AdaptiveRiemannianParallelTransformer (mu_log
    optional — if present it is subtracted, doesn't affect within/between
    ratio or multimodality of log(S)).
  - BCI-2a is the default dataset (22 channels, 4 classes). Script works on
    any dataset the downstream loader understands.
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.downstream_model import DownstreamRiemannTransformerPara as Downstream
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader


# --------------------------------------------------------------------------
# Hook: capture log(S) tensors returned by AdaptiveRiemannianAttentionBias
# --------------------------------------------------------------------------

class LogSCollector:
    """Registers forward hooks on every attn.riemannian_bias module to store
    the L tensor (= log(S) possibly minus learned μ). Shape per call:
    (B*N, C, C)."""

    def __init__(self, model, layers):
        self.layers = layers
        self.storage = {l: [] for l in layers}
        self.handles = []
        for l, transformer in enumerate(model.encoder):
            if l not in layers:
                continue
            mod = transformer.attn.riemannian_bias
            h = mod.register_forward_hook(self._make_hook(l))
            self.handles.append(h)

    def _make_hook(self, layer):
        def _hook(module, inputs, output):
            # output is (bias, L) where L is (B*N, C, C)
            _, L = output
            self.storage[layer].append(L.detach().cpu())
        return _hook

    def pop_batch(self):
        """After running one batch, return list of (B*N,C,C) tensors per layer
        and clear storage. Also clear in case hooks fire multiple times per
        module within one batch (they shouldn't, but defensive)."""
        out = {l: torch.cat(v, dim=0) if v else None for l, v in self.storage.items()}
        for l in self.storage:
            self.storage[l].clear()
        return out

    def close(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
# Core analyses
# --------------------------------------------------------------------------

def within_between_variation(logS, N):
    """
    logS: (num_trials * N, C, C) tensor of log(S_{i,t}) for a single layer,
          laid out as trial-major, patch-minor.
    N   : number of patches per trial.

    Returns dict with sigma_intra, sigma_inter, ratio.
    """
    num_trials = logS.shape[0] // N
    if num_trials * N != logS.shape[0]:
        raise ValueError(f"logS row count {logS.shape[0]} not divisible by N={N}")

    C = logS.shape[-1]
    x = logS.view(num_trials, N, C, C).double()  # (num_trials, N, C, C)

    mu_trial  = x.mean(dim=1, keepdim=True)                      # (num_trials, 1, C, C)
    mu_global = mu_trial.mean(dim=0, keepdim=True)               # (1, 1, C, C)

    intra = (x - mu_trial).flatten(2).norm(dim=-1).mean().item()
    inter = (mu_trial - mu_global).flatten(2).norm(dim=-1).mean().item()

    return {
        "sigma_intra": intra,
        "sigma_inter": inter,
        "ratio": intra / (inter + 1e-12),
        "num_trials": num_trials,
        "N": N,
        "C": C,
    }


def multimodality_gmm(trial_means, k_list=(1, 2, 4, 8, 16)):
    """
    trial_means: (num_trials, C, C) trial-mean log(S) per trial.
    Returns list of (K, BIC) and the K with minimum BIC.
    """
    from sklearn.mixture import GaussianMixture

    C = trial_means.shape[-1]
    iu = np.triu_indices(C)
    X = trial_means.view(trial_means.shape[0], C, C).double().numpy()
    X = X[:, iu[0], iu[1]]  # (num_trials, C*(C+1)/2)

    # Diagonal covariance to keep degrees of freedom manageable at small N.
    bics = []
    for K in k_list:
        if K >= X.shape[0]:
            bics.append((K, float("inf")))
            continue
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="diag",
            reg_covar=1e-4,
            max_iter=200,
            n_init=2,
            random_state=0,
        )
        gmm.fit(X)
        bics.append((K, gmm.bic(X)))

    k_best = min(bics, key=lambda kb: kb[1])[0]
    return bics, k_best, X


def pca_2d(X, labels=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    return Z, var


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(checkpoint, data_path, config_path, layers, max_trials,
        batch_size, use_rope, num_classes, out_dir):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[device] {device}")
    print(f"[ckpt]   {checkpoint}")
    print(f"[data]   {data_path}")
    print(f"[layers] {layers}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    # Build model (loads checkpoint inside __init__ when checkpoint is given)
    model = Downstream(
        checkpoint_path=checkpoint,
        enc_dim=512, depth_e=8, patch_size=16,
        num_classes=num_classes,
        use_rope=use_rope,
    )
    model.to(device)
    model.eval()

    # Data
    loader = DownstreamDataLoader(
        data_path=data_path,
        config=config_path,
        custom_dataset_class=Downstream_Dataset,
        base_sfreq=250,
    )
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    def collate_fn(batch):
        eegs, labels, chan_ids = zip(*batch)
        return torch.stack(eegs), torch.stack(labels), torch.stack(chan_ids)

    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=0)

    # Register hooks
    collector = LogSCollector(model, layers)

    # Iterate data, collect log(S) per trial per layer
    all_logS = {l: [] for l in layers}  # list of (B*N, C, C) per batch
    all_labels = []
    N_patches = None
    C = None

    total_trials = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            eeg, labels, chan_ids = batch
            if total_trials >= max_trials:
                break

            eeg = eeg.to(device)
            chan_ids = chan_ids.to(device)
            B = eeg.shape[0]

            # Run forward — hooks capture log(S) inside
            _ = model(eeg, chan_ids)

            per_layer = collector.pop_batch()
            # Each per_layer[l] has shape (B*N, C, C). Record N once.
            example = next(v for v in per_layer.values() if v is not None)
            if N_patches is None:
                N_patches = example.shape[0] // B
                C = example.shape[-1]
                print(f"[shape]  N_patches={N_patches}  C={C}")

            for l in layers:
                # Reorder (B*N, C, C) so rows are trial-major, patch-minor.
                # The hook flattens (B*N) with (b,n) order = n-fastest per our
                # code path (BN = B*N from rearrange 'b (n c) d -> (b n) c d'),
                # which iterates n outermost within a batch item. That means
                # rows already go (trial 0, n=0), (trial 0, n=1), ... — trial-major.
                # Verified via the code path in AdaptiveRiemannianAttentionBias.forward.
                all_logS[l].append(per_layer[l])

            all_labels.append(labels)
            total_trials += B

    collector.close()

    labels = torch.cat(all_labels, dim=0)[:total_trials].numpy()
    for l in layers:
        all_logS[l] = torch.cat(all_logS[l], dim=0)
        # Truncate to full trials
        full = (all_logS[l].shape[0] // N_patches) * N_patches
        all_logS[l] = all_logS[l][:full]
    labels = labels[: all_logS[layers[0]].shape[0] // N_patches]

    print(f"[collected] {all_logS[layers[0]].shape[0] // N_patches} trials"
          f" × {N_patches} patches × {C}×{C}")
    print()

    # ---- Analysis 1: within / between ----
    print("=" * 74)
    print("ANALYSIS 1  —  Within-trial vs between-trial variation of log(S_t)")
    print("=" * 74)
    print(f"{'layer':>6} {'σ_intra':>14} {'σ_inter':>14} {'ratio':>10}   verdict")
    print("-" * 74)
    variation_rows = []
    for l in layers:
        v = within_between_variation(all_logS[l], N_patches)
        if v["ratio"] < 0.3:
            verdict = "stable-within-trial  (mixture weak)"
        elif v["ratio"] < 1.0:
            verdict = "state shifts          (mixture plausible)"
        else:
            verdict = "patch-local noise     (prior weak overall)"
        print(f"{l:>6} {v['sigma_intra']:>14.4f} {v['sigma_inter']:>14.4f}"
              f" {v['ratio']:>10.3f}   {verdict}")
        variation_rows.append((l, v))
    print()

    # ---- Analysis 2: multimodality ----
    print("=" * 74)
    print("ANALYSIS 2a —  Multimodality of TRIAL-MEAN log(S) (GMM BIC sweep)")
    print("=" * 74)

    trial_results = {}
    for l in layers:
        trial_mean = all_logS[l].view(-1, N_patches, C, C).mean(dim=1)
        try:
            bics, k_best, X_trial = multimodality_gmm(trial_mean)
        except ImportError:
            print("[warn] sklearn not available — skip GMM/PCA")
            return

        row = "  ".join([f"K={K}:{bic:>10.1f}" for K, bic in bics])
        print(f"layer {l}  {row}   →  best K={k_best}")
        trial_results[l] = (bics, k_best, X_trial)

    # ---- Analysis 2b: patch-level multimodality ----
    print()
    print("=" * 74)
    print("ANALYSIS 2b —  Multimodality of PER-PATCH log(S) (GMM BIC sweep)")
    print("=" * 74)
    print("If K_patch matches K_trial, per-patch soft assignment (C3-Mixture's")
    print("design) has signal at the patch level. If K_patch=1 but K_trial>1,")
    print("patches are isotropic noise around the trial center → mixture will")
    print("collapse to C3 under per-patch assignment.")
    print()

    patch_results = {}
    rng = np.random.default_rng(0)
    for l in layers:
        X_all_patches = all_logS[l].view(-1, C, C)
        # subsample to keep GMM cheap
        n_patch = min(5000, X_all_patches.shape[0])
        idx = rng.choice(X_all_patches.shape[0], n_patch, replace=False)
        X_patch = X_all_patches[idx]
        try:
            bics_p, k_best_p, _ = multimodality_gmm(X_patch)
        except Exception as e:
            print(f"layer {l}  [patch GMM failed: {e}]")
            continue
        row_p = "  ".join([f"K={K}:{bic:>10.1f}" for K, bic in bics_p])
        print(f"layer {l}  {row_p}   →  best K={k_best_p}   [N={n_patch}]")
        patch_results[l] = (bics_p, k_best_p)

    # ---- Plots: BIC trial + BIC patch + PCA ----
    print()
    for l in layers:
        bics, k_best, X_trial = trial_results[l]
        bics_p, k_best_p = patch_results.get(l, (None, None))

        # Plot BIC (trial + patch) + PCA
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            Ks, scores = zip(*bics)
            axes[0].plot(Ks, scores, "o-", color="C0", label="trial-mean")
            axes[0].axvline(k_best, color="C0", ls="--", alpha=0.6,
                            label=f"trial best K={k_best}")
            if bics_p is not None:
                Ks_p, scores_p = zip(*bics_p)
                ax2 = axes[0].twinx()
                ax2.plot(Ks_p, scores_p, "s-", color="C3", label="per-patch")
                ax2.axvline(k_best_p, color="C3", ls=":", alpha=0.6,
                            label=f"patch best K={k_best_p}")
                ax2.set_ylabel("BIC (per-patch)", color="C3")
                ax2.tick_params(axis="y", labelcolor="C3")
            axes[0].set_xscale("log", base=2)
            axes[0].set_xlabel("K (GMM components)")
            axes[0].set_ylabel("BIC (trial-mean)", color="C0")
            axes[0].tick_params(axis="y", labelcolor="C0")
            axes[0].set_title(f"Layer {l}  —  BIC sweep (both levels)")
            axes[0].grid(True, alpha=0.3)

            Z, var = pca_2d(X_trial)
            labels_arr = labels[: X_trial.shape[0]]
            unique = np.unique(labels_arr)
            for c in unique:
                mask = labels_arr == c
                axes[1].scatter(Z[mask, 0], Z[mask, 1], s=16, alpha=0.65,
                                label=f"class {int(c)}")
            axes[1].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
            axes[1].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
            axes[1].set_title(f"Layer {l}  —  trial-mean log(S) PCA-2D")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            # Third panel: patch-level PCA colored by trial-class, to see if
            # patches still cluster by class (vs. being uniform noise).
            X_all = all_logS[l].view(-1, N_patches, C, C)
            iu = np.triu_indices(C)
            n_trials_plot = X_all.shape[0]
            # flatten (trial, patch, C, C) → (trial*patch, D)
            flat = X_all.view(-1, C, C).double().numpy()[:, iu[0], iu[1]]
            # label each patch by its trial's class
            patch_labels = np.repeat(labels_arr[:n_trials_plot], N_patches)
            # subsample for plot speed
            idx_plot = rng.choice(flat.shape[0], min(4000, flat.shape[0]), replace=False)
            Zp, varp = pca_2d(flat[idx_plot])
            pl = patch_labels[idx_plot]
            for c in unique:
                mask = pl == c
                axes[2].scatter(Zp[mask, 0], Zp[mask, 1], s=4, alpha=0.4,
                                label=f"class {int(c)}")
            axes[2].set_xlabel(f"PC1 ({varp[0]*100:.1f}%)")
            axes[2].set_ylabel(f"PC2 ({varp[1]*100:.1f}%)")
            axes[2].set_title(f"Layer {l}  —  per-patch log(S) PCA-2D")
            axes[2].legend(fontsize=8, markerscale=2)
            axes[2].grid(True, alpha=0.3)

            fig.tight_layout()
            out_png = os.path.join(out_dir, f"layer_{l:02d}.png")
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            print(f"layer {l}  saved → {out_png}")
        except ImportError:
            print("[warn] matplotlib not available — skip plots")

    print()
    print("=" * 74)
    print("GLOBAL VERDICT")
    print("=" * 74)

    # Combine trial-level K and patch-level K across mid/late layers (skip l=0).
    mid_late = [l for l in layers if l > 0]
    trial_ks = [trial_results[l][1] for l in mid_late]
    patch_ks = [patch_results[l][1] for l in mid_late if l in patch_results]

    trial_multimodal = sum(k > 1 for k in trial_ks) >= max(1, len(trial_ks) // 2)
    patch_multimodal = (
        sum(k > 1 for k in patch_ks) >= max(1, len(patch_ks) // 2)
        if patch_ks else False
    )

    if trial_multimodal and patch_multimodal:
        verdict_global = (
            "TRIAL-MEAN log(S) is multimodal AND PER-PATCH log(S) is multimodal.\n"
            "  → Per-patch soft assignment has real signal to latch onto.\n"
            "  → KEEP C3-Mixture.  Set K = typical best-K from mid/late layers."
        )
    elif trial_multimodal and not patch_multimodal:
        verdict_global = (
            "TRIAL-MEAN log(S) is multimodal but PER-PATCH is not.\n"
            "  → Patches within a trial are isotropic noise around the trial center.\n"
            "  → Per-patch soft assignment (current C3-Mixture design) will collapse\n"
            "    to uniform. Either: (a) redesign assignment at trial/EMA level, or\n"
            "    (b) fall back to C3 only."
        )
    elif not trial_multimodal:
        verdict_global = (
            "TRIAL-MEAN log(S) is effectively unimodal in mid/late layers.\n"
            "  → Single global μ (C3) already captures it.\n"
            "  → DROP C3-Mixture.  Paper falls back to C1 + C3."
        )
    print(verdict_global)
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to C1 MAE checkpoint (.ckpt)")
    parser.add_argument("--data_path", type=str,
                        default="downstream/data/bci_comp_2a")
    parser.add_argument("--config_path", type=str,
                        default="MAE_pretraining/info_dataset/bci_comp_2a.yaml")
    parser.add_argument("--layers", type=str, default="0,3,6,7",
                        help="Comma-separated layer indices to analyze")
    parser.add_argument("--max_trials", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="BCI-2a=4, FACED=9, etc. Affects head size only.")
    parser.add_argument("--out_dir", type=str,
                        default="analysis/figures/logS_structure")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    run(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config_path,
        layers=layers,
        max_trials=args.max_trials,
        batch_size=args.batch_size,
        use_rope=args.use_rope,
        num_classes=args.num_classes,
        out_dir=args.out_dir,
    )
