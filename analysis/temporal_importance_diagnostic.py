"""
Temporal Importance Diagnostic (No Training Required)
=====================================================

Measures how much the encoder's output features depend on temporal
structure by comparing features under normal vs ablated inputs.

Metrics:
    - Cosine similarity between normal and ablated pooled features
    - L2 distance between normal and ablated pooled features
    - Feature variance ratio (ablated / normal)

If cosine sim ≈ 1.0 under shuffle → encoder ignores temporal order.
If cosine sim drops → encoder relies on temporal order.

Usage:
    python analysis/temporal_importance_diagnostic.py \
        --checkpoint /path/to/checkpoint.ckpt
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downstream.downstream_model import DownstreamRiemannTransformerPara as Downstream
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader


def extract_features(model, dataloader, device, mode="normal", max_batches=20):
    """
    Extract mean-pooled encoder features under a given temporal ablation.

    Uses a hook on model.patch to intercept and modify patch output.
    Returns: features (N_samples, D), labels (N_samples,)
    """
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    def ablation_hook(module, input, output):
        """output: (B, N, C, D) — modify temporal dim N."""
        if mode == "normal":
            return output
        B, N, C, D = output.shape
        if mode == "shuffle":
            out = output.clone()
            for b in range(B):
                perm = torch.randperm(N, device=output.device)
                out[b] = output[b, perm]
            return out
        elif mode == "reverse":
            return output.flip(1)
        elif mode == "constant":
            return output[:, :1].expand(-1, N, -1, -1).contiguous()
        elif mode == "channel_only":
            return output.mean(dim=1, keepdim=True).expand(-1, N, -1, -1).contiguous()
        return output

    handle = model.patch.register_forward_hook(ablation_hook)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            eeg, labels, chan_ids = batch
            eeg = eeg.to(device)
            chan_ids = chan_ids.to(device)

            # Run through model but stop before head — get encoder output
            B, C, T = eeg.shape
            x = model.patch(eeg)  # hook modifies this
            N = x.shape[1]
            from einops import rearrange
            x = rearrange(x, "b n c d -> b (n c) d")
            L = x.shape[1]

            # Channel embedding
            cl = chan_ids
            if cl.dim() == 1:
                cl = cl.unsqueeze(0).expand(B, -1)
            x = x + model._get_channel_embedding(cl, N, B, L)

            # Temporal embedding (skip if RoPE)
            if not getattr(model, '_use_rope', False):
                seq_idx = torch.arange(0, N, device=device).unsqueeze(0).unsqueeze(-1)
                seq_idx = seq_idx.repeat(B, 1, C).view(B, L)
                x = x + model.temporal_embedding(seq_idx)

            # Encoder
            channel_idx = cl[0]
            x = model._run_encoder(x, C, channel_idx=channel_idx)
            x = model.norm_enc(x)

            # Mean pool → (B, D)
            features = x.mean(dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    handle.remove()

    return torch.cat(all_features), torch.cat(all_labels)


def run_diagnostic(checkpoint_path, data_path="downstream/data/bci_comp_2a",
                   config_path="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
                   batch_size=32, use_rope=False, max_batches=20,
                   n_shuffle_runs=3):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"RoPE: {use_rope}")
    print()

    # Load model
    print("[1/3] Loading model...")
    model = Downstream(
        checkpoint_path=checkpoint_path,
        enc_dim=512, depth_e=8, patch_size=16,
        num_classes=4,
        use_rope=use_rope,
    )
    model.to(device)
    model.eval()

    # Load data
    print("[2/3] Loading data...")
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

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # ── Extract normal features ──
    print(f"[3/3] Extracting features (max {max_batches} batches)...")
    print()

    feat_normal, labels = extract_features(model, test_loader, device,
                                            mode="normal", max_batches=max_batches)

    conditions = ["shuffle", "reverse", "constant", "channel_only"]
    results = {}

    for cond in conditions:
        if cond == "shuffle":
            # Average over runs
            cos_sims = []
            l2_dists = []
            for _ in range(n_shuffle_runs):
                feat_abl, _ = extract_features(model, test_loader, device,
                                                mode=cond, max_batches=max_batches)
                cos = torch.nn.functional.cosine_similarity(feat_normal, feat_abl, dim=-1)
                l2 = (feat_normal - feat_abl).norm(dim=-1)
                cos_sims.append(cos.mean().item())
                l2_dists.append(l2.mean().item())

            results[cond] = {
                "cosine_sim": np.mean(cos_sims),
                "cosine_std": np.std(cos_sims),
                "l2_dist": np.mean(l2_dists),
            }
        else:
            feat_abl, _ = extract_features(model, test_loader, device,
                                            mode=cond, max_batches=max_batches)
            cos = torch.nn.functional.cosine_similarity(feat_normal, feat_abl, dim=-1)
            l2 = (feat_normal - feat_abl).norm(dim=-1)
            results[cond] = {
                "cosine_sim": cos.mean().item(),
                "l2_dist": l2.mean().item(),
            }

    # ── Also measure class separability ──
    # Simple: compute inter-class vs intra-class distance ratio
    unique_labels = labels.unique()
    class_means_normal = {}
    for c in unique_labels:
        mask = labels == c
        class_means_normal[c.item()] = feat_normal[mask].mean(dim=0)

    # Inter-class distance
    inter_dists = []
    classes = list(class_means_normal.keys())
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            d = (class_means_normal[classes[i]] - class_means_normal[classes[j]]).norm().item()
            inter_dists.append(d)
    mean_inter = np.mean(inter_dists)

    # Intra-class distance
    intra_dists = []
    for c in unique_labels:
        mask = labels == c
        class_feats = feat_normal[mask]
        cm = class_means_normal[c.item()]
        intra_dists.append((class_feats - cm).norm(dim=-1).mean().item())
    mean_intra = np.mean(intra_dists)

    separability = mean_inter / (mean_intra + 1e-8)

    # ── Print results ──
    print("=" * 65)
    print("TEMPORAL IMPORTANCE DIAGNOSTIC")
    print("=" * 65)
    print()
    print(f"{'Condition':<15} {'Cosine Sim':>12} {'L2 Dist':>12} {'Interpretation':>20}")
    print("-" * 65)

    for cond in conditions:
        r = results[cond]
        cos_val = r["cosine_sim"]

        if cos_val > 0.99:
            interp = "No effect"
        elif cos_val > 0.95:
            interp = "Minimal effect"
        elif cos_val > 0.85:
            interp = "Moderate effect"
        else:
            interp = "Strong effect"

        if cond == "shuffle" and "cosine_std" in r:
            print(f"   {cond:<15} {cos_val:>10.4f}+/-{r['cosine_std']:.3f} "
                  f"{r['l2_dist']:>10.4f}   {interp:>15}")
        else:
            print(f"   {cond:<15} {cos_val:>12.4f} {r['l2_dist']:>12.4f}   {interp:>15}")

    print()
    print(f"   Class separability (inter/intra): {separability:.4f}")
    print(f"   Mean inter-class distance:        {mean_inter:.4f}")
    print(f"   Mean intra-class distance:        {mean_intra:.4f}")
    print()
    print("GUIDE:")
    print("   Cosine ~1.0  = ablation doesn't change features = model ignores that info")
    print("   Cosine <0.95 = ablation changes features = model uses that info")
    print("   Shuffle tests ORDER, Constant tests CONTENT, Channel-only tests VARIATION")
    print()
    print("   Higher class separability = better linear probe accuracy (typically)")
    print()

    return results, separability


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="downstream/data/bci_comp_2a")
    parser.add_argument("--config_path", type=str,
                        default="MAE_pretraining/info_dataset/bci_comp_2a.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_shuffle_runs", type=int, default=3)
    args = parser.parse_args()

    run_diagnostic(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config_path,
        batch_size=args.batch_size,
        use_rope=args.use_rope,
        max_batches=args.max_batches,
        n_shuffle_runs=args.n_shuffle_runs,
    )
