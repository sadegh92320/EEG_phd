"""
CSP Alignment Analysis — Run with C1-only checkpoint on BCI-IV 2a.

Usage (notebook cell or script):
    %run run_csp_analysis.py --checkpoint /path/to/c1_checkpoint.ckpt
"""
import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from downstream.downstream_model import DownstreamRiemannTransformerPara as Downstream
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader
from analysis.csp_alignment import (
    compute_csp_filters,
    extract_riemannian_spatial_filters,
    compute_alignment,
)


def run_csp(checkpoint_path, data_path="downstream/data/bci_comp_2a",
            config_path="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
            n_csp_pairs=3, num_layers=8, batch_size=32):
    """
    Full CSP alignment pipeline for C1-only checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # ── 1. Load model ──
    print("[1/4] Loading model...")
    model = Downstream(
        checkpoint_path=checkpoint_path,
        enc_dim=512,
        depth_e=8,
        patch_size=16,
        num_classes=4,  # BCI 2a has 4 MI classes
    )
    model.to(device)
    model.eval()
    print(f"   Model loaded. Encoder has {len(model.encoder)} layers.")

    # ── 2. Load BCI-IV 2a data ──
    print("[2/4] Loading BCI-IV 2a data...")
    loader = DownstreamDataLoader(
        data_path=data_path,
        config=config_path,
        custom_dataset_class=Downstream_Dataset,
        base_sfreq=250,  # stored at 250Hz, Downstream_Dataset resamples to 128Hz
    )
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    # Use test set for analysis (unseen data)
    # Dataset returns (eeg, label, channel_id) — we need a collate
    # that provides (eeg, channel_id, label) for the analysis
    def collate_csp(batch):
        eegs, labels, chan_ids = zip(*batch)
        eegs = torch.stack(eegs)
        labels = torch.stack(labels)
        chan_ids = torch.stack(chan_ids)
        return eegs, chan_ids, labels

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_csp, num_workers=0)
    print(f"   Test set: {len(test_ds)} trials")

    # ── 3. Compute classical CSP filters ──
    print("[3/4] Computing CSP filters...")
    all_eeg = []
    all_labels = []
    for eeg, _, label in test_loader:
        all_eeg.append(eeg.numpy())
        all_labels.append(label.numpy())
    all_eeg = np.concatenate(all_eeg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    unique_classes = np.unique(all_labels)
    print(f"   Classes: {unique_classes}, Channels: {all_eeg.shape[1]}")

    csp_filters = {}
    for i, c1 in enumerate(unique_classes):
        for c2 in unique_classes[i + 1:]:
            X1 = all_eeg[all_labels == c1]
            X2 = all_eeg[all_labels == c2]
            W, evals = compute_csp_filters(X1, X2, n_pairs=n_csp_pairs)
            csp_filters[(c1, c2)] = W
            print(f"   CSP({int(c1)} vs {int(c2)}): "
                  f"eigenvalue range [{evals[0]:.4f}, {evals[-1]:.4f}]")

    # ── 4. Extract learned filters & compute alignment ──
    print(f"[4/4] Extracting Riemannian spatial filters from {num_layers} layers...")

    # The extract function needs a dataloader that yields (eeg, channel_list, labels)
    # Our collate already does this
    filters_per_layer, biases_per_layer = extract_riemannian_spatial_filters(
        model, test_loader, device, num_layers=num_layers
    )

    # Compute alignment per layer per class pair
    print("\n" + "=" * 60)
    print("CSP ↔ RIEMANNIAN ALIGNMENT RESULTS")
    print("=" * 60)

    results = {}
    for layer_idx in sorted(filters_per_layer.keys()):
        results[layer_idx] = {}
        layer_aligns = []
        for (c1, c2), W_csp in csp_filters.items():
            if c1 in filters_per_layer[layer_idx]:
                W_learned = filters_per_layer[layer_idx][c1]
                alignment_matrix, mean_align = compute_alignment(
                    W_csp, W_learned, n_compare=2 * n_csp_pairs
                )
                results[layer_idx][(c1, c2)] = {
                    'alignment_matrix': alignment_matrix,
                    'mean_alignment': mean_align,
                }
                layer_aligns.append(mean_align)

        if layer_aligns:
            avg = np.mean(layer_aligns)
            print(f"   Layer {layer_idx}: avg alignment = {avg:.4f}  "
                  f"(across {len(layer_aligns)} class pairs)")

    print("\nInterpretation:")
    print("   0.0–0.3 = low  (model uses different spatial structure)")
    print("   0.3–0.6 = moderate (partial CSP-like structure)")
    print("   0.6–1.0 = high (model implicitly rediscovers CSP)")
    print()

    return results, csp_filters, filters_per_layer, biases_per_layer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="downstream/data/bci_comp_2a")
    parser.add_argument("--n_csp_pairs", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=8)
    args = parser.parse_args()

    run_csp(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        n_csp_pairs=args.n_csp_pairs,
        num_layers=args.num_layers,
    )
