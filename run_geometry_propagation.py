"""
Geometry Propagation Analysis — Run with checkpoint on BCI-IV 2a.

Usage:
    python run_geometry_propagation.py --checkpoint /path/to/checkpoint.ckpt
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
from analysis.geometry_propagation import run_geometry_propagation


def run(checkpoint_path, data_path="downstream/data/bci_comp_2a",
        config_path="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
        num_layers=8, batch_size=32, max_batches=None):
    """
    Full geometry propagation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # ── 1. Load model ──
    print("[1/3] Loading model...")
    model = Downstream(
        checkpoint_path=checkpoint_path,
        enc_dim=512,
        depth_e=8,
        patch_size=16,
        num_classes=4,
    )
    model.to(device)
    model.eval()
    print(f"   Model loaded. Encoder has {len(model.encoder)} layers.")

    # ── 2. Load BCI-IV 2a data ──
    print("[2/3] Loading BCI-IV 2a data...")
    loader = DownstreamDataLoader(
        data_path=data_path,
        config=config_path,
        custom_dataset_class=Downstream_Dataset,
        base_sfreq=250,
    )
    train_ds, val_ds, test_ds = loader.get_data_for_population()

    def collate_fn(batch):
        eegs, labels, chan_ids = zip(*batch)
        eegs = torch.stack(eegs)
        labels = torch.stack(labels)
        chan_ids = torch.stack(chan_ids)
        return eegs, chan_ids, labels

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    print(f"   Test set: {len(test_ds)} trials")

    # ── 3. Run analysis ──
    print(f"[3/3] Running geometry propagation analysis ({num_layers} layers)...")
    print()

    results = run_geometry_propagation(
        model, test_loader, device,
        num_layers=num_layers,
        max_batches=max_batches,
    )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Geometry Propagation Analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default="downstream/data/bci_comp_2a",
                        help="Path to BCI-IV 2a data")
    parser.add_argument("--config_path", type=str,
                        default="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
                        help="Path to dataset config yaml")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of encoder layers to analyze")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches (for quick test)")
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
