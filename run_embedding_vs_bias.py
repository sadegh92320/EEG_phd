"""
Embedding vs Riemannian Bias — Run with checkpoint on BCI-IV 2a.

Decomposes spatial attention into Q·K (channel embedding driven) vs
Riemannian covariance bias to see which dominates per layer.

Usage:
    python run_embedding_vs_bias.py --checkpoint /path/to/checkpoint.ckpt
"""
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from downstream.downstream_model import DownstreamRiemannTransformerPara as Downstream
from downstream.downstream_dataset import Downstream_Dataset
from downstream.split_data_downstream import DownstreamDataLoader
from analysis.embedding_vs_bias import run_embedding_vs_bias


def run(checkpoint_path, data_path="downstream/data/bci_comp_2a",
        config_path="MAE_pretraining/info_dataset/bci_comp_2a.yaml",
        num_layers=8, batch_size=32, max_batches=None, use_rope=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    print("[1/3] Loading model...")
    model = Downstream(
        checkpoint_path=checkpoint_path,
        enc_dim=512,
        depth_e=8,
        patch_size=16,
        num_classes=4,
        use_rope=use_rope,
    )
    model.to(device)
    model.eval()
    print(f"   Model loaded. Encoder has {len(model.encoder)} layers.")

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

    print(f"[3/3] Running embedding vs bias analysis ({num_layers} layers)...")
    print()

    results = run_embedding_vs_bias(
        model, test_loader, device,
        num_layers=num_layers,
        max_batches=max_batches,
    )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="downstream/data/bci_comp_2a")
    parser.add_argument("--config_path", type=str,
                        default="MAE_pretraining/info_dataset/bci_comp_2a.yaml")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--use_rope", action="store_true", default=False,
                        help="Build encoder with EEG-RoPE (must match pretrained checkpoint).")
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        config_path=args.config_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        use_rope=args.use_rope,
    )
