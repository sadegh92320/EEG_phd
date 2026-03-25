"""
Quick diagnostic: verify EEGPT checkpoint loads correctly and produces
meaningful features. Run this before the full training loop.

Usage:
    python -m eegpt.debug_eegpt \
        --gdf_dir /path/to/BCICIV_2a_gdf \
        --ckpt_path /path/to/eegpt_mcae_58chs_4s_large4E.ckpt
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downstream.models.foundation_models.eegpt import (
    EEGTransformer, Conv1dWithConstraint, CHANNEL_DICT,
)
from eegpt.reproduce_bci2a import (
    extract_trials_from_gdf, euclidean_alignment, USE_CHANNELS, seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Check checkpoint ──
    print("=" * 50)
    print("1. CHECKPOINT INSPECTION")
    print("=" * 50)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Top-level keys: {list(ckpt.keys())}")

    if "state_dict" in ckpt:
        all_keys = list(ckpt["state_dict"].keys())
        encoder_keys = [k for k in all_keys if k.startswith("target_encoder.")]
        other_keys = [k for k in all_keys if not k.startswith("target_encoder.")]
        print(f"  Total keys in state_dict: {len(all_keys)}")
        print(f"  target_encoder.* keys: {len(encoder_keys)}")
        print(f"  Other keys: {len(other_keys)}")
        if other_keys:
            print(f"    Examples: {other_keys[:5]}")
        print(f"  First 5 encoder keys: {encoder_keys[:5]}")
    else:
        print("  WARNING: no 'state_dict' key found!")
        print(f"  Available keys: {list(ckpt.keys())}")
        return

    # ── 2. Build encoder and load weights ──
    print("\n" + "=" * 50)
    print("2. MODEL CONSTRUCTION & WEIGHT LOADING")
    print("=" * 50)

    target_encoder = EEGTransformer(
        img_size=[19, 1024],
        patch_size=64,
        embed_num=4,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    chans_id = target_encoder.prepare_chan_ids(USE_CHANNELS)
    print(f"  chans_id shape: {chans_id.shape}, dtype: {chans_id.dtype}")
    print(f"  chans_id values: {chans_id.squeeze().tolist()}")

    # Load weights
    state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("target_encoder."):
            state[k[15:]] = v

    missing, unexpected = target_encoder.load_state_dict(state, strict=False)
    print(f"\n  Loaded {len(state)} keys into encoder")
    if missing:
        print(f"  MISSING keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): {unexpected[:10]}")
    if not missing and not unexpected:
        print("  All keys matched perfectly!")

    target_encoder = target_encoder.to(device)
    target_encoder.eval()

    # ── 3. Test with real data ──
    print("\n" + "=" * 50)
    print("3. FORWARD PASS WITH REAL DATA")
    print("=" * 50)

    train_file = os.path.join(args.gdf_dir, "A01T.gdf")
    if not os.path.exists(train_file):
        print(f"  File not found: {train_file}")
        print("  Trying with random data instead...")
        X_train = np.random.randn(10, 22, 1024).astype(np.float32)
        y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    else:
        X_train, y_train = extract_trials_from_gdf(train_file)
        X_train = euclidean_alignment(X_train)
        print(f"  Loaded {X_train.shape[0]} trials from A01T.gdf")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_train stats: mean={X_train.mean():.6f}, std={X_train.std():.6f}, "
          f"min={X_train.min():.6f}, max={X_train.max():.6f}")
    print(f"  y_train distribution: {np.bincount(y_train)}")

    # Convert to mV (matching paper)
    x = torch.from_numpy(X_train[:8] * 1e3).float().to(device)
    print(f"\n  Input tensor (after mV scaling):")
    print(f"    shape: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")

    # Apply chan_conv (22 → 19)
    chan_conv = Conv1dWithConstraint(22, 19, 1, max_norm=1).to(device)
    x_proj = chan_conv(x)
    print(f"\n  After chan_conv (22→19):")
    print(f"    shape: {x_proj.shape}, mean={x_proj.mean():.4f}, std={x_proj.std():.4f}")

    # Forward through encoder
    with torch.no_grad():
        chans_id_dev = chans_id.to(device)
        print(f"\n  Calling encoder with x={x_proj.shape}, chans_id={chans_id_dev.shape}")

        z = target_encoder(x_proj, chans_id_dev)
        print(f"\n  Encoder output:")
        print(f"    shape: {z.shape}")
        print(f"    mean={z.mean():.6f}, std={z.std():.6f}")
        print(f"    min={z.min():.6f}, max={z.max():.6f}")
        print(f"    any NaN: {torch.isnan(z).any()}")
        print(f"    all zeros: {(z == 0).all()}")

        # Check variance across samples (should be different per sample)
        per_sample_mean = z.flatten(1).mean(dim=1)
        print(f"    per-sample means: {per_sample_mean.cpu().numpy()}")

        # Flatten and check linear separability
        z_flat = z.flatten(1)
        print(f"\n  Flattened features: {z_flat.shape}")
        print(f"    Feature variance across batch: {z_flat.var(dim=0).mean():.6f}")

    # ── 4. Quick linear probe test ──
    print("\n" + "=" * 50)
    print("4. SANITY CHECK: 1-epoch linear probe on 8 samples")
    print("=" * 50)

    # Get features for all training data
    with torch.no_grad():
        all_features = []
        batch_size = 32
        for i in range(0, len(X_train), batch_size):
            batch = torch.from_numpy(X_train[i:i+batch_size] * 1e3).float().to(device)
            batch = chan_conv(batch)
            feat = target_encoder(batch, chans_id_dev)
            all_features.append(feat.flatten(1).cpu())
        all_features = torch.cat(all_features, dim=0)

    print(f"  Feature matrix: {all_features.shape}")
    print(f"  Feature stats: mean={all_features.mean():.4f}, std={all_features.std():.4f}")

    # Check if features differ between classes
    for c in range(4):
        mask = y_train == c
        if mask.sum() > 0:
            class_mean = all_features[mask].mean(dim=0).mean()
            class_std = all_features[mask].std(dim=0).mean()
            print(f"  Class {c} ({mask.sum()} samples): feat_mean={class_mean:.4f}, feat_std={class_std:.4f}")

    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    print("=" * 50)
    if z.std() < 1e-4:
        print("  PROBLEM: Encoder output has near-zero variance.")
        print("  → Weights likely didn't load correctly.")
    elif (z == 0).all():
        print("  PROBLEM: Encoder output is all zeros.")
        print("  → Check load_state_dict for missing keys.")
    elif torch.isnan(z).any():
        print("  PROBLEM: NaN in encoder output.")
        print("  → Check data scaling or model numerical stability.")
    else:
        print("  Encoder output looks healthy (non-zero, variable, no NaN).")
        print("  If accuracy is still 25%, the issue may be:")
        print("    - Data scaling (try without *1e3)")
        print("    - EA normalization issues")
        print("    - Learning rate too high/low")
        print("    - Need more epochs")


if __name__ == "__main__":
    main()
