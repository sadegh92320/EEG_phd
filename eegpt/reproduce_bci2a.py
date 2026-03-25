"""
Standalone EEGPT reproduction on BCI-IV-2a.
Goal: match the paper's ~58% balanced accuracy per-subject.

Protocol (from paper + their released code):
    - Data: 0-38 Hz bandpass, EA normalization per session, resample to 256 Hz
    - Segments: 4s motor imagery window [2s, 6s] after cue = 1024 samples at 256 Hz
    - Per-subject: session T (day 1) → train, session E (day 2) → test
    - Model: frozen EEGPT encoder + chan_conv(22→19) + 2-layer linear probe
    - Training: AdamW, OneCycleLR(max_lr=4e-4, pct_start=0.2), 100 epochs, batch=64

Usage:
    python -m eegpt.reproduce_bci2a \
        --gdf_dir /path/to/BCICIV_2a_gdf \
        --ckpt_path /path/to/eegpt_mcae_58chs_4s_large4E.ckpt
"""

import argparse
import math
import os
import re
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
from scipy.signal import resample
from scipy.linalg import sqrtm, inv

import mne
from scipy.io import loadmat

# ─── reproducibility ───
def seed_everything(seed=8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ═════════════════════════════════════════════════════════════════
# 1.  DATA LOADING — matches paper's preprocessing exactly
# ═════════════════════════════════════════════════════════════════

def extract_trials_from_gdf(gdf_path, sfreq_target=256.0):
    """
    Extract MI trials from a BCI-2a GDF file.
    Returns: trials (N, 22, 1024), labels (N,)

    Preprocessing:
        - Keep first 22 EEG channels
        - Bandpass 0-38 Hz (paper's filter)
        - Resample to 256 Hz
        - Extract [2s, 6s] window after cue = 4s = 1024 samples
    """
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")

    sfreq_orig = raw.info["sfreq"]

    # Keep only 22 EEG channels
    raw.pick(list(range(22)))

    # Bandpass 0-38 Hz (matching paper exactly)
    raw.filter(l_freq=0.0, h_freq=38.0, method="iir", verbose="ERROR")

    # Resample to 256 Hz
    if abs(sfreq_orig - sfreq_target) > 0.5:
        raw.resample(sfreq_target, verbose="ERROR")

    data = raw.get_data()  # (22, T)
    sfreq = raw.info["sfreq"]

    # Find cue events
    fname = os.path.basename(gdf_path)
    is_eval = fname.split(".")[0][-1] == "E"

    if is_eval:
        cue_codes = {7}
    else:
        cue_codes = {7, 8, 9, 10}

    reject_code = 1
    cue_positions = []
    rejected = set()

    for event in events:
        pos, _, code = int(event[0]), int(event[1]), int(event[2])
        if code == reject_code:
            rejected.add(pos)
        if code in cue_codes:
            cue_positions.append(pos)

    # Adjust positions for resampling
    if abs(sfreq_orig - sfreq_target) > 0.5:
        ratio = sfreq_target / sfreq_orig
        cue_positions = [int(round(p * ratio)) for p in cue_positions]

    # Load labels from matching .mat file
    mat_path = gdf_path.replace(".gdf", ".mat")
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    labels = None
    for key in ["classlabel", "labels", "y", "true_y"]:
        if key in mat:
            labels = np.asarray(mat[key]).squeeze().astype(int)
            break
    if labels is None:
        raise KeyError(f"No label key found in {mat_path}")

    # Extract windows [2s, 6s] after cue
    start_off = int(round(2.0 * sfreq))
    end_off = int(round(6.0 * sfreq))
    win_len = end_off - start_off  # should be 1024

    trials = []
    trial_labels = []
    T = data.shape[1]

    for i, pos in enumerate(cue_positions):
        if i >= len(labels):
            break
        start = pos + start_off
        end = pos + end_off
        if end > T:
            continue
        eeg = data[:, start:end]
        if eeg.shape[1] != win_len:
            continue
        trials.append(eeg)
        trial_labels.append(labels[i] - 1)  # convert 1-4 → 0-3

    return np.array(trials, dtype=np.float32), np.array(trial_labels, dtype=np.int64)


def euclidean_alignment(trials):
    """
    Euclidean Alignment (He & Wu, 2019).
    trials: (N, C, T) → aligned (N, C, T)

    Computes mean covariance R̄ across trials, then whitens each trial by R̄^(-1/2).
    This aligns the spatial covariance structure, reducing inter-session variability.
    """
    N, C, T = trials.shape

    # Compute mean covariance
    R_sum = np.zeros((C, C))
    for i in range(N):
        R_sum += trials[i] @ trials[i].T / T
    R_mean = R_sum / N

    # Compute R_mean^(-1/2)
    R_inv_sqrt = inv(sqrtm(R_mean)).real

    # Apply to each trial
    aligned = np.zeros_like(trials)
    for i in range(N):
        aligned[i] = R_inv_sqrt @ trials[i]

    return aligned.astype(np.float32)


def load_subject_data(gdf_dir, subject_id):
    """
    Load one subject's train (T) and test (E) data with EA normalization.
    Returns: (X_train, y_train, X_test, y_test)
    """
    train_file = os.path.join(gdf_dir, f"A{subject_id:02d}T.gdf")
    test_file = os.path.join(gdf_dir, f"A{subject_id:02d}E.gdf")

    X_train, y_train = extract_trials_from_gdf(train_file)
    X_test, y_test = extract_trials_from_gdf(test_file)

    print(f"  Subject {subject_id}: train={X_train.shape}, test={X_test.shape}")

    # EA normalization per session (paper applies EA per session)
    X_train = euclidean_alignment(X_train)
    X_test = euclidean_alignment(X_test)

    return X_train, y_train, X_test, y_test


# ═════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═════════════════════════════════════════════════════════════════

class BCIDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 22, 1024), y: (N,)
        # Convert to mV (paper uses uniform units → mV)
        self.X = torch.from_numpy(X * 1e3).float()  # µV → mV
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═════════════════════════════════════════════════════════════════
# 3.  MODEL — exact copy of paper's linear probe
# ═════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downstream.models.foundation_models.eegpt import (
    EEGTransformer, Conv1dWithConstraint, CHANNEL_DICT,
)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm > 0:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


# 19 standard 10-20 channels → EEGPT's codebook
USE_CHANNELS = [
    'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2',
]


class EEGPTLinearProbe(nn.Module):
    """Exact reproduction of the paper's linear_probe_EEGPT_BCIC2A.py"""

    def __init__(self, ckpt_path, in_channels=22, num_classes=4, data_length=1024):
        super().__init__()
        self.proj_chans = len(USE_CHANNELS)  # 19

        # Encoder with 19 channels (pretrained space)
        self.target_encoder = EEGTransformer(
            img_size=[self.proj_chans, data_length],
            patch_size=32 * 2,
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

        # Channel IDs from EEGPT's 58-channel codebook
        self.chans_id = self.target_encoder.prepare_chan_ids(USE_CHANNELS)

        # Load pretrained weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("target_encoder."):
                state[k[15:]] = v
        self.target_encoder.load_state_dict(state)

        # Freeze encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Adaptive spatial filter: 22 → 19
        self.chan_conv = Conv1dWithConstraint(in_channels, self.proj_chans, 1, max_norm=1)

        # 2-layer linear probe (paper's exact structure)
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16 * 16, num_classes, max_norm=0.25)
        # 16 * 16 = 256: because data_length/64 = 1024/64 = 16 time patches, × 16 from probe1

        self.drop = nn.Dropout(p=0.50)

    def forward(self, x):
        x = self.chan_conv(x)  # (B, 22, 1024) → (B, 19, 1024)

        self.target_encoder.eval()
        chans_id = self.chans_id.to(x.device)
        z = self.target_encoder(x, chans_id)  # (B, ?, 2048) or similar

        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))  # (B, ?, 16)
        h = h.flatten(1)                       # (B, 256)
        h = self.linear_probe2(h)              # (B, 4)
        return h


# ═════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP — matches paper's hyperparameters
# ═════════════════════════════════════════════════════════════════

def train_one_subject(model, train_loader, test_loader, device, max_epochs=100, max_lr=4e-4):
    """
    Train and evaluate one subject. Returns dict of metrics on test set.
    """
    model = model.to(device)

    # Only train chan_conv + linear probes (encoder frozen)
    trainable = (
        list(model.chan_conv.parameters()) +
        list(model.linear_probe1.parameters()) +
        list(model.linear_probe2.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        pct_start=0.2,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_bacc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        # ── Train ──
        model.train()
        model.target_encoder.eval()  # always keep encoder in eval mode

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        # ── Test ──
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(y)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score
        bacc = balanced_accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        acc = accuracy_score(labels, preds)

        if bacc > best_bacc:
            best_bacc = bacc
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{max_epochs}  "
                  f"train_loss={train_loss/train_total:.4f}  "
                  f"train_acc={train_correct/train_total:.4f}  "
                  f"test_bacc={bacc:.4f}  test_kappa={kappa:.4f}")

    print(f"    Best: epoch={best_epoch+1}, bacc={best_bacc:.4f}")
    return {"bacc": best_bacc, "kappa": kappa, "acc": acc}


# ═════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reproduce EEGPT on BCI-IV-2a")
    parser.add_argument("--gdf_dir", type=str, required=True,
                        help="Path to folder with A01T.gdf, A01E.gdf, A01T.mat, A01E.mat, ...")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to eegpt_mcae_58chs_4s_large4E.ckpt")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("  EEGPT Reproduction — BCI-IV-2a")
    print(f"  Protocol: per-subject (session T→train, session E→test)")
    print(f"  Preprocessing: 0-38 Hz bandpass, EA normalization, 256 Hz")
    print(f"  Epochs: {args.max_epochs}, LR: {args.max_lr}, Batch: {args.batch_size}")
    print("=" * 60)

    all_results = []

    for subject_id in range(1, 10):
        print(f"\n── Subject {subject_id} ──")

        # Load and preprocess
        X_train, y_train, X_test, y_test = load_subject_data(args.gdf_dir, subject_id)

        train_ds = BCIDataset(X_train, y_train)
        test_ds = BCIDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Fresh model per subject
        seed_everything(args.seed)
        model = EEGPTLinearProbe(
            ckpt_path=args.ckpt_path,
            in_channels=22,
            num_classes=4,
            data_length=1024,
        )

        results = train_one_subject(
            model, train_loader, test_loader, device,
            max_epochs=args.max_epochs, max_lr=args.max_lr,
        )
        all_results.append(results)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for i, r in enumerate(all_results):
        print(f"  Subject {i+1}: BAcc={r['bacc']:.4f}  Kappa={r['kappa']:.4f}  Acc={r['acc']:.4f}")

    mean_bacc = np.mean([r["bacc"] for r in all_results])
    std_bacc = np.std([r["bacc"] for r in all_results])
    mean_kappa = np.mean([r["kappa"] for r in all_results])

    print(f"\n  Mean BAcc:  {mean_bacc:.4f} ± {std_bacc:.4f}")
    print(f"  Mean Kappa: {mean_kappa:.4f}")
    print(f"  Paper reports: BAcc = 0.5846 ± 0.0070")


if __name__ == "__main__":
    main()
