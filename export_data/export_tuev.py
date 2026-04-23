"""
TUEV v2.0.1 — TUH EEG Events downstream exporter  →  train.h5 / val.h5

Source:  Temple University Hospital EEG Event corpus v2.0.1
         290 train subjects, 80 eval subjects (pre-split by TUH).
         21 EEG channels (10-20 referential montage), 250 Hz native.
         Per-channel annotations in .lab files (HTK-like format):
             start_1e5  end_1e5  label_string   (times in 1e-5 s units)
         6 event classes:
             0 = spsw   (spike-and-slow-wave)
             1 = gped   (generalized periodic epileptiform discharges)
             2 = pled   (periodic lateralized epileptiform discharges)
             3 = eyem   (eye movement)
             4 = artf   (artifact)
             5 = bckg   (background / normal)

Label strategy:
    Each .lab file gives per-channel, per-second annotations.
    We take a MAJORITY VOTE across all channels for each 1-second window
    to produce a single label per window. Consecutive seconds with the
    same majority-vote label are merged into contiguous EVENT SEGMENTS.

Segment strategy (following CBraMod convention):
    Target segment length: 5 seconds (1250 samples at 250 Hz).
    - Events >= 5s: split into non-overlapping 5s chunks (remainder discarded
      only if < 1s; otherwise zero-padded to 5s).
    - Events < 5s but >= 1s: zero-padded to 5s.
    - Events < 1s: discarded.

Split:
    Train and eval participants are pre-defined by TUH.
    We write train.h5 from edf/train/ and val.h5 from edf/eval/.
    No further splitting is needed.

Data format:
    Each sample is a 5-second segment: (C, T) = (21, 1250).
"""

import numpy as np
import mne
from mne.filter import notch_filter, filter_data
import h5py
from pathlib import Path
from collections import Counter


# ── Channel selection ──
# Keep 21 standard EEG channels, drop non-EEG (ROC, LOC, EKG1, T1, T2, PHOTIC)
EEG_CHANNELS_RAW = [
    "EEG FP1-REF", "EEG FP2-REF",
    "EEG F3-REF", "EEG F4-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG FZ-REF",
    "EEG C3-REF", "EEG C4-REF", "EEG CZ-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
    "EEG P3-REF", "EEG P4-REF", "EEG PZ-REF",
    "EEG O1-REF", "EEG O2-REF",
    "EEG A1-REF", "EEG A2-REF",
]

# Standard 10-20 names for downstream compatibility
STANDARD_NAMES = [
    "Fp1", "Fp2",
    "F3", "F4", "F7", "F8",
    "Fz",
    "C3", "C4", "Cz",
    "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz",
    "O1", "O2",
    "A1", "A2",
]

RENAME_MAP = dict(zip(EEG_CHANNELS_RAW, STANDARD_NAMES))

NUM_CHANNELS = len(EEG_CHANNELS_RAW)  # 21
NATIVE_SFREQ = 250.0
SEGMENT_SEC = 5.0       # 5-second target segment length
MIN_EVENT_SEC = 1.0     # discard events shorter than 1 second

# Label mapping
LABEL_STR_TO_INT = {
    "spsw": 0,
    "gped": 1,
    "pled": 2,
    "eyem": 3,
    "artf": 4,
    "bckg": 5,
}
NUM_CLASSES = 6
CLASS_NAMES = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]


def _parse_lab_file(lab_path):
    """
    Parse a .lab file (HTK format).
    Returns list of (start_sec, end_sec, label_str).

    Format: start<tab>end<tab>label_string
    Times are in units of 1e-5 seconds (divide by 1e5 to get seconds).
    Verified by cross-referencing .lab entry 1080000 → 10.8s with .rec annotation.
    """
    annotations = []
    with open(lab_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start_raw = int(parts[0])
            end_raw = int(parts[1])
            label = parts[2].lower()

            if label == "(null)" or label not in LABEL_STR_TO_INT:
                continue

            start_sec = start_raw / 1e5
            end_sec = end_raw / 1e5
            annotations.append((start_sec, end_sec, label))

    return annotations


def _build_label_array(lab_files, duration_sec):
    """
    Build a per-second label array from all channel .lab files.
    Uses majority vote across channels for each 1-second window.

    Returns: np.array of shape (n_windows,) with integer labels.
    """
    n_windows = int(duration_sec)

    # Collect votes: for each second, count label occurrences across channels
    votes = [Counter() for _ in range(n_windows)]

    for lab_path in lab_files:
        annotations = _parse_lab_file(lab_path)
        for start_sec, end_sec, label in annotations:
            win_start = int(start_sec)
            win_end = int(end_sec)
            for w in range(win_start, min(win_end, n_windows)):
                votes[w][label] += 1

    # Majority vote per window
    labels = np.full(n_windows, LABEL_STR_TO_INT["bckg"], dtype=np.int64)
    for w in range(n_windows):
        if votes[w]:
            winner = votes[w].most_common(1)[0][0]
            labels[w] = LABEL_STR_TO_INT[winner]

    return labels


def _merge_consecutive_labels(labels):
    """
    Merge consecutive seconds with the same label into event segments.

    Returns: list of (start_sec, end_sec, label_int)
    """
    if len(labels) == 0:
        return []

    events = []
    start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            events.append((start, i, int(current_label)))
            start = i
            current_label = labels[i]

    # Last event
    events.append((start, len(labels), int(current_label)))

    return events


def _segment_event(data, event_start_sec, event_end_sec, sfreq, seg_samples):
    """
    Extract fixed-length segments from one event.

    - Events >= SEGMENT_SEC: split into non-overlapping chunks.
      Remainder >= MIN_EVENT_SEC is zero-padded; otherwise discarded.
    - Events < SEGMENT_SEC but >= MIN_EVENT_SEC: zero-padded to seg_samples.
    - Events < MIN_EVENT_SEC: discarded (returns empty list).

    Returns: list of np.array (C, seg_samples) float32
    """
    C = data.shape[0]
    start_samp = int(event_start_sec * sfreq)
    end_samp = int(event_end_sec * sfreq)
    end_samp = min(end_samp, data.shape[1])

    event_samples = end_samp - start_samp
    event_sec = event_samples / sfreq

    if event_sec < MIN_EVENT_SEC:
        return []

    segments = []

    if event_samples >= seg_samples:
        # Split into non-overlapping chunks
        pos = start_samp
        while pos + seg_samples <= end_samp:
            seg = data[:, pos:pos + seg_samples].astype(np.float32)
            segments.append(seg)
            pos += seg_samples

        # Handle remainder
        remainder = end_samp - pos
        if remainder >= int(MIN_EVENT_SEC * sfreq):
            seg = np.zeros((C, seg_samples), dtype=np.float32)
            seg[:, :remainder] = data[:, pos:end_samp].astype(np.float32)
            segments.append(seg)
    else:
        # Short event: zero-pad to target length
        seg = np.zeros((C, seg_samples), dtype=np.float32)
        seg[:, :event_samples] = data[:, start_samp:end_samp].astype(np.float32)
        segments.append(seg)

    return segments


def _process_recording(folder_path):
    """
    Process one recording folder.
    Returns: list of (eeg_array, label_int) where eeg_array is (C, 1250).
    """
    folder = Path(folder_path)

    # Find .edf file
    edf_files = list(folder.glob("*.edf"))
    if not edf_files:
        print(f"  No .edf in {folder.name}, skipping")
        return []
    edf_path = edf_files[0]

    # Find .lab files
    lab_files = sorted(folder.glob("*.lab"))
    if not lab_files:
        print(f"  No .lab files in {folder.name}, skipping")
        return []

    # Load EEG
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    except Exception as e:
        print(f"  ERROR reading {edf_path.name}: {e}")
        return []

    sfreq = raw.info["sfreq"]

    # Pick EEG channels
    available = [ch for ch in EEG_CHANNELS_RAW if ch in raw.ch_names]
    if len(available) < NUM_CHANNELS:
        print(f"  WARNING: only {len(available)}/{NUM_CHANNELS} EEG channels in {folder.name}")
        if len(available) == 0:
            return []

    raw.pick(available)

    # Rename to standard 10-20
    rename = {ch: RENAME_MAP[ch] for ch in available}
    raw.rename_channels(rename)

    # ── Preprocessing: notch 50/60 Hz + bandpass 0.1–64 Hz (IIR) ──
    raw.notch_filter(freqs=50, method="iir", verbose="ERROR")
    raw.notch_filter(freqs=60, method="iir", verbose="ERROR")
    raw.filter(l_freq=0.1, h_freq=64.0, method="iir", verbose="ERROR")

    data = raw.get_data()  # (C, T)
    C, T_total = data.shape
    duration_sec = T_total / sfreq

    # Build per-second labels from .lab files
    labels = _build_label_array(lab_files, duration_sec)

    # Merge consecutive same-label seconds into event segments
    events = _merge_consecutive_labels(labels)

    # Extract fixed-length segments from each event
    seg_samples = int(SEGMENT_SEC * sfreq)  # 1250
    segments = []

    for ev_start, ev_end, ev_label in events:
        segs = _segment_event(data, ev_start, ev_end, sfreq, seg_samples)
        for seg in segs:
            segments.append((seg, ev_label))

    return segments


def _downsample_majority(rows, max_ratio=5.0, seed=42):
    """
    Downsample the majority class (bckg) so it has at most `max_ratio` times
    as many samples as the largest minority class.

    This keeps the dataset realistic (bckg is still the biggest class) while
    preventing 98%+ imbalance that makes balanced-accuracy metrics meaningless.

    Example: if the largest minority class has 300 samples and max_ratio=5,
    bckg is capped at 1500 samples.
    """
    from collections import defaultdict
    rng = np.random.default_rng(seed)

    by_label = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)

    bckg_label = LABEL_STR_TO_INT["bckg"]  # 5
    minority_counts = [len(v) for k, v in by_label.items() if k != bckg_label and len(v) > 0]

    if not minority_counts or bckg_label not in by_label:
        return rows  # nothing to downsample

    max_minority = max(minority_counts)
    cap = int(max_ratio * max_minority)
    bckg_rows = by_label[bckg_label]

    if len(bckg_rows) > cap:
        idx = rng.choice(len(bckg_rows), size=cap, replace=False)
        by_label[bckg_label] = [bckg_rows[i] for i in sorted(idx)]
        print(f"  Downsampled bckg: {len(bckg_rows)} → {cap} "
              f"(cap = {max_ratio}× largest minority class = {max_minority})")

    out = []
    for label in sorted(by_label.keys()):
        out.extend(by_label[label])
    return out


def _write_h5(rows, h5_path, split):
    """Write segments to HDF5."""
    if len(rows) == 0:
        print(f"  [{split}] nothing to write")
        return

    C, T = rows[0][0].shape
    n = len(rows)

    with h5py.File(h5_path, "w") as f:
        x_ds = f.create_dataset("x", shape=(n, C, T), dtype="f4")
        y_ds = f.create_dataset("y", shape=(n,), dtype="i8")
        part_ds = f.create_dataset("participant", shape=(n,), dtype="i8")

        for i, (eeg, label, part_id) in enumerate(rows):
            x_ds[i] = eeg
            y_ds[i] = label
            part_ds[i] = part_id

        f.attrs["split"] = split
        f.attrs["n_samples"] = n
        f.attrs["sampling_rate"] = NATIVE_SFREQ
        f.attrs["n_channels"] = C
        f.attrs["n_classes"] = NUM_CLASSES
        f.attrs["task"] = "classification"
        f.attrs["classes"] = ",".join(CLASS_NAMES)

    print(f"  [{split}] wrote {n} rows → {h5_path}")


def import_tuev(
    input_dir,
    output_dir="downstream/data/tuev",
):
    """
    Main entry point.

    Parameters
    ----------
    input_dir : str
        Path to TUEV v2.0.1 edf/ folder containing train/ and eval/.
    output_dir : str
        Output directory for train.h5 / val.h5.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = input_dir / "train"
    eval_dir = input_dir / "eval"

    if not train_dir.exists() or not eval_dir.exists():
        raise FileNotFoundError(
            f"Expected train/ and eval/ inside {input_dir}"
        )

    train_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    eval_folders = sorted([d for d in eval_dir.iterdir() if d.is_dir()])

    print(f"TUEV v2.0.1: {len(train_folders)} train subjects, {len(eval_folders)} eval subjects")

    # ── Process train ──
    print(f"\n{'='*60}")
    print("Processing TRAIN split...")
    print(f"{'='*60}")

    train_rows = []
    for part_idx, folder in enumerate(train_folders):
        print(f"\n[{part_idx+1}/{len(train_folders)}] {folder.name}")
        segments = _process_recording(folder)
        for seg, label in segments:
            train_rows.append((seg, label, part_idx))
        print(f"  → {len(segments)} segments")

    # ── Process eval ──
    print(f"\n{'='*60}")
    print("Processing EVAL split...")
    print(f"{'='*60}")

    eval_rows = []
    # Offset participant IDs so they don't overlap with train
    part_offset = len(train_folders)
    for part_idx, folder in enumerate(eval_folders):
        print(f"\n[{part_idx+1}/{len(eval_folders)}] {folder.name}")
        segments = _process_recording(folder)
        for seg, label in segments:
            eval_rows.append((seg, label, part_offset + part_idx))
        print(f"  → {len(segments)} segments")

    # ── Downsample bckg to reduce extreme class imbalance ──
    # Cap bckg at 5× the largest minority class (keeps bckg dominant but usable)
    print(f"\n{'='*60}")
    print("Downsampling majority class (bckg)...")
    print(f"{'='*60}")
    train_rows = _downsample_majority(train_rows, max_ratio=5.0, seed=42)
    eval_rows = _downsample_majority(eval_rows, max_ratio=5.0, seed=42)

    # ── Write HDF5 ──
    _write_h5(train_rows, output_dir / "train.h5", "train")
    _write_h5(eval_rows, output_dir / "val.h5", "val")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split_name, rows in [("train", train_rows), ("val", eval_rows)]:
        if not rows:
            continue
        labels = np.array([r[1] for r in rows])
        n_parts = len(set(r[2] for r in rows))
        print(f"\n  {split_name}: {len(rows)} segments, {n_parts} participants")
        for c in range(NUM_CLASSES):
            count = np.sum(labels == c)
            if count > 0:
                print(f"    {CLASS_NAMES[c]:6s}: {count:>8d}  ({100*count/len(labels):5.1f}%)")


if __name__ == "__main__":
    import_tuev(
        input_dir="/Users/sadeghemami/v2.0.1/edf",
        output_dir="downstream/data/tuev",
    )

    # Quick verification
    for split in ["train", "val"]:
        path = f"downstream/data/tuev/{split}.h5"
        with h5py.File(path, "r") as f:
            print(f"\n=== {split}.h5 ===")
            print(f"  x:           {f['x'].shape}")
            print(f"  y:           {f['y'].shape}   unique: {np.unique(f['y'][:])}")
            print(f"  participant: unique count: {len(np.unique(f['participant'][:]))}")
