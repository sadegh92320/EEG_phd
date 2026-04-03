"""
Re-export the Large Motor Imagery (LMI) dataset from raw .mat files.

Bug fix: The original export_large_MI.py had a bug where unprocessed data
was appended instead of the preprocessed version. Additionally, the raw data
at 200 Hz was windowed with sampling_rate=128, giving incorrect window sizes.

Usage (Colab):
    1. Mount your drive and set INPUT_DIR / OUTPUT_DIR below
    2. Run this script: !python scripts/reexport_lmi.py

This will:
    - Read each .mat file from INPUT_DIR
    - Apply preprocessing: notch 50/60 Hz, bandpass 0.1–64 Hz, resample to 128 Hz
    - Window into 6-second segments with 2.5-second hops
    - Save to OUTPUT_DIR/train.h5 and OUTPUT_DIR/val.h5
"""

import sys
import os

# ── CONFIGURE THESE PATHS ──
INPUT_DIR  = "/content/drive/MyDrive/eeg_data/Experiment_CLA"   # raw .mat files
OUTPUT_DIR = "/content/drive/MyDrive/eeg_data/lmi"              # output h5
# ────────────────────────────

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from export_data.export_large_MI import ImportLargeMI

if __name__ == "__main__":
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Verify input dir exists
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        print("Please update INPUT_DIR in this script.")
        sys.exit(1)

    # Count .mat files
    mat_files = [f for f in os.listdir(INPUT_DIR)
                 if f.endswith(".mat") and not f.startswith("._")]
    print(f"Found {len(mat_files)} .mat files")

    if len(mat_files) == 0:
        print("ERROR: No .mat files found. Check the path.")
        sys.exit(1)

    # Remove old h5 files to start fresh
    for name in ["train.h5", "val.h5"]:
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed old {p}")

    data_import = ImportLargeMI(num_chan=21)
    data_import.import_data(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Verify output
    import h5py
    for name in ["train.h5", "val.h5"]:
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                n = f["x"].shape
                print(f"{name}: shape={n}, dtype={f['x'].dtype}")
        else:
            print(f"WARNING: {name} not created")

    print("\nDone! LMI dataset re-exported with correct preprocessing.")
