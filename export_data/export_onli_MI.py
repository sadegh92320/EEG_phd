
from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml
import h5py
import numpy as np
import os
from scipy.io import loadmat
import re
from export_data.export_data_pretrain import ImportDataPre
from pathlib import Path
import torch
from pymatreader import read_mat
import mne
import mat73
import scipy


class ImportOnlineMI(ImportDataPre):

    def get_participant_number(self, file: Path):
        m = re.match(r"S(\d+)_Session_", file.name)
        participant_nb = int(m.group(1)) if m else file.name.split("_")[0]
        return participant_nb

    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".mat":
            return True

        return False

    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/online_bci_cla.yaml"

    def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """

        trials = []
        if file_path.name == "S8_Session_7.mat":
            return []
        f = loadmat(file_path, struct_as_record=False, squeeze_me=True)

        BCI = f["BCI"]

        for d in BCI.data:
            arr = np.asarray(d, dtype=np.float32)

            if arr.shape[0] != 62 and arr.shape[1] == 62:
                arr = arr.T
            data = self.apply_preprocessing_pretrain(arr)

            trials.append(data)

        return trials


    def import_data_resumable(
        self,
        input_dir,
        output_dir,
        val_ratio=0.2,
        random_seed=92,
        use_float16=False,
        compression="gzip",
        stop_after_participant=None,
    ):
        """
        Resumable version of import_data.

        - Uses the SAME deterministic train/val split every time (same seed).
        - On first run: creates train.h5 and val.h5 from scratch.
        - On resume: detects which participants are already in the h5 files and skips them.
        - stop_after_participant: if set, stop after finishing this participant number.
          Next time you run, it will pick up where it left off.

        Usage:
            # First run — process participants until you stop (e.g. after participant 5)
            data_import.import_data_resumable(
                input_dir="...", output_dir="...",
                stop_after_participant=5
            )

            # Resume — processes remaining participants (automatically skips 1-5)
            data_import.import_data_resumable(
                input_dir="...", output_dir="..."
            )
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_h5_path = output_dir / "train.h5"
        val_h5_path = output_dir / "val.h5"

        rng = np.random.default_rng(random_seed)

        # ── 1. Group files by participant ──
        subject_to_files = {}
        for file in sorted(input_dir.iterdir()):
            if self.condition_file_name(file):
                continue
            participant_nb = self.get_participant_number(file)
            subject_to_files.setdefault(participant_nb, []).append(file)

        if len(subject_to_files) == 0:
            raise FileNotFoundError(f"No valid .mat files found in {input_dir}")

        # ── 2. Deterministic train/val split (always the same with same seed) ──
        participant_ids = sorted(subject_to_files.keys())
        n_participants = len(participant_ids)
        n_val = max(1, int(round(n_participants * val_ratio)))

        perm = rng.permutation(n_participants)
        val_idx = set(perm[:n_val])

        train_participants = []
        val_participants = []

        for i, participant_nb in enumerate(participant_ids):
            if i in val_idx:
                val_participants.append(participant_nb)
            else:
                train_participants.append(participant_nb)

        print(f"Total participants: {n_participants}")
        print(f"Train participants ({len(train_participants)}): {train_participants}")
        print(f"Val participants   ({len(val_participants)}): {val_participants}")

        # ── 3. Check what's already done ──
        done_train = self._get_existing_participants(train_h5_path)
        done_val = self._get_existing_participants(val_h5_path)
        print(f"Already in train.h5: {sorted(done_train) if done_train else 'empty/new'}")
        print(f"Already in val.h5:   {sorted(done_val) if done_val else 'empty/new'}")

        # ── 4. Write train participants ──
        stopped = self._write_split_resumable(
            participant_ids=train_participants,
            subject_to_files=subject_to_files,
            h5_path=train_h5_path,
            split_name="train",
            already_done=done_train,
            use_float16=use_float16,
            compression=compression,
            stop_after_participant=stop_after_participant,
        )

        if stopped:
            print(f"\nStopped after participant {stop_after_participant}. Run again to resume.")
            return

        # ── 5. Write val participants ──
        self._write_split_resumable(
            participant_ids=val_participants,
            subject_to_files=subject_to_files,
            h5_path=val_h5_path,
            split_name="val",
            already_done=done_val,
            use_float16=use_float16,
            compression=compression,
            stop_after_participant=stop_after_participant,
        )

        print(f"\nDone! Saved:")
        print(f"  train: {train_h5_path}")
        print(f"  val:   {val_h5_path}")


    @staticmethod
    def _get_existing_participants(h5_path):
        """Read which participants are already in an h5 file. Returns a set."""
        if not h5_path.exists():
            return set()
        try:
            with h5py.File(h5_path, "r") as f:
                if "participant" in f:
                    return set(np.unique(f["participant"][:]).tolist())
        except Exception as e:
            print(f"  Warning: {h5_path.name} is corrupted ({e}), will recreate it.")
            os.remove(h5_path)
        return set()


    def _write_split_resumable(
        self,
        participant_ids,
        subject_to_files,
        h5_path,
        split_name,
        already_done,
        use_float16=False,
        compression="gzip",
        stop_after_participant=None,
    ):
        """
        Append participants to an h5 file, skipping those already present.
        Returns True if stopped early (hit stop_after_participant).
        """
        dtype_x = np.float16 if use_float16 else np.float32
        x_dtype = "f2" if use_float16 else "f4"

        # Open in append mode (creates if doesn't exist)
        with h5py.File(h5_path, "a") as f:
            # Get or create datasets
            if "x" in f:
                x_ds = f["x"]
                participant_ds = f["participant"]
                count = x_ds.shape[0]
            else:
                x_ds = None
                participant_ds = f.create_dataset(
                    "participant",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="i8",
                )
                count = 0

            for participant_nb in participant_ids:
                # Skip if already done
                if participant_nb in already_done:
                    print(f"[{split_name}] skipping participant {participant_nb} (already in h5)")
                    continue

                files = subject_to_files[participant_nb]
                for file_path in files:
                    print(f"[{split_name}] processing subject={participant_nb}, file={file_path.name}")

                    trials = self._extract_trials(file_path)

                    for data in trials:
                        split_data = self.split_with_hops(
                            data=data,
                            window_s=6,
                            hop_s=5,
                            sampling_rate=128,
                            channels_expected=self.num_chan,
                        )

                        if len(split_data) == 0:
                            continue

                        for eeg, _ in split_data:
                            eeg = eeg.astype(dtype_x, copy=False)

                            if x_ds is None:
                                x_shape = eeg.shape
                                create_kwargs = dict(
                                    shape=(0, *x_shape),
                                    maxshape=(None, *x_shape),
                                    dtype=x_dtype,
                                    chunks=(1, *x_shape),
                                )
                                if compression is not None:
                                    create_kwargs["compression"] = compression
                                x_ds = f.create_dataset("x", **create_kwargs)

                            x_ds.resize(count + 1, axis=0)
                            participant_ds.resize(count + 1, axis=0)

                            x_ds[count] = eeg
                            participant_ds[count] = participant_nb

                            count += 1

                # Flush after each participant so data is safe on disk
                f.flush()
                print(f"[{split_name}] participant {participant_nb} done — {count} total samples so far")

                # Check stop condition
                if stop_after_participant is not None and participant_nb >= stop_after_participant:
                    f.attrs["split"] = split_name
                    f.attrs["n_samples"] = count
                    f.attrs["sampling_rate"] = 128.0
                    f.attrs["n_channels"] = self.num_chan
                    f.attrs["window_s"] = 6.0
                    f.attrs["hop_s"] = 0.5
                    return True

            # Write final attrs
            f.attrs["split"] = split_name
            f.attrs["n_samples"] = count
            f.attrs["sampling_rate"] = 128.0
            f.attrs["n_channels"] = self.num_chan
            f.attrs["window_s"] = 6.0
            f.attrs["hop_s"] = 0.5

        return False



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resumable Online MI import")
    parser.add_argument("--input_dir", type=str, default="/Volumes/Elements/EEG_data/pretraining/Online_MI_BCI_Classification")
    parser.add_argument("--output_dir", type=str, default="MAE_pretraining/data/onlinemi")
    parser.add_argument("--stop_after", type=int, default=None,
                        help="Stop after this participant number (e.g. --stop_after 5). Omit to process all.")
    parser.add_argument("--check", action="store_true",
                        help="Just check what participants are in the h5 files, don't import anything.")
    args = parser.parse_args()

    if args.check:
        # Just inspect existing h5 files
        for name in ["train.h5", "val.h5"]:
            p = Path(args.output_dir) / name
            if p.exists():
                try:
                    with h5py.File(p, "r") as f:
                        if "participant" in f:
                            vals = np.unique(f["participant"][:])
                            print(f"{name}: {f['x'].shape[0]} samples, participants: {vals}")
                        else:
                            print(f"{name}: empty")
                except Exception as e:
                    print(f"{name}: corrupted ({e})")
            else:
                print(f"{name}: does not exist")
    else:
        data_import = ImportOnlineMI(num_chan=62)
        data_import.import_data_resumable(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            stop_after_participant=args.stop_after,
        )
