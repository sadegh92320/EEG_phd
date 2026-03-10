import os
import re
import gc
import numpy as np
from scipy.io import loadmat
from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods

class ImportOnlineMI(DataImport):

    def get_config(self):
        self.config = r"MAE_pretraining\info_dataset\online_bci_cla.yaml"

    def iter_data(self):
        path = r"D:\EEG_data\pretraining\Online_MI_BCI_Classification"

        for file in sorted(os.listdir(path)):
            if not file.endswith(".mat") or file.startswith("._"):
                continue

            mat_path = os.path.join(path, file)

            m = re.match(r"S(\d+)_Session_", file)
            participant_nb = int(m.group(1)) if m else file.split("_")[0]

            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            BCI = mat["BCI"]

            for i, d in enumerate(BCI.data):
                arr = np.asarray(d, dtype=np.float32)

                if arr.ndim == 2 and arr.shape[0] != 62 and arr.shape[1] == 62:
                    arr = arr.T

                yield participant_nb, i, arr

            del mat, BCI
            gc.collect()

    def preprocessing_and_save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        sample_idx = 0

        for p, trial_idx, d in self.iter_data():
            processed = self.apply_preprocessing_array(data=d)

            split_data = self.split_with_hops(
                data=processed,
                participant=p,
                window_s=6,
                hop_s=2.5,
                sampling_rate=128,
                channels_expected=62
            )

            if x_participant in [45,55,15,60,8,22]:
                for x_participant, x_data in split_data:
                    out_path = os.path.join(out_dir, "val",f"{x_participant}_{sample_idx}.npz")
                    np.savez(out_path, x=x_data)
                    sample_idx += 1

                del d, processed, split_data
                gc.collect()
            else:
                for x_participant, x_data in split_data:
                    out_path = os.path.join(out_dir, "train",f"{x_participant}_{sample_idx}.npz")
                    np.savez(out_path, x=x_data)
                    sample_idx += 1

                del d, processed, split_data
                gc.collect()

        return self