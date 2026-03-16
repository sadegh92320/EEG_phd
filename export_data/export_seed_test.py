def _extract_trials(self, file_path):
        """
        Read one .mat file, preprocess each trial, return list of arrays (C, T).
        """
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        trials = []

        for key, value in mat.items():
            # skip matlab metadata
            if key.startswith("__"):
                continue

            # adjust this condition depending on actual SEED key names
            
            data = value

            if not isinstance(data, np.ndarray):
                continue

            if data.ndim != 2:
                continue

            if data.shape[0] != 62 and data.shape[1] == 62:
                data = data.T

            if data.shape[0] != 62:
                continue

            data = self.apply_preprocessing_pretrain(data)
            trials.append(data)

        return trials

def get_config(self):
        self.config = "MAE_pretraining/info_dataset/seed2.yaml"s

def get_participant_number(self, file: Path):
        participant_nb = file.stem.split("_")[0]
        return int(participant_nb)

def condition_file(file):
    if file.name.startswith("._"):
        return True

    if file.name.startswith("label"):
        return True

    if file.suffix.lower() != ".mat":
        return True