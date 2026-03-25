from export_data.export_data_pretrain import ImportDataPre
from pathlib import Path
import torch
import mne




class ImportPhysio(ImportDataPre):

    def get_participant_number(self, file: Path):
        participant_nb = file.name[1:4]
        participant_nb = int(participant_nb)
        return participant_nb
    
    def condition_file_name(self, file):
        if file.name.startswith("._"):
            return True

        if file.suffix.lower() != ".edf":
            return True
        
        return False
        
    def get_config(self):
        self.config = "MAE_pretraining/info_dataset/physionet.yaml"

    def _extract_trials(self, file_path):
        
        
        trials = []

       
        data = mne.io.read_raw_edf(file_path)
            
        data = (data.get_data())
                    
        if data.shape[0] != 64 and data.shape[1] == 64:
            data = data.T
        data = self.apply_preprocessing_pretrain(data)

        trials.append(data)

        return trials

if __name__ == "__main__":
    data_import = ImportPhysio(num_chan=64)
    data_import.import_data(input_dir = "/Volumes/Elements/EEG_data/pretraining/physioMI", output_dir= "MAE_pretraining/data/physio")
