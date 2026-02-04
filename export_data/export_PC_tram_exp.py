import os
import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml


class PCWorkloadImport(DataImport):
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = self.config["input_data_path"]
        to_check = sorted(os.listdir(path))
        for file in to_check:
            mat_path = os.path.join(path, file, f"{file}.mat")
            with h5py.File(mat_path, "r") as f:
                y = np.array(f["y"])
                dict_seg = self.open_excel(self.config["csv"])
                exp_eeg = self.segmentation_based_dict(str(file), dict_seg, y)
                data_eeg.extend(exp_eeg)
        return data_eeg
        
    def open_excel(self, csv_path):
        """Open excel file associated with experiment"""
        df = pd.read_excel(csv_path)
        to_return = {}
        for line in range(len(df)):
            dict_start = df.iloc[line].to_dict()
            to_return[str(dict_start["pnr"])] = dict_start

        return to_return

    def segmentation_based_dict(self, part_number, seg_dict, array):
        """Segment the data according to the start and end of experiment"""
        seg = seg_dict[str(part_number)]
        dt = datetime.strptime(seg["start_time"], "%y%m%d_%H%M%S")
        dates = array[:,-6:]
        to_start = 0
        for i in dates:
            date = datetime(year=int(i[0]), month=int(i[1]), day = int(i[2]), hour=int(i[3]), minute=int(i[4]), second=int(i[5]))
            if  date >= dt:
                break
            to_start += 1
        date_restart = array[to_start:, :]
        second_start = date_restart[0,0]
        date_restart[:,0] = date_restart[:,0] - second_start
        timestamp = date_restart[:,0]
        
        mask_low = ((timestamp  >= seg["low_workload_start"]) & (timestamp  <= seg["low_workload_end"]))
        mask_high = ((timestamp  >= seg["high_workload_start"]) & (timestamp  <= seg["high_workoad_end"]))
        low_workload = date_restart[mask_low]
        high_workload = date_restart[mask_high]
        low_workload = low_workload[:, 1:33]
        high_workload = high_workload[:, 1:33]
        print(low_workload.shape)

        return [(part_number,low_workload, 0), (part_number,high_workload, 1)]
    

if __name__ == "__main__":
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    data = PCWorkloadImport(config=config, mne_process=MNEMethods(config=config))
    data().remove_artifacts().partition_data().save_data()
    
    