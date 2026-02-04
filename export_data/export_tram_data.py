import os
import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from export_data.export_data import DataImport
from process_data.mne_constructor import MNEMethods
import yaml


class TramImporter(DataImport):
    def import_data(self):
        """Import data from matlab file and segment it according to experiment"""
        data_eeg = []
        path = self.config["input_data_path"]
        to_check = sorted(os.listdir(path))
        for file in to_check:
            if int(file) == (10) or int(file) == (24) or int(file) == 25 or int(file) == 15 or int(file) == 9 or int(file) == 64:
                continue
            
            
            mat_paths = [
                os.path.join(path,file, f)
                for f in os.listdir(f'{path}/{file}')
                if f.endswith(".mat")
            ]

           
            for mat_path in mat_paths:
                mat_files = []
                with h5py.File(mat_path, "r") as f:
                    mat_files.append(np.array(f["y"]))
            
            y = np.concatenate(mat_files, axis=0)
            
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
            to_return[(str(dict_start["Recording number"]), line)] = dict_start

        return to_return

    def segmentation_based_dict(self, recording_number, seg_dict, array):
        """Segment the data according to the start and end of experiment"""
        seg = [d for key, d in seg_dict.items() if int(recording_number) == int(key[0])]
        
        start_time = [datetime.combine(s["Date"].date(), s["Start"]) for s in seg]
        end_time = [datetime.combine(s["Date"].date(), s["End"]) for s in seg]
        for i in range(len(start_time)):
            if end_time[i] < start_time[i]:
                end_time[i] += timedelta(days=1)
        if seg[0]["City"] == "Antwerp" or seg[0]["City"] == "Utrecht":
            start_time = [s - timedelta(hours=1) for s in start_time]
            end_time = [e - timedelta(hours=1) for e in end_time]
        ovt = [s["OTIV on?"] for s in seg]
        ovt = list(map(lambda x: 1 if x == "Y" else 0, ovt))

        current_start = start_time[0]
        current_end = end_time[0]
        dates = array[:,-6:]
        arrays = []
        to_start, to_end = 0, 0
        start, end = 0, 0
        done = False
        for i in dates:
            date = datetime(year=int(i[0]), month=int(i[1]), day = int(i[2]), hour=int(i[3]), minute=int(i[4]), second=int(i[5]))
            if  date >= current_start + timedelta(minutes=1) and done == False:
                try:
                    start = to_start
                    try:
                        current_start = start_time[start_time.index(current_start) + 1]
                    except:
                        done = True
                except:
                    pass
            if  date >= current_end - timedelta(minutes=1):
                end = to_end
                arrays.append((recording_number,array[start:end,1:33], ovt[end_time.index(current_end)]))
               
                
                if array[start:end,1:33].shape[0] == 0 or array[start:end,1:33].shape[1] == 0:
                        with open("dimension_issue.txt", "a") as f:
                            f.write(recording_number + "\n")
                try:
                    current_end = end_time[end_time.index(current_end) + 1]
                    print(current_end)
                except:
                    break
            to_start += 1
            to_end += 1
        
        return arrays
def segmentation_based_dict(recording_number, seg_dict, array):
        """Segment the data according to the start and end of experiment"""
        seg = [d for key, d in seg_dict.items() if int(recording_number) == int(key[0])]
        print(seg)
        print(recording_number)
        start_time = [s["Start"] for s in seg]
        end_time = [s["End"] for s in seg]
        ovt = [s["OTIV on?"] for s in seg]
        ovt = list(map(lambda x: 1 if x == "Y" else 0, ovt))
        current_start = start_time[0]
        current_end = end_time[0]
        dates = array[:,-6:]
        arrays = []
        to_start, to_end = 0, 0
        start, end = 0, 0
        for i in dates:
            date = datetime.time(hour=int(i[3]), minute=int(i[4]))
            if  date >= current_start + timedelta(minutes=1):
                try:
                    start = to_start
                    current_start = start_time[start_time.index(current_start) + 1]
                except:
                    pass
            if  date >= current_end - timedelta(minutes=1):
                try:
                    end = to_end

                    arrays.append((array[start:end,1:33], ovt[end_time.index(current_start)]))
                    
                    current_end = end_time[end_time.index(current_start) + 1]
                except:
                    pass
            to_start += 1
            to_end += 1
        if seg["OTIV on?"] == "Y":
            label = 1
        elif seg["OTIV on?"] == "N":
            label = 0
        to_return = [(a,label) for a in arrays]
        return []

def open_excel(csv_path):
        """Open excel file associated with experiment"""
        df = pd.read_excel(csv_path)
        to_return = {}
        for line in range(len(df)):
            dict_start = df.iloc[line].to_dict()
            to_return[(str(dict_start["Recording number"]), line)] = dict_start

        return to_return

if __name__ == "__main__":
    with open("setting.yaml") as f:
        config = yaml.safe_load(f)
    data = TramImporter(config=config, mne_process=MNEMethods(config=config))
    data().save_data()
    #remove_artifacts().save_data()
    seg_dict = (open_excel("/Users/sadeghemami/TramTiming.xlsx"))
    recording_number = "65"
    seg = ([d for key, d in seg_dict.items() if int(recording_number) == int(key[0])])
    start_time = [s["Start"] for s in seg]
    end_time = [s["End"] for s in seg]
    otv = [s["OTIV on?"] for s in seg]
   
    
    