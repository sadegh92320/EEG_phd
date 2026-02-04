import os
import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

def loop_trough_file(path):
    to_check = sorted(os.listdir(path))

    for file in to_check:
        mat_path = os.path.join(path, file, f"{file}.mat")
        with h5py.File(mat_path, "r") as f:
            y = np.array(f["y"])
            
        
def open_csv(csv_path):
    df = pd.read_excel(csv_path)
    to_return = {}
    for line in range(len(df)):
        dict_start = df.iloc[line].to_dict()
        to_return[str(dict_start["pnr"])] = dict_start

    return to_return

def segmentation_based_dict(part_number, seg_dict, array):
    seg = seg_dict[str(part_number)]
    
    
    dt = datetime.strptime(seg["start_time"], "%y%m%d_%H%M%S")
    print(dt)
    dates = array[:,-6:]
    print(dates[-1,:])
    to_start = 0
    for i in dates:
        date = datetime(year=int(i[0]), month=int(i[1]), day = int(i[2]), hour=int(i[3]), minute=int(i[4]), second=int(i[5]))
        if  date >= dt:
            break
        to_start += 1
    print(to_start)
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
    
    


        
    





if __name__ == "__main__":
    #loop_trough_file("/Volumes/Elements/untitled folder/PC")
    #dict_seg = open_csv("/Users/sadeghemami/OpenMATBTimings.xlsx")
    ##array = loop_trough_file("/Volumes/Elements/untitled folder/PC")
    #segmentation_based_dict("12", dict_seg, array)
    print(list([1,2,3,4]))


