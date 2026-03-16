import mne


if __name__ == "__main__":
    file = "/Users/sadeghemami/Downloads/BCICIV_2a_gdf/A01T.gdf"
    raw = mne.io.read_raw_gdf(file, preload=True)
    print(raw)
    events, e = mne.events_from_annotations(raw, verbose="ERROR")

    # Define labels based on your provided dictionary
    event_id = {
        'left': 7,    # Code 769
        'right': 8,   # Code 770
        'foot': 9,    # Code 771
        'tongue': 10  # Code 772
    }

    # The imagery window is 4 seconds long [cite: 43]
    # tmin=0.0 starts exactly at the cue (769-772)
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=0.0, tmax=4.0, 
                        baseline=None, preload=True)

    # Remove EOG channels (23, 24, 25) before classification [cite: 61, 82]
    epochs.pick_types(eeg=True) 

    # X contains your training data: (trials, 22 channels, 1001 samples)
    X = epochs.get_data()
    y = epochs.events[:, -1] # These will be your labels 7, 8, 9, 10
    print(X.shape)
    
