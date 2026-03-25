from scipy.io import loadmat

if __name__ == "__main__":
    path = "/Volumes/Elements/EEG_data/pretraining/SEED/label.mat"
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    print(mat)