import mne
from mne.preprocessing import ICA


class MNEMethods:
    def __init__(self, config):
        self.config = config
    def create_mne_object(self, eeg,description):
        ch_types = ["eeg"] * self.config["number_channels"]
        info = mne.create_info(self.config["channel_list"], ch_types=ch_types, sfreq=self.config["freq"])
        
        info["description"] = description 
        simulated_raw = mne.io.RawArray((eeg*10e-6).transpose(), info)
        return simulated_raw
    
    def clean_raw_with_ica(self, X):
        raw = self.create_mne_object(X, "dataset")
        raw = raw.copy().load_data()
        raw.filter(1., 40., fir_design='firwin')

        ica = ICA(n_components=raw.info['nchan'], method='fastica', random_state=97)
        ica.fit(raw)

     
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=self.config["eog_proxies"], threshold=3.0)
        ica.exclude = eog_inds

        
        raw_clean = ica.apply(raw.copy())
        X_clean = raw_clean.get_data().T
        return X_clean
    
    def compute_band(self, X):
        bands = {"delta": [1,4], "theta": [4,8], "alpha": [8,12],"beta": [12,30]}
        bands_data = {}
        raw = self.create_mne_object(X, "dataset")
        raw = raw.copy().load_data()
        for band, value in bands.items():
            raw_band = raw.copy()
            bands_data[band] = (raw_band.filter(value[0], value[1], fir_design='firwin')).get_data()
        return bands_data

    def compute_PSD(self, X):
        pass

    def compute_CSP(self, X):
        pass

    def compute_PCA(self, X):
        pass

    def compute_ICA(self,X):
        pass

    def compute_cov_mat(self, X):
        X = X - X.mean(dim = 1, keep_dim = True)
        cov = (X @ X.T)/(X.shape[1] - 1 )
        return cov