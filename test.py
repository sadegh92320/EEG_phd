import mne
from braindecode.datasets import MOABBDataset

dataset = MOABBDataset(dataset_name="Schirrmeister2017")
raw = dataset.datasets[0].raw
print(raw.ch_names)
