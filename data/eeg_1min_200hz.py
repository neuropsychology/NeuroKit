import pickle

import mne

raw = mne.io.read_raw_fif(
    mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif",
    preload=True,
    verbose=False,
)
raw = raw.pick(["eeg", "eog", "stim"], verbose=False)
raw = raw.crop(0, 60)
raw = raw.resample(200)

# raw.ch_names

# raw.info["sfreq"]

# Store data (serialize)
with open("eeg_1min_200hz.pickle", "wb") as handle:
    pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
