import mne


raw = mne.io.read_raw_fif(
    mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif",
    preload=True,
    verbose=False,
)
raw = raw.pick(["eeg", "eog", "stim"], verbose=False)
raw = raw.crop(0, 60)
raw = raw.resample(200)

raw.save("eeg_1min_200hz_raw.fif", overwrite=True)
