"""
https://openneuro.org/datasets/ds003775/versions/1.0.0
"""
import os
import shutil

import mne
import numpy as np
import openneuro as on

import neurokit2 as nk

# Download cleaned data (takes some time)
on.download(
    dataset="ds003775",
    target_dir="eeg/raw",
    include="sub-*",
    exclude="derivatives/cleaned_data",
)

# Convert to MNE
path = "eeg/raw/"
for sub in os.listdir(path):
    if "sub" not in sub:
        continue
    print(f"Participant: {sub}")
    file = [f for f in os.listdir(path + sub + "/ses-t1/eeg/") if ".edf" in f][0]
    raw = mne.io.read_raw_edf(path + sub + "/ses-t1/eeg/" + file, preload=True, verbose=False)
    raw = raw.set_montage("biosemi64")

    # Clean
    raw = raw.notch_filter(freqs=np.arange(50, 501, 50), verbose=False)
    raw.info["bads"], _ = nk.eeg_badchannels(
        raw, bad_threshold=0.33, distance_threshold=0.99, show=False
    )
    print("Bad channels: " + str(len(raw.info['bads'])))
    raw = raw.interpolate_bads()

    raw.save("eeg/" + sub + "_raw.fif", overwrite=True)

print("FINISHED.")

# Clean-up
shutil.rmtree("eeg/raw/")
