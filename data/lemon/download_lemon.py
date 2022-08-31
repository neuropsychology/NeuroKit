# -*- coding: utf-8 -*-
"""Script for formatting the LEMON EEG dataset

https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/

Steps:
    1. Download the ZIP database from https://physionet.org/content/nstdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-noise-stress-test-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    pycrostates package by Mathieu Scheltienne and Victor Férat
"""
import os

import mne
import numpy as np
import pooch

# Path of the database
path = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/"

# Create a registry with the file names
files = {
    f"sub-01{i:04d}_{j}.{k}": None
    for i in range(2, 319)
    for j in ["EC", "EO"]
    for k in ["fdt", "set"]
}

# Create fetcher
fetcher = pooch.create(
    path="lemon/",
    base_url=path,
    registry=files,
)

# Download the files
for sub in files.keys():
    try:
        _ = fetcher.fetch(sub)
    except:
        pass

print("Finished downloading!")

# Preprocessing

# fmt: off
standard_channels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
        "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz",
        "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7",
        "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7",
        "FC3", "FC4", "FT8", "C5", "C1", "C2", "C6", "TP7",
        "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6",
        "PO7", "PO3", "POz", "PO4", "PO8",
    ]
# fmt: on

for sub in os.listdir("lemon/"):
    if sub.endswith("fdt") is True or sub.endswith("fif") or "sub" not in sub:
        continue
    raw = mne.io.read_raw_eeglab("lemon/" + sub, preload=True)

    missing_channels = list(set(standard_channels) - set(raw.info["ch_names"]))

    if len(missing_channels) != 0:
        # add the missing channels as bads (array of zeros)
        missing_data = np.zeros((len(missing_channels), raw.n_times))
        data = np.vstack([raw.get_data(), missing_data])
        ch_names = raw.info["ch_names"] + missing_channels
        ch_types = raw.get_channel_types() + ["eeg"] * len(missing_channels)
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=raw.info["sfreq"])
        raw = mne.io.RawArray(data=data, info=info)
        raw.info["bads"].extend(missing_channels)

    raw = raw.add_reference_channels("FCz")
    raw = raw.reorder_channels(standard_channels)
    raw = raw.set_montage("standard_1005")
    raw = raw.interpolate_bads()
    raw = raw.set_eeg_reference("average").apply_proj()

    raw.save("lemon/" + sub.replace(".set", "") + "_raw.fif", overwrite=True)


# Clean-up
for sub in os.listdir("lemon/"):
    if sub.endswith("fif"):
        continue
    os.remove(f"lemon/{sub}")

print("FINISHED.")
