"""
https://openneuro.org/datasets/ds003685/
"""
import os
import re
import shutil

import mne
import numpy as np
import openneuro as on

import neurokit2 as nk

# Download cleaned data (takes some time)
on.download(
    dataset="ds003685",
    target_dir="eeg/raw",
    include="sub-*/ses-session1/*eyes*",
)

# Convert to MNE
path = "eeg/raw/"
for sub in os.listdir(path):
    if "sub" not in sub or "sub-60" in sub:
        continue
    print(f"Participant: {sub}")
    newpath = path + sub + "/ses-session1/eeg/"

    # The header file is broken as the name in it is incorrect
    # -------------------------------------------------------------------------
    for file in [f for f in os.listdir(newpath) if ".vmrk" in f]:
        with open(newpath + file, "r+") as f:
            text = f.read()  # read everything in the file
            pattern = re.search("DataFile=.*\\n", text).group(0)
            text = text.replace(pattern, pattern.replace(" ", ""))
        with open(newpath + file, "r+") as f:
            f.write(text)

    for file in [f for f in os.listdir(newpath) if ".vhdr" in f]:

        with open(newpath + file, "r+") as f:
            text = f.read()  # read everything in the file
            pattern = re.search("DataFile=.*\\n", text).group(0)
            text = text.replace(pattern, pattern.replace(" ", ""))
            pattern = re.search("MarkerFile=.*\\n", text).group(0)
            text = text.replace(pattern, pattern.replace(" ", ""))
        with open(newpath + file, "r+") as f:
            f.write(text)
        # -------------------------------------------------------------------------

        raw = mne.io.read_raw_brainvision(newpath + file, preload=True, verbose=False)
        raw = raw.set_eeg_reference("average")
        # raw = raw.set_montage("biosemi64")
        if "eyesopen" in file:
            raw.save("eeg/" + sub + "_eyesopen_raw.fif", overwrite=True)
        else:
            raw.save("eeg/" + sub + "_eyesclosed_raw.fif", overwrite=True)

print("FINISHED.")
# Clean-up
shutil.rmtree("eeg/raw/")
