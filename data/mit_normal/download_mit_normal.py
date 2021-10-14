# -*- coding: utf-8 -*-
"""Script for formatting the MIT-Normal Sinus Rhythm Database

Steps:
    1. Download the ZIP database from https://physionet.org/content/nsrdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-normal-sinus-rhythm-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import pandas as pd
import numpy as np
import wfdb
import os

data_files = ["mit-bih-normal-sinus-rhythm-database-1.0.0/" + file for file in os.listdir("mit-bih-normal-sinus-rhythm-database-1.0.0") if ".dat" in file]



dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))


    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 1]})
    data["Participant"] = "MIT-Normal_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 128
    data["Database"] = "MIT-Normal"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-Normal_%.2i" %(participant)
    anno["Sampling_Rate"] = 128
    anno["Database"] = "MIT-Normal"

    # Select only 1h of recording (otherwise it's too big)
    data = data[460800:460800*3].reset_index(drop=True)
    anno = anno[(anno["Rpeaks"] > 460800) & (anno["Rpeaks"] <= 460800*2)].reset_index(drop=True)
    anno["Rpeaks"] = anno["Rpeaks"] - 460800


    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)



# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)


# Quick test
#import neurokit2 as nk
#nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1001])
