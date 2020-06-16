# -*- coding: utf-8 -*-
"""Script for formatting the MIT-Noise Stress Test database

Steps:
    1. Download the ZIP database from https://physionet.org/content/nstdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-noise-stress-test-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import pandas as pd
import numpy as np
import wfdb
import os

data_files = ["mit-bih-noise-stress-test-database-1.0.0/" + file for file in os.listdir("mit-bih-noise-stress-test-database-1.0.0") if ".dat" in file]



dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    if ('mit-bih-noise-stress-test-database-1.0.0/119' in file or 'mit-bih-noise-stress-test-database-1.0.0/118' in file) is False:
        break

    print("Record: " + str(participant + 1) + "/" + str(len(data_files)-3))


    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = "MIT-NST_118" if "118e" in file else "MIT-NST_119"
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-NST"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-NST_118" if "118e" in file else "MIT-NST_119"
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-NST"

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)


# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

# Quick test
#import neurokit2 as nk
#nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1002])