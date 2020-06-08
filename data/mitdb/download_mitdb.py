# -*- coding: utf-8 -*-
"""Script for formatting the MIT database

Steps:
    1. Download the ZIP database from https://alpha.physionet.org/content/mitdb/1.0.0/
    2. Open it with a zip-opener (WinZip, 7zip).
    3. Extract the folder of the same name (named 'mit-bih-arrhythmia-database-1.0.0') to the same folder as this script.
    4. Run this script.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
"""
import pandas as pd
import numpy as np
import wfdb
import os



data_files = ["mit-bih-arrhythmia-database-1.0.0/" + file for file in os.listdir("mit-bih-arrhythmia-database-1.0.0") if ".dat" in file] + ["mit-bih-arrhythmia-database-1.0.0/x_mitdb/" + file for file in os.listdir("mit-bih-arrhythmia-database-1.0.0/x_mitdb/") if ".dat" in file]

dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = participant
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MITDB-x" if "x_mitdb" in file else "MITDB"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = participant
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MITDB-x" if "x_mitdb" in file else "MITDB"

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)

# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

