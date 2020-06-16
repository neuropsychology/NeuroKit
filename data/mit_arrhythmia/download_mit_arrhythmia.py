# -*- coding: utf-8 -*-
"""Script for formatting the MIT-Arrhythmia database

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

data_files = ["mit-bih-arrhythmia-database-1.0.0/" + file for file in os.listdir("mit-bih-arrhythmia-database-1.0.0") if ".dat" in file]


def read_file(file, participant):
    """Utility function
    """
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    return data, anno




dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

    data, anno = read_file(file, participant)

    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)

    # Store additional recording if available
    if "x_" + file.replace("mit-bih-arrhythmia-database-1.0.0/", "") in os.listdir("mit-bih-arrhythmia-database-1.0.0/x_mitdb/"):
        print("  - Additional recording detected.")
        data, anno = read_file("mit-bih-arrhythmia-database-1.0.0/x_mitdb/" + "x_" + file.replace("mit-bih-arrhythmia-database-1.0.0/", ""), participant)
        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)



# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

# Quick test
#import neurokit2 as nk
#nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1002])