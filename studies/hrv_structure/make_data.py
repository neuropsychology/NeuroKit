import numpy as np
import pandas as pd

import neurokit2 as nk

print("NeuroKit version: " + str(nk.__version__))
# Load True R-peaks location
datafiles = [pd.read_csv("../../data/gudb/Rpeaks.csv"),
             pd.read_csv("../../data/mit_arrhythmia/Rpeaks.csv"),
             pd.read_csv("../../data/fantasia/Rpeaks.csv"),
             pd.read_csv("../../data/mit_normal/Rpeaks.csv"),
             pd.read_csv("../../data/mit_long-term/Rpeaks.csv")]


# Get results
all_results = pd.DataFrame()

for file in datafiles:
    for database in np.unique(file["Database"]):

        print(str(database))
        data = file[file["Database"] == database]

        for participant in np.unique(data["Participant"]):

            data_participant = data[data["Participant"] == participant]
            sampling_rate = np.unique(data_participant["Sampling_Rate"])[0]
            rpeaks = data_participant["Rpeaks"].values

            results = nk.hrv(rpeaks, sampling_rate=sampling_rate)
            results["Participant"] = participant
            results["Database"] = database
            results["Recording_Length"] = rpeaks[-1] / sampling_rate / 60

            all_results = pd.concat([all_results, results], axis=0)

all_results.to_csv("data.csv", index=False)




