import pandas as pd
import numpy as np
import neurokit2 as nk

# Load True R-peaks location
datafiles = [pd.read_csv("../../data/gudb/Rpeaks.csv"),
             pd.read_csv("../../data/mit_arrhythmia/Rpeaks.csv"),
             pd.read_csv("../../data/mit_normal/Rpeaks.csv"),
             pd.read_csv("../../data/fantasia/Rpeaks.csv")]

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

            # Interpolate
            rri = np.diff(rpeaks) / sampling_rate * 1000
            desired_length = int(np.rint(rpeaks[-1] / sampling_rate * sampling_rate))
            rri = nk.signal_interpolate(rpeaks[1:], rri, x_new=np.arange(desired_length))

            # Get PSD
            psd = nk.signal_psd(rri, sampling_rate=sampling_rate)
#
#            results = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate)
#            results["Participant"] = participant
#            results["Database"] = database
#            results["Recording_Length"] = rpeaks[-1] / sampling_rate / 60
#
#            all_results = pd.concat([all_results, results], axis=0)
#
#all_results.to_csv("data.csv", index=False)




