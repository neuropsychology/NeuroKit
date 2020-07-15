import mne
import scipy
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Microstates
# =============================================================================
# Read original file (too big to be uploaded on github)
raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif', preload=True)

# Selecting the sensor types to use in the analysis. In this example, we use only EEG channels
raw = raw.pick_types(meg=False, eeg=True)

# Always use an average EEG reference when doing microstate analysis
raw = nk.eeg_rereference(raw, 'average')

# Highpass filter the data a little bit
raw = raw.filter(1, 35)

# Segment the data into 6 microstates
out = nk.microstates_segment(raw, train="gfp", n_microstates=4, standardize_eeg=True)

# Plot the topographic maps of the found microstates
nk.microstates_plot(out, gfp=out["GFP"][0:500])

# =============================================================================
# Epochs
# =============================================================================
# Get Events
events = mne.read_events(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')
events = pd.DataFrame(events[:, [0, 2]], columns=["Index", "Condition"])

# Conditions = {'audio/left': 1, 'audio/right': 2, 'visual/left': 3, 'visual/right': 4}
events = events[events["Condition"].isin([1, 2, 3, 4])]
events["Condition"].loc[events["Condition"].isin([1, 2])] = "Audio"
events["Condition"].loc[events["Condition"] != "Audio"] = "Visual"

epochs = nk.epochs_create(out["Sequence"], events["Index"], sampling_rate=150, epochs_end=0.5, event_conditions=events["Condition"])


# =============================================================================
# Results
# =============================================================================
nk.microstates_static(out["Sequence"], sampling_rate=150)["Microstate_Average_DurationMean"]

df = []  # Initialize an empty dict
for i in epochs.keys():
    data = nk.microstates_static(epochs[i]["Signal"], sampling_rate=150)
    data = pd.concat([data,
                      nk.microstates_dynamic(epochs[i]["Signal"]),
                      nk.microstates_complexity(epochs[i]["Signal"])], axis=1)
    data["Condition"] = epochs[i]["Condition"][0]
    df.append(data)
df = pd.concat(df, axis=0).reset_index(drop=True)


# =============================================================================
# Analysis
# =============================================================================
variables = [("Microstate_" + str(state) + "_" + var) for state in np.unique(out["Sequence"]) for var in ["Proportion", "LifetimeDistribution", "DurationMedian", "DurationMean"]]
variables += list(df.copy().filter(regex='_to_').columns)
variables += ["Microstate_Entropy_Shannon"]


for var in variables:
    ttest = scipy.stats.ttest_ind(df[df["Condition"] == "Audio"][var], df[df["Condition"] == "Visual"][var], equal_var=False, nan_policy="omit")
    if ttest[1] < 0.2:
        print(var + ": t = %.2f, p = %.2f" % (ttest[0], ttest[1]))

df.copy().filter(regex='Proportion|Condition').boxplot(by="Condition")
