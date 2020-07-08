import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import mne

# Download example dataset
raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif')
events = mne.read_events(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

# Create epochs including different events
event_id = {'audio/left': 1, 'audio/right': 2,
            'visual/left': 3, 'visual/right': 4}

# Create epochs (100 ms baseline + 500 ms)
epochs = mne.Epochs(raw,
                    events,
                    event_id,
                    tmin=-0.1,
                    tmax=0.5,
                    picks='eeg',
                    preload=True,
                    detrend=0,
                    baseline=(None, 0))

# Generate list of evoked objects from conditions names
evoked = [epochs[name].average() for name in ('audio', 'visual')]

# Plot topo
#mne.viz.plot_compare_evokeds(evoked, picks='eeg', axes='topo')
#plt.savefig("figures/fig1.png")
#plt.clf()

# Select subset of frontal electrodes
picks = ["EEG 001", "EEG 002", "EEG 003",
         "EEG 005", "EEG 006",
         "EEG 010", "EEG 011", "EEG 012", "EEG 013", "EEG 014"]
epochs = epochs.pick_channels(picks)


# Downsample
epochs = epochs.resample(sfreq=150)

# Convert to data frame and save
#nk.mne_to_df(epochs).to_csv("data.csv", index=False)

# =============================================================================
# MNE-based ERP analysis
# =============================================================================

# Transform each condition to array
condition1 = np.mean(epochs["audio"].get_data(), axis=1)
condition2 = np.mean(epochs["visual"].get_data(), axis=1)

# Permutation test to find significant cluster of differences
t_vals, clusters, p_vals, h0 = mne.stats.permutation_cluster_test([condition1, condition2],
                                                                  out_type='mask',
                                                                  seed=111)

# Visualize
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)

# Evoked
#evoked = [epochs[name].average() for name in ('audio', 'visual')]
#mne.viz.plot_compare_evokeds(evoked, picks=picks, combine="mean"), axes=ax0)

times = epochs.times
ax0.axvline(x=0, linestyle="--", color="black")
ax0.plot(times, np.mean(condition1, axis=0), label="Audio")
ax0.plot(times, np.mean(condition2, axis=0), label="Visual")
ax0.legend(loc="upper right")
ax0.set_ylabel("uV")

# Difference
ax1.axvline(x=0, linestyle="--", color="black")
ax1.plot(times, condition1.mean(axis=0) - condition2.mean(axis=0))
ax1.axhline(y=0, linestyle="--", color="black")
ax1.set_ylabel("Difference")

# T-values
ax2.axvline(x=0, linestyle="--", color="black")
h = None
for i, c in enumerate(clusters):
    c = c[0]
    if p_vals[i] <= 0.05:
        h = ax2.axvspan(times[c.start],
                        times[c.stop - 1],
                        color='red',
                        alpha=0.5)
    else:
        ax2.axvspan(times[c.start],
                    times[c.stop - 1],
                    color=(0.3, 0.3, 0.3),
                    alpha=0.3)
hf = ax2.plot(times, t_vals, 'g')
if h is not None:
    plt.legend((h, ), ('cluster p-value < 0.05', ))
plt.xlabel("time (ms)")
plt.ylabel("t-values")
plt.savefig("figures/fig2.png")
plt.clf()