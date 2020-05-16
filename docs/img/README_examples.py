import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk




# =============================================================================
# Quick Example
# =============================================================================


# Download example data
data = nk.data("bio_eventrelated_100hz")

# Preprocess the data (filter, find peaks, etc.)
processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)

# Compute relevant features
results = nk.bio_analyze(processed_data, sampling_rate=100)

# =============================================================================
# Simulate physiological signals
# =============================================================================

# Generate synthetic signals
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
ppg = nk.ppg_simulate(duration=10, heart_rate=70)
rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
eda = nk.eda_simulate(duration=10, scr_number=3)
emg = nk.emg_simulate(duration=10, burst_number=2)

# Visualise biosignals
data = pd.DataFrame({"ECG": ecg,
                     "PPG": ppg,
                     "RSP": rsp,
                     "EDA": eda,
                     "EMG": emg})
nk.signal_plot(data, subplots=True)


# Save it
data = pd.DataFrame({"ECG": nk.ecg_simulate(duration=10, heart_rate=70, noise=0),
                     "PPG": nk.ppg_simulate(duration=10, heart_rate=70, powerline_amplitude=0),
                     "RSP": nk.rsp_simulate(duration=10, respiratory_rate=15, noise=0),
                     "EDA": nk.eda_simulate(duration=10, scr_number=3, noise=0),
                     "EMG": nk.emg_simulate(duration=10, burst_number=2, noise=0)})
plot = data.plot(subplots=True, layout=(5, 1), color=['#f44336', "#E91E63", "#2196F3", "#9C27B0", "#FF9800"])
fig = plt.gcf()
fig.set_size_inches(10, 6, forward=True)
[ax.legend(loc=1) for ax in plt.gcf().axes]
fig.savefig("README_simulation.png", dpi=300, h_pad=3)

# =============================================================================
# Electrodermal Activity (EDA) processing
# =============================================================================

# Generate 10 seconds of EDA signal (recorded at 250 samples / second) with 2 SCR peaks
eda = nk.eda_simulate(duration=10, sampling_rate=250, scr_number=2, drift=0.1)

# Process it
signals, info = nk.eda_process(eda, sampling_rate=250)

# Visualise the processing
nk.eda_plot(signals, sampling_rate=None)

# Save it
plot = nk.eda_plot(signals, sampling_rate=None)
plot.set_size_inches(10, 6, forward=True)
plot.savefig("README_eda.png", dpi=300, h_pad=3)

# =============================================================================
# Cardiac activity (ECG) processing
# =============================================================================

# Generate 15 seconds of ECG signal (recorded at 250 samples / second)
ecg = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70, random_state=333)

# Process it
signals, info = nk.ecg_process(ecg, sampling_rate=250)

# Visualise the processing
nk.ecg_plot(signals, sampling_rate=250)

# Save it
plot = nk.ecg_plot(signals, sampling_rate=250)
plot.set_size_inches(10, 6, forward=True)
plot.savefig("README_ecg.png", dpi=300, h_pad=3)

# =============================================================================
# Respiration (RSP) processing
# =============================================================================

# Generate one minute of RSP signal (recorded at 250 samples / second)
rsp = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)

# Process it
signals, info = nk.rsp_process(rsp, sampling_rate=250)

# Visualise the processing
nk.rsp_plot(signals, sampling_rate=250)

# Save it
plot = nk.rsp_plot(signals, sampling_rate=250)
plot.set_size_inches(10, 6, forward=True)
plot.savefig("README_rsp.png", dpi=300, h_pad=3)

# =============================================================================
# Electromyography (EMG) processing
# =============================================================================

# Generate 10 seconds of EMG signal (recorded at 250 samples / second)
emg = nk.emg_simulate(duration=10, sampling_rate=250, burst_number=3)

# Process it
signals, info = nk.emg_process(emg, sampling_rate=250)

# Visualise the processing
nk.emg_plot(signals, sampling_rate=250)

# Save it
plot = nk.emg_plot(signals, sampling_rate=250)
plot.set_size_inches(10, 6, forward=True)
plot.savefig("README_emg.png", dpi=300, h_pad=3)

# =============================================================================
# Photoplethysmography (PPG/BVP)
# =============================================================================

# Generate 15 seconds of PPG signal (recorded at 250 samples / second)
ppg = nk.ppg_simulate(duration=15, sampling_rate=250, heart_rate=70, random_state=333)

# Process it
#signals, info = nk.ppg_process(emg, sampling_rate=250)






# =============================================================================
# Complexity
# =============================================================================

# Generate signal
signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)

# Find optimal time delay, embedding dimension and r
parameters = nk.complexity_optimize(signal, show=True)
parameters
# Save plot
fig = plt.gcf()
fig.set_size_inches(10, 6)
fig.savefig("README_complexity_optimize.png", dpi=300, h_pad=3)


nk.entropy_sample(signal)

# =============================================================================
# Statistics
# =============================================================================

x = np.random.normal(loc=0, scale=1, size=100000)

ci_min, ci_max = nk.hdi(x, ci=0.95, show=True)

# Save plot
fig = plt.gcf()
fig.set_size_inches(10/1.5, 6/1.5)
fig.savefig("README_hdi.png", dpi=300, h_pad=3)

# =============================================================================
# Popularity
# =============================================================================
import popularipy  # https://github.com/DominiqueMakowski/popularipy

downloads = popularipy.pypi_downloads("neurokit2")
stars = popularipy.github_stars("neuropsychology/neurokit", "b547333010d0b1253ab44569df3efd94c8a93a63 ")

data = downloads.merge(stars)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(7, 3))

data.plot.area(x="Date", y="Downloads", ax=axes[0], legend=False, color="#2196F3")
data.plot(x="Date", y="Trend", ax=axes[0], legend=False, color="#E91E63")
data.plot.area(x="Date", y="Stars", ax=axes[1], legend=False, color="#FF9800")

# Clean axes
axes[0].xaxis.label.set_visible(False)
axes[0].xaxis.set_ticks_position("none")
axes[0].set_xticklabels([])
axes[0].text(0.5, 0.9, "Downloads / Day", horizontalalignment='center', transform=axes[0].transAxes)
axes[1].text(0.5, 0.9, "GitHub Stars", horizontalalignment='center', transform=axes[1].transAxes)
axes[1].xaxis.label.set_visible(False)

fig = plt.gcf()
fig.set_size_inches(4*3, 2*3, forward=True)
fig.savefig("README_popularity.png", dpi=300)
