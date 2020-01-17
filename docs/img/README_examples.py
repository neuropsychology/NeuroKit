import numpy as np
import pandas as pd
import neurokit2 as nk


# =============================================================================
# Simulate physiological signals
# =============================================================================

# Generate synthetic signals
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
eda = nk.eda_simulate(duration=10, n_scr=3)
emg = nk.emg_simulate(duration=10, n_bursts=2)

# Visualise biosignals
data = pd.DataFrame({"ECG": ecg,
                     "RSP": rsp,
                     "EDA": eda,
                     "EMG": emg})
nk.signal_plot(data, subplots=True)


# Save it
data = pd.DataFrame({"ECG": nk.ecg_simulate(duration=10, heart_rate=70, noise=0),
                     "RSP": nk.rsp_simulate(duration=10, respiratory_rate=15, noise=0),
                     "EDA": nk.eda_simulate(duration=10, n_scr=3, noise=0),
                     "EMG": nk.emg_simulate(duration=10, n_bursts=2, noise=0)})
plot = data.plot(subplots=True, layout=(4, 1), color=['#f44336', "#2196F3", "#9C27B0", "#FF9800"])
plot[0][0].get_figure().savefig("README_simulation.png", dpi=300)



# =============================================================================
# Electrodermal Activity (EDA) processing
# =============================================================================

# Generate 30 seconds of EDA signal (recorded at 250 samples / second)
eda = nk.eda_simulate(duration=30, sampling_rate=250, n_scr=5, drift=0.01)

# Process it
signals, info = nk.eda_process(eda, sampling_rate=250)

# Visualise the processing
nk.eda_plot(signals, sampling_rate=250)

# Save it
plot = nk.eda_plot(signals, sampling_rate=250)
plot.savefig("README_eda.png", dpi=300)


# =============================================================================
# Cardiac activity (ECG) processing
# =============================================================================

# Generate 20 seconds of ECG signal (recorded at 250 samples / second)
ecg = nk.ecg_simulate(duration=20, sampling_rate=250, heart_rate=70, random_state=333)

# Process it
signals, info = nk.ecg_process(ecg, sampling_rate=250)

# Visualise the processing
nk.ecg_plot(signals, sampling_rate=250)

# Save it
plot = nk.ecg_plot(signals, sampling_rate=250)
plot.savefig("README_ecg.png", dpi=300)


# =============================================================================
# Respiration (RSP) processing
# =============================================================================

# Generate one minute of respiratory (RSP) signal (recorded at 250 samples / second)
rsp = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)

# Process it
signals, info = nk.rsp_process(rsp, sampling_rate=250)

# Visualise the processing
nk.rsp_plot(signals, sampling_rate=250)

# Save it
plot = nk.rsp_plot(signals, sampling_rate=250)
plot.savefig("README_respiration.png", dpi=300)

