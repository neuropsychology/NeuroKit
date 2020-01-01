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
data.plot(subplots=True, layout=(4, 1))


# Save it
plot = data.plot(subplots=True, layout=(4, 1), color=['#f44336', "#2196F3", "#9C27B0", "#FF9800"])
plot[0][0].get_figure().savefig("README_simulation.png", dpi=300)



# =============================================================================
# Respiration (RSP) processing
# =============================================================================

# Generate one minute of respiratory signal
rsp = nk.rsp_simulate(duration=60, respiratory_rate=15)

# Process it
signals, info = nk.rsp_process(rsp)

# Visualise the processing
nk.rsp_plot(signals)

# Save it
plot = nk.rsp_plot(signals)
plot.savefig("README_respiration.png", dpi=300)
