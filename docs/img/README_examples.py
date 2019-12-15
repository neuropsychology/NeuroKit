import numpy as np
import pandas as pd
import neurokit2 as nk


# =============================================================================
# Example 1
# =============================================================================

# Generate synthetic signals
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
emg = nk.emg_simulate(duration=10, n_bursts=3)

# Visualise biosignals
data = pd.DataFrame({"ECG": ecg,
                     "RSP": rsp,
                     "EMG": emg})
data.plot(subplots=True, layout=(3, 1))


# Save it
plot = data.plot(subplots=True, layout=(3, 1))
plot[0][0].get_figure().savefig("README_simulation.png", dpi=300)
