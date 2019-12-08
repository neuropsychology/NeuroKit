import numpy as np
import pandas as pd
import neurokit2 as nk


# =============================================================================
# Example 1
# =============================================================================

# Generate synthetic signals
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
emg = nk.emg_simulate(duration=10, n_bursts=3)

# Visualise biosignals
pd.DataFrame({"ECG": ecg, "EMG": emg}).plot(subplots=True, layout=(2, 1))


plot = pd.DataFrame({"ECG": ecg, "EMG": emg}).plot(subplots=True, layout=(2, 1))
plot[0][0].get_figure().savefig("README_simulation.png", dpi=300)
