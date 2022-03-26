import matplotlib.pyplot as plt
import numpy as np

import neurokit2 as nk

ecg12 = nk.ecg_simulate(duration=10, method="multileads")

# Visualize results
ecg12[0:10000].plot(subplots=True, figsize=(14, 10))
plt.savefig("ECG12_normal.png")

# Normal parameters (used by default)
# ===================================
# t, the starting position along the circle of each interval in radius
ti = np.array((-70, -15, 0, 15, 100))
# a, the amplitude of each spike
ai = np.array((1.2, -5, 30, -7.5, 0.75))
# b, the width of each spike
bi = np.array((0.25, 0.1, 0.1, 0.1, 0.4))

# Add noise
# ===============
ti = np.random.normal(ti, np.ones(5) * 3)
ai = np.random.normal(ai, np.abs(ai / 5))
bi = np.random.normal(bi, np.abs(bi / 5))

ecg12 = nk.ecg_simulate(duration=10, method="multileads", ti=ti, ai=ai, bi=bi)

# Visualize results
ecg12[0:10000].plot(subplots=True, figsize=(14, 10))
plt.savefig("ECG12_abnormal.png")
