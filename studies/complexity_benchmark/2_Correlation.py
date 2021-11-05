import numpy as np
import pandas as pd

import neurokit2 as nk

signal = nk.signal_simulate(duration=10, sampling_rate=200, frequency=[5, 10], noise=0)
signal = nk.standardize(signal)
# nk.complexity_embedding(signal, delay=20, dimension=3, show=True)


data = pd.DataFrame()
for noise in np.linspace(-2, 2, 20):
    print(f"noise: {noise}")
    for noise_intensity in np.linspace(0.01, 2, 20):
        x = nk.signal_noise(duration=10, sampling_rate=200, beta=noise)
        sig = signal + (nk.standardize(x) * noise_intensity)
        d, _ = nk.complexity(sig, which=["fast", "medium"], delay=20, dimension=3)
        d = d.drop("ShanEn", axis=1)
        d["ShanEn_10"], _ = nk.entropy_shannon(pd.cut(sig, 10, labels=False))
        d["ShanEn_100"], _ = nk.entropy_shannon(pd.cut(sig, 100, labels=False))
        d["ShanEn_1000"], _ = nk.entropy_shannon(pd.cut(sig, 1000, labels=False))
        d["CREn_10"], _ = nk.entropy_cumulative_residual(pd.cut(sig, 10, labels=False))
        d["CREn_100"], _ = nk.entropy_cumulative_residual(pd.cut(sig, 100, labels=False))
        d["CREn_1000"], _ = nk.entropy_cumulative_residual(pd.cut(sig, 1000, labels=False))
        d["PFD_a"], _ = nk.fractal_petrosian(sig, method="A")
        d["PFD_b"], _ = nk.fractal_petrosian(sig, method="B")
        d["PFD_c"] = d["PFD"]  # already computed
        d["PFD_d"], _ = nk.fractal_petrosian(sig, method="D")
        d = d.drop(["PFD"], axis=1)
        d["Noise"] = noise
        d["Intensity"] = noise_intensity
        data = pd.concat([data, d], axis=0)
data.to_csv("data_Correlations.csv", index=False)
print("DONE.")
