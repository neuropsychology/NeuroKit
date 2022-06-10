import numpy as np
import pandas as pd

import neurokit2 as nk

# Parameters
# df = pd.read_csv("data_Signals.csv")

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "Random-Walk"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), color="red", show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_10_2.5_28"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "lorenz_20_2_30"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "oscillatory"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "fractal"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=3, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)


# ================
# Generate Signal
# ================
def run_benchmark(noise_intensity=0.01):
    # Initialize data storage
    data_signal = []
    data_tolerance = []

    print("Noise intensity: {}".format(noise_intensity))
    for duration in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for method in [
            "Random-Walk",
            "lorenz_10_2.5_28",
            "lorenz_20_2_30",
            "oscillatory",
            "fractal",
        ]:
            if method == "Random-Walk":
                delay = 10
                signal = nk.complexity_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    method="random",
                )
            elif method == "lorenz_10_2.5_28":
                delay = 4
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=10.0,
                    beta=2.5,
                    rho=28.0,
                )
            elif method == "lorenz_20_2_30":
                delay = 15
                signal = nk.complexity_simulate(
                    duration=duration * 2,
                    sampling_rate=500,
                    method="lorenz",
                    sigma=20.0,
                    beta=2,
                    rho=30.0,
                )

            elif method == "oscillatory":
                delay = 10
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[2, 5, 11, 18, 24, 42, 60, 63],
                )
            elif method == "fractal":
                delay = 5
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[4, 8, 16, 32, 64],
                    amplitude=[2, 2, 1, 1, 0.5],
                )

            # Standardize
            signal = nk.standardize(signal)

            # Add Noise
            for noise in np.linspace(-2, 2, 5):
                noise_ = nk.signal_noise(duration=duration, sampling_rate=1000, beta=noise)
                signal_ = nk.standardize(signal + (nk.standardize(noise_) * noise_intensity))

                # Save the signal to visualize the type of signals fed into the benchmarking
                if duration == 1:

                    data_signal.append(
                        pd.DataFrame(
                            {
                                "Signal": signal_,
                                "Length": len(signal_),
                                "Duration": range(1, len(signal_) + 1),
                                "Noise": noise,
                                "Noise_Intensity": noise_intensity,
                                "Method": method,
                            }
                        )
                    )

                r_range = np.linspace(0.02, 2, 50)

                for m in range(1, 9):
                    r1, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="recurrence",
                    )
                    rez = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    rez["Method"] = "Recurrence"

                    r2, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="maxApEn",
                    )
                    rez2 = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    rez2["Method"] = "ApEn"
                    rez = pd.concat([rez, rez2], axis=0)

                    r3, info = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="neighbours",
                    )
                    rez3 = pd.DataFrame({"Tolerance": info["Values"], "Score": info["Scores"]})
                    rez3["Method"] = "Neighbours"
                    rez = pd.concat([rez, rez3], axis=0)

                    r4, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="sd",
                    )
                    r5, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="adjusted_sd",
                    )
                    r6, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="chon2009",
                    )
                    r7, _ = nk.complexity_tolerance(
                        signal_,
                        delay=delay,
                        dimension=m,
                        r_range=r_range,
                        method="neurokit",
                    )

                    # rez4 = pd.DataFrame(
                    #     {
                    #         "Tolerance": r_range,
                    #         "Score": [
                    #             nk.entropy_sample(signal_, dimension=3, delay=delay, tolerance=r)[0]
                    #             for r in r_range
                    #         ],
                    #     }
                    # )
                    # rez4["Method"] = "SampEn"
                    # rez = pd.concat([rez, rez4], axis=0)

                    # rez5 = pd.DataFrame(
                    #     {
                    #         "Tolerance": r_range,
                    #         "Score": [
                    #             nk.entropy_range(signal_, dimension=3, delay=delay, tolerance=r)[0]
                    #             for r in r_range
                    #         ],
                    #     }
                    # )
                    # rez5["Method"] = "RangeEn"
                    # rez = pd.concat([rez, rez5], axis=0)

                    # rez6 = pd.DataFrame(
                    #     {
                    #         "Tolerance": r_range,
                    #         "Score": [
                    #             nk.entropy_fuzzy(signal_, dimension=3, delay=delay, tolerance=r)[0]
                    #             for r in r_range
                    #         ],
                    #     }
                    # )
                    # rez6["Method"] = "FuzzyEn"
                    # rez = pd.concat([rez, rez6], axis=0)

                    # Add info
                    rez["Optimal_Recurrence"] = r1
                    rez["Optimal_maxApEn"] = r2
                    rez["Optimal_Neighbours"] = r3
                    rez["Optimal_SD"] = r4
                    rez["Optimal_SDadj"] = r5
                    rez["Optimal_Chon"] = r6
                    rez["Optimal_NeuroKit"] = r7
                    rez["Length"] = len(signal_)
                    rez["Noise_Type"] = noise
                    rez["Noise_Intensity"] = noise_intensity
                    rez["Signal"] = method
                    rez["Dimension"] = m

                    data_tolerance.append(rez)
    return pd.concat(data_signal), pd.concat(data_tolerance)


# run_benchmark(noise_intensity=0.01)
out = nk.parallel_run(
    run_benchmark,
    [{"noise_intensity": i} for i in np.linspace(0.01, 2, 32)],
    n_jobs=32,
    verbose=5,
)

pd.concat([out[i][0] for i in range(len(out))]).to_csv("data_Signals.csv", index=False)
pd.concat([out[i][1] for i in range(len(out))]).to_csv("data_Tolerance.csv", index=False)


print("FINISHED.")
