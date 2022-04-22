from timeit import default_timer as timer

import numpy as np
import pandas as pd

import neurokit2 as nk


# Utility function
def time_function(
    x,
    fun=nk.fractal_petrosian,
    index="FD_Petrosian",
    name="nk_fractal_petrosian",
    **kwargs,
):
    t0 = timer()
    rez, _ = fun(x, **kwargs)
    if isinstance(rez, dict):
        rez = rez["RecurrenceRate"]
    t1 = timer() - t0
    dat = {
        "Duration": [t1],
        "Result": [rez],
        "Index": [index],
        "Method": [name],
    }
    return pd.DataFrame.from_dict(dat)


# Parameters


# nk.complexity_delay(
#     signal=nk.complexity_simulate(
#         duration=10,
#         sampling_rate=1000,
#         method="random",
#     ),
#     show=True,
# )
# nk.complexity_delay(
#     signal=nk.complexity_simulate(
#         duration=10,
#         sampling_rate=500,
#         method="lorenz",
#         sigma=10.0,
#         beta=2.5,
#         rho=28.0,
#     ),
#     show=True,
# )
# nk.complexity_attractor(
#     nk.complexity_embedding(
#         nk.complexity_simulate(
#             duration=10,
#             sampling_rate=500,
#             method="lorenz",
#             sigma=20,
#             beta=2,
#             rho=30,
#         ),
#         delay=15,
#     ),
#     show=True,
# )


# ================
# Generate Signal
# ================
def run_benchmark(noise_intensity=0.01):
    # Initialize data storage
    data_signal = []
    data_complexity = []

    print("Noise intensity: {}".format(noise_intensity))
    for duration in [0.5, 1, 2, 4]:
        for method in ["Random-Walk", "lorenz_10_2.5_28", "lorenz_20_2_30"]:
            if method == "Random-Walk":
                delay = 1
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

            else:
                signal = nk.signal_simulate(
                    duration=10, sampling_rate=200, frequency=[5, 10], noise=0
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

            # ================
            # Complexity
            # ================

            # Fractals
            # ----------
            rez = time_function(
                signal_,
                nk.fractal_petrosian,
                index="PFD (A)",
                name="nk_fractal_petrosian",
                method="A",
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_petrosian,
                        index="PFD (B)",
                        name="nk_fractal_petrosian",
                        method="B",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_petrosian,
                        index="PFD (C)",
                        name="nk_fractal_petrosian",
                        method="C",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_petrosian,
                        index="PFD (D)",
                        name="nk_fractal_petrosian",
                        method="D",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_katz,
                        index="KFD",
                        name="nk_fractal_katz",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_sevcik,
                        index="SFD",
                        name="nk_fractal_sevcik",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_sda,
                        index="SDAFD",
                        name="nk_fractal_sda",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_nld,
                        index="NLDFD",
                        name="nk_fractal_nld",
                        corrected=False,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_psdslope,
                        index="PSDFD (Voss1998)",
                        name="nk_fractal_psdslope",
                        method="voss1988",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_psdslope,
                        index="PSDFD (Hasselman2013)",
                        name="nk_fractal_psdslope",
                        method="hasselman2013",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_higuchi,
                        index="HFD",
                        name="nk_fractal_higuchi",
                    ),
                ]
            )

            # Entropy
            # ----------
            for x in [3, 10, 100, 1000]:
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            pd.cut(signal_, x, labels=False),
                            nk.entropy_shannon,
                            index=f"ShanEn{x}",
                            name="nk_entropy_shannon",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            pd.cut(signal_, x, labels=False),
                            nk.entropy_cumulative_residual,
                            index=f"CREn{x}",
                            name="nk_entropy_cumulative_residual",
                        ),
                    ]
                )

            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_svd,
                        index="SVDEn",
                        name="nk_entropy_svd",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_differential,
                        index="DiffEn",
                        name="nk_entropy_differential",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="PEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_spectral,
                        index="SPEn",
                        name="entropy_spectral",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_approximate,
                        index="ApEn",
                        name="entropy_approximate",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="MSPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        scale="default",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_permutation,
                        index="WPEn",
                        name="nk_entropy_permutation",
                        delay=delay,
                        dimension=3,
                        weighted=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_sample,
                        index="SampEn",
                        name="nk_entropy_sample",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_fuzzy,
                        index="FuzzyEn",
                        name="nk_entropy_fuzzy",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.entropy_multiscale,
                        index="MSEn",
                        name="nk_entropy_multiscale",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            # Other
            # ----------
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hjorth,
                        index="Hjorth",
                        name="nk_complexity_hjorth",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_rr,
                        index="RR",
                        name="nk_complexity_rr",
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fisher_information,
                        index="FI",
                        name="nk_fisher_information",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hurst,
                        index="H (corrected)",
                        name="nk_complexity_hurst",
                        corrected=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_hurst,
                        index="H (uncorrected)",
                        name="nk_complexity_hurst",
                        corrected=False,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_lempelziv,
                        index="LZC",
                        name="nk_complexity_lempelziv",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.complexity_lempelziv,
                        index="PLZC",
                        name="nk_complexity_lempelziv",
                        delay=delay,
                        dimension=3,
                        permutation=True,
                    ),
                ]
            )
            rez = pd.concat(
                [
                    rez,
                    time_function(
                        signal_,
                        nk.fractal_correlation,
                        index="CD",
                        name="nk_fractal_correlation",
                        delay=delay,
                        dimension=3,
                    ),
                ]
            )

            # Add info
            rez["Length"] = len(signal_)
            rez["Noise"] = noise
            rez["Noise_Intensity"] = noise_intensity
            rez["Signal"] = method

            data_complexity.append(rez)
    return pd.concat(data_signal), pd.concat(data_complexity)


out = nk.parallel_run(
    run_benchmark,
    [{"noise_intensity": i} for i in np.linspace(0.01, 3, 16)],
    n_jobs=8,
    verbose=5,
)

pd.concat([out[i][0] for i in range(len(out))]).to_csv("data_Signals.csv", index=False)
pd.concat([out[i][1] for i in range(len(out))]).to_csv("data_Complexity.csv", index=False)


print("FINISHED.")
