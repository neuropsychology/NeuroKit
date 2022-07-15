from timeit import default_timer as timer

import numpy as np
import pandas as pd

import neurokit2 as nk

rez = pd.DataFrame({"Dupa": [1], "Strupa": [2]})
# Utility function
def time_function(
    x,
    fun=nk.fractal_petrosian,
    index="FD_Petrosian",
    name="nk_fractal_petrosian",
    **kwargs,
):
    t0 = timer()
    rez, info = fun(x, **kwargs)
    t1 = timer() - t0

    if name == "nk_complexity_rqa":
        rez = rez.add_prefix("RQA_")
        out = pd.DataFrame({"Result": rez.iloc[0].to_numpy(), "Index": rez.columns})
        out["Index"] = out["Index"].str.replace("_", " (") + ")"
    elif index == "MFDFA":
        rez = rez.add_prefix("MFDFA_")
        out = pd.DataFrame({"Result": rez.iloc[0].to_numpy(), "Index": rez.columns})
        out["Index"] = out["Index"].str.replace("_", " (") + ")"
    elif index == "DispEn":
        out = pd.DataFrame({"Result": [rez, info["RDEn"]], "Index": [index, "RDEn"]})
    elif index == "MIG":
        out = pd.DataFrame({"Result": [rez, info["FC"]], "Index": [index, "FC"]})
    elif name == "nk_entropy_rate":
        out = pd.DataFrame({"Result": [info["MaxRatEn"]], "Index": [index]})
    else:
        out = pd.DataFrame({"Result": [rez], "Index": [index]})
    out["Duration"] = t1
    out["Method"] = name

    return out


# Parameters
# df = pd.read_csv("data_Signals.csv")

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "Random-Walk"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=10, dimension=3), color="red", show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)
# _, _ = nk.entropy_rate(signal, kmax=np.arange(1, 71, 2), show=True)

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

# signal = df[df["Noise_Intensity"] == 0.01][df["Method"] == "EEG"]["Signal"].values
# _, _ = nk.complexity_delay(signal, show=True)
# _, _ = nk.complexity_dimension(signal, delay=10, show=True)
# nk.complexity_attractor(nk.complexity_embedding(signal, delay=20, dimension=3), show=True)
# _, _ = nk.complexity_k(signal, k_max=100, show=True)

# ================
# Generate Signal
# ================
def run_benchmark(noise_intensity=0.01):
    # Initialize data storage
    data_signal = []
    data_complexity = []

    print("Noise intensity: {}".format(noise_intensity))
    for duration in [0.5, 1, 1.5, 2, 2.5, 3]:
        for method in [
            "Random-Walk",
            "lorenz_10_2.5_28",
            "lorenz_20_2_30",
            "oscillatory",
            "fractal",
            "EEG",
        ]:
            if method == "Random-Walk":
                delay = 10
                k = 30
                signal = nk.complexity_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    method="random",
                )
            elif method == "lorenz_10_2.5_28":
                delay = 4
                k = 30
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
                k = 30
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
                k = 30
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[2, 5, 11, 18, 24, 42, 60, 63],
                )
            elif method == "fractal":
                delay = 5
                k = 30
                signal = nk.signal_simulate(
                    duration=duration,
                    sampling_rate=1000,
                    frequency=[4, 8, 16, 32, 64],
                    amplitude=[2, 2, 1, 1, 0.5],
                )
            elif method == "EEG":
                delay = 20
                k = 20
                signal = nk.eeg_simulate(
                    duration=duration,
                    sampling_rate=1000,
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
                psd = nk.signal_psd(signal_, method="burg")
                rez = pd.DataFrame(
                    {
                        "Duration": [np.nan, np.nan, np.nan, np.nan, np.nan],
                        "Result": [
                            np.nanstd(signal_),
                            noise_intensity,
                            len(signal_),
                            psd["Frequency"].values[psd["Power"].argmax()],
                            np.random.uniform(),
                        ],
                        "Index": ["SD", "Noise", "Length", "Frequency", "Random"],
                        "Method": ["np.std", "noise", "len", "psd", "random"],
                    }
                )

                # Methods that rely on discretization
                # -------------------------------------
                for x in ["Mean", "Sign", 3, 10, 100]:
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.fractal_petrosian,
                                index=f"PFD ({x})",
                                name="nk_fractal_petrosian",
                                symbolize=x,
                            ),
                        ]
                    )
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.entropy_shannon,
                                symbolize=x,
                                index=f"ShanEn ({x})",
                                name="nk_entropy_shannon",
                            ),
                        ]
                    )
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.entropy_cumulativeresidual,
                                symbolize=x,
                                index=f"CREn ({x})",
                                name="nk_entropy_cumulativeresidual",
                            ),
                        ]
                    )
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.entropy_rate,
                                symbolize=x,
                                index=f"MaxRatEn ({x})",
                                name="nk_entropy_rate",
                            ),
                        ]
                    )

                # This method also relies on discretization, but is too long to loop over
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.complexity_lempelziv,
                            index="LZC",
                            name="nk_complexity_lempelziv",
                        ),
                    ]
                )

                # Fractals
                # ----------

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
                            k_max=k,
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
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.fractal_hurst,
                            index="H",
                            name="nk_fractal_hurst",
                            corrected=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.fractal_dfa,
                            index="DFA",
                            name="nk_fractal_dfa",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.fractal_dfa,
                            index="MFDFA",
                            name="nk_fractal_dfa",
                            multifractal=True,
                        ),
                    ]
                )

                # Entropy
                # ----------
                for bins in [3, 5, 9]:
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.entropy_ofentropy,
                                index=f"EnofEn ({bins})",
                                name="nk_entropy_ofentropy",
                                scale=10,
                                bins=bins,
                            ),
                        ]
                    )
                for bins in [10, 50, 100]:
                    rez = pd.concat(
                        [
                            rez,
                            time_function(
                                signal_,
                                nk.entropy_spectral,
                                index=f"SPEn ({bins})",
                                name="entropy_spectral",
                                c=bins,
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
                            nk.entropy_kolmogorov,
                            index="K2En",
                            name="nk_entropy_kolmogorov",
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
                            nk.entropy_attention,
                            index="AttEn",
                            name="nk_entropy_attention",
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
                            name="nk_entropy_approximate",
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
                            nk.entropy_kl,
                            index="KLEn",
                            name="nk_entropy_kl",
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
                            nk.entropy_approximate,
                            index="cApEn",
                            name="entropy_approximate",
                            delay=delay,
                            dimension=3,
                            corrected=True,
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
                            nk.entropy_power,
                            index="PowEn",
                            name="nk_entropy_power",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_distribution,
                            index="DistrEn",
                            name="nk_entropy_distribution",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_phase,
                            index="PhasEn (4)",
                            name="nk_entropy_phase",
                            delay=delay,
                            n=4,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_phase,
                            index="PhasEn (8)",
                            name="nk_entropy_phase",
                            delay=delay,
                            n=8,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_grid,
                            index="GridEn (3)",
                            name="nk_entropy_grid",
                            delay=delay,
                            n=3,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_grid,
                            index="GridEn (10)",
                            name="nk_entropy_grid",
                            delay=delay,
                            n=10,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_increment,
                            index="IncrEn",
                            name="nk_entropy_increment",
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
                            index="MSIncrEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSIncrEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_slope,
                            index="SlopEn (2)",
                            name="nk_entropy_slope",
                            dimension=3,
                            thresholds=[0.1, 45],
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_slope,
                            index="SlopEn (7)",
                            name="nk_entropy_slope",
                            dimension=3,
                            thresholds=[0.1, 15, 30, 45, 60, 75, 90],
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MSSlopEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSSlopEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_dispersion,
                            index="DispEn",
                            name="nk_entropy_dispersion",
                            dimension=3,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_dispersion,
                            index="DispEn (fluctuation)",
                            name="nk_entropy_dispersion",
                            dimension=3,
                            fluctuation=True,
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
                            nk.entropy_fuzzy,
                            index="FuzzyApEn",
                            name="nk_entropy_fuzzy",
                            delay=delay,
                            dimension=3,
                            approximate=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_fuzzy,
                            index="FuzzycApEn",
                            name="nk_entropy_fuzzy",
                            delay=delay,
                            dimension=3,
                            approximate=True,
                            corrected=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_range,
                            index="RangeEn",
                            name="entropy_range",
                            delay=delay,
                            dimension=3,
                            approximate=False,
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
                            nk.entropy_permutation,
                            index="CPEn",
                            name="nk_entropy_permutation",
                            delay=delay,
                            dimension=3,
                            conditional=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_permutation,
                            index="CWPEn",
                            name="nk_entropy_permutation",
                            delay=delay,
                            dimension=3,
                            weighted=True,
                            conditional=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_permutation,
                            index="CRPEn",
                            name="nk_entropy_permutation",
                            delay=delay,
                            dimension=3,
                            conditional=True,
                            algorithm=nk.entropy_renyi,
                            alpha=2,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_bubble,
                            index="BubbEn",
                            name="nk_entropy_bubble",
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
                            nk.entropy_cosinesimilarity,
                            index="CoSiEn",
                            name="nk_entropy_cosinesimilarity",
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
                            nk.entropy_hierarchical,
                            index="HEn",
                            name="nk_entropy_hierarchical",
                            dimension=3,
                            scale=5,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MSCoSiEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSCoSiEn",
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
                            dimension=3,
                            method="MSEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MSApEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSApEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MSPEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSPEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="CMSPEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="CMSPEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MMSPEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MMSPEn",
                        ),
                    ]
                )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="IMSPEn",
                #             name="nk_entropy_multiscale",
                #             dimension=3,
                #             method="IMSPEn",
                #         ),
                #     ]
                # )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MSWPEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSWPEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MMSWPEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MMSWPEn",
                        ),
                    ]
                )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="IMSWPEn",
                #             name="nk_entropy_multiscale",
                #             dimension=3,
                #             method="IMSWPEn",
                #         ),
                #     ]
                # )
                # Super slow
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="CMSWPEn",
                            name="nk_entropy_multiscale",
                            delay=delay,
                            dimension=3,
                            method="CMSWPEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="CMSEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="CMSEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="RCMSEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="RCMSEn",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="MMSEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MMSEn",
                        ),
                    ]
                )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="IMSEn",
                #             name="nk_entropy_multiscale",
                #             dimension=3,
                #             method="IMSEn",
                #         ),
                #     ]
                # )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_symbolicdynamic,
                            index="SyDyEn",
                            name="nk_entropy_symbolicdynamic",
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
                            index="MSSyDyEn",
                            name="nk_entropy_multiscale",
                            dimension=3,
                            method="MSSyDyEn",
                        ),
                    ]
                )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="MMSyDyEn",
                #             name="nk_entropy_multiscale",
                #             dimension=3,
                #             method="MMSyDyEn",
                #         ),
                #     ]
                # )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="FuzzyMSEn",
                            name="nk_entropy_multiscale",
                            delay=delay,
                            dimension=3,
                            method="MSEn",
                            fuzzy=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="FuzzyCMSEn",
                            name="nk_entropy_multiscale",
                            delay=delay,
                            dimension=3,
                            method="CMSEn",
                            fuzzy=True,
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.entropy_multiscale,
                            index="FuzzyRCMSEn",
                            name="nk_entropy_multiscale",
                            delay=delay,
                            dimension=3,
                            method="RCMSEn",
                            fuzzy=True,
                        ),
                    ]
                )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="FuzzyMMSEn",
                #             name="nk_entropy_multiscale",
                #             delay=delay,
                #             dimension=3,
                #             method="MMSEn",
                #             fuzzy=True,
                #         ),
                #     ]
                # )
                # rez = pd.concat(
                #     [
                #         rez,
                #         time_function(
                #             signal_,
                #             nk.entropy_multiscale,
                #             index="FuzzyIMSEn",
                #             name="nk_entropy_multiscale",
                #             delay=delay,
                #             dimension=3,
                #             method="IMSEn",
                #             fuzzy=True,
                #         ),
                #     ]
                # )

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
                            nk.complexity_relativeroughness,
                            index="RR",
                            name="nk_complexity_relativeroughness",
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
                            nk.entropy_multiscale,
                            index="MSLZC",
                            name="nk_entropy_multiscale",
                            method="LZC",
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
                            index="MSPLZC",
                            name="nk_entropy_multiscale",
                            method="PLZC",
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
                            nk.complexity_lyapunov,
                            index="LLE",
                            name="nk_complexity_lyapunov",
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
                            nk.complexity_rqa,
                            index="RQA",
                            name="nk_complexity_rqa",
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
                            nk.fishershannon_information,
                            index="FSI",
                            name="nk_fishershannon_information",
                        ),
                    ]
                )
                rez = pd.concat(
                    [
                        rez,
                        time_function(
                            signal_,
                            nk.information_gain,
                            index="MIG",
                            name="nk_information_gain",
                            delay=delay,
                        ),
                    ]
                )

                # Add info
                rez["Length"] = len(signal_)
                rez["Noise_Type"] = noise
                rez["Noise_Intensity"] = noise_intensity
                rez["Signal"] = method

                data_complexity.append(rez)
    return pd.concat(data_signal), pd.concat(data_complexity)


# run_benchmark(noise_intensity=0.01)
out = nk.parallel_run(
    run_benchmark,
    [{"noise_intensity": i} for i in np.linspace(0.01, 4, 16)],
    n_jobs=8,
    verbose=5,
)

pd.concat([out[i][0] for i in range(len(out))]).to_csv("data_Signals.csv", index=False)
pd.concat([out[i][1] for i in range(len(out))]).to_csv("data_Complexity.csv", index=False)


print("FINISHED.")
