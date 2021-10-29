from timeit import default_timer as timer

import numpy as np
import pandas as pd

import neurokit2 as nk


# Utility function
def time_function(
    x, fun=nk.fractal_petrosian, index="FD_Petrosian", name="nk_fractal_petrosian", **kwargs
):
    t0 = timer()
    rez, _ = fun(x, **kwargs)
    if isinstance(rez, dict):
        rez = rez["RecurrenceRate"]
    t1 = timer() - t0
    dat = {
        "Duration": [t1],
        "Result": [rez],
        "Length": [len(x)],
        "Index": [index],
        "Method": [name],
    }
    return pd.DataFrame.from_dict(dat)


def run_methods(i, x, delay=1, dimension=2):
    data = []
    data.append(time_function(x, nk.complexity_hjorth, index="Hjorth", name="nk_complexity_hjorth"))
    data.append(time_function(x, nk.complexity_hurst, index="Hurst", name="nk_complexity_hurst"))
    data.append(
        time_function(
            x,
            nk.complexity_lzc,
            index="LZC",
            name="nk_complexity_lzc",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_plzc,
            index="PLZC",
            name="nk_complexity_plzc",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_mlzc,
            index="MPLZC",
            name="nk_complexity_mplzc",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_lyapunov,
            index="Lyapunov",
            name="nk_complexity_lyapunov",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_rqa,
            index="RQA",
            name="nk_complexity_rqa",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(time_function(x, nk.complexity_rr, index="RR", name="nk_complexity_rr"))
    data.append(
        time_function(
            x,
            nk.entropy_approximate,
            index="ApEn",
            name="nk_entropy_approximate",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.entropy_cumulative_residual,
            index="CREn",
            name="nk_entropy_cumulative_residual",
        )
    )
    data.append(
        time_function(x, nk.entropy_differential, index="DiffEn", name="nk_entropy_differential")
    )
    data.append(
        time_function(
            x,
            nk.entropy_fuzzy,
            index="FuzzyEn",
            name="nk_entropy_fuzzy",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_mse,
            index="MSE",
            name="nk_complexity_mse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_cmse,
            index="CMSE",
            name="nk_complexity_cmse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_rcmse,
            index="RCMSE",
            name="nk_complexity_rcmse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_fuzzymse,
            index="FuzzyMSE",
            name="nk_complexity_fuzzymse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_fuzzycmse,
            index="FuzzyCMSE",
            name="nk_complexity_fuzzycmse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_fuzzyrcmse,
            index="FuzzyRCMSE",
            name="nk_complexity_fuzzyrcmse",
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_pe,
            index="PE",
            name="nk_complexity_pe",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_wpe,
            index="WPE",
            name="nk_complexity_wpe",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.complexity_mpe,
            index="MPE",
            name="nk_complexity_mpe",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(
        time_function(
            x,
            nk.entropy_range,
            index="RangeEn",
            name="nk_entropy_range_mSampEn",
            delay=delay,
            dimension=dimension,
            method="mSampEn",
        )
    )
    data.append(
        time_function(
            x,
            nk.entropy_range,
            index="RangeEn",
            name="nk_entropy_range_mApEn",
            delay=delay,
            dimension=dimension,
            method="mApEn",
        )
    )
    data.append(
        time_function(
            x,
            nk.entropy_sample,
            index="RangeEn_mApEn",
            name="nk_entropy_sample",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(time_function(x, nk.entropy_shannon, index="ShanEn", name="nk_entropy_shannon"))
    data.append(time_function(x, nk.entropy_spectral, index="SpEn", name="nk_entropy_spectral"))
    data.append(time_function(x, nk.entropy_svd, index="SVDen", name="nk_entropy_svd"))
    data.append(
        time_function(
            x,
            nk.fractal_correlation,
            index="CD",
            name="nk_fractal_correlation",
            delay=delay,
            dimension=dimension,
        )
    )
    data.append(time_function(x, nk.fractal_dfa, index="DFA", name="nk_fractal_dfa"))
    data.append(time_function(x, nk.fractal_mfdfa, index="MFDFA", name="nk_fractal_mfdfa"))
    # data.append(time_function(x, nk.fractal_higuchi, index="HFD", name="nk_fractal_higuchi"))
    data.append(time_function(x, nk.fractal_katz, index="Katz", name="nk_fractal_katz"))
    data.append(time_function(x, nk.fractal_nld, index="NLD", name="nk_fractal_nld"))
    data.append(
        time_function(x, nk.fractal_petrosian, index="Petrosian", name="nk_fractal_petrosian")
    )
    data.append(time_function(x, nk.fractal_psdslope, index="PSDslope", name="nk_fractal_psdslope"))
    data.append(time_function(x, nk.fractal_sda, index="SDA", name="nk_fractal_sda"))
    data.append(time_function(x, nk.fractal_sevcik, index="Sevcik", name="nk_fractal_sevcik"))

    data.append(
        time_function(x, nk.fisher_information, index="Fisher", name="nk_fisher_information")
    )

    data = pd.concat(data)
    data["Iteration"] = i
    return data


# Iterations
data = []
for n in nk.expspace(100, 10 ** 4, 10).astype(int):
    print(n)
    x = nk.signal_simulate(duration=2, sampling_rate=n, frequency=[5, 10], noise=0.5)
    delay = int(n / 100)
    dimension = 3
    tolerance = 0.2
    k = 4

    args = [{"x": x, "i": i, "dimension": dimension, "delay": delay} for i in range(20)]
    out = nk.parallel_run(
        run_methods,
        args,
        n_jobs=-3,
        verbose=5,
    )
    data.append(pd.concat(out))

    pd.concat(data).to_csv("data.csv", index=False)
print("DONE.")
