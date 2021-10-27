from timeit import default_timer as timer

import numpy as np
import pandas as pd

import neurokit2 as nk


# Utility function
def time_function(
    i, x, fun=nk.fractal_petrosian, index="FD_Petrosian", name="nk_fractal_petrosian", **kwargs
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
        "Iteration": [i],
    }
    return pd.DataFrame.from_dict(dat)


# Iterations
data = []
for n in nk.expspace(100, 10 ** 4, 7).astype(int):
    print(n)
    x = nk.signal_simulate(duration=2, sampling_rate=n, frequency=[5, 10], noise=0.5)
    delay = int(n / 100)
    dimension = 3
    tolerance = 0.2

    for i in range(100):
        # data.append(
        #     time_function(i, x, nk.complexity_hjorth, index="Hjorth", name="nk_complexity_hjorth")
        # )
        # data.append(
        #     time_function(i, x, nk.complexity_hurst, index="Hurst", name="nk_complexity_hurst")
        # )
        # data.append(
        #     time_function(
        #         i,
        #         x,
        #         nk.complexity_lempelziv,
        #         index="LZC",
        #         name="nk_complexity_lempelziv_lzc",
        #         delay=delay,
        #         dimension=dimension,
        #     )
        # )
        # data.append(
        #     time_function(
        #         i,
        #         x,
        #         nk.complexity_lempelziv,
        #         index="PLZC",
        #         name="nk_complexity_lempelziv_plzc",
        #         delay=delay,
        #         dimension=dimension,
        #     )
        # )
        # data.append(
        #     time_function(
        #         i,
        #         x,
        #         nk.complexity_lyapunov,
        #         index="Lyapunov",
        #         name="nk_complexity_lyapunov",
        #         delay=delay,
        #         dimension=dimension,
        #     )
        # )
        # data.append(
        #     time_function(
        #         i,
        #         x,
        #         nk.complexity_rqa,
        #         index="RQA",
        #         method="nk_complexity_rqa",
        #         delay=delay,
        #         dimension=dimension,
        #     )
        # )
        data.append(time_function(i, x, nk.complexity_rr, index="RR", name="nk_complexity_rr"))
        data.append(
            time_function(
                i, x, nk.complexity_rr, index="RR", name="nk_complexity_rr_fft", method="fft"
            )
        )
        # data.append(
        #     time_function(
        #         i, x, nk.fisher_information, index="Fisher", name="nk_fisher_information"
        #     )
        # )
        # data.append(
        #     time_function(i, x, nk.entropy_shannon, index="ShanEn", name="nk_entropy_shannon")
        # )
        # data.append(
        #     time_function(
        #         i,
        #         x,
        #         nk.entropy_cumulative_residual,
        #         index="CREn",
        #         name="nk_entropy_cumulative_residual",
        #     )
        # )
        # data.append(
        #     time_function(
        #         i, x, nk.entropy_differential, index="DiffEn", name="nk_entropy_differential"
        #     )
        # )
        # data.append(time_function(i, x, nk.entropy_svd, index="SVDen", name="nk_entropy_svd"))
        # data.append(
        #     time_function(i, x, nk.entropy_spectral, index="SpEn", name="nk_entropy_spectral")
        # )
        # data.append(time_function(i, x, nk.fractal_katz, index="Katz", name="nk_fractal_katz"))
        # data.append(
        #     time_function(i, x, nk.fractal_sevcik, index="Sevcik", name="nk_fractal_sevcik")
        # )
        # data.append(
        #     time_function(
        #         i, x, nk.fractal_petrosian, index="FD_Petrosian", name="nk_fractal_petrosian"
        #     )
        # )

pd.concat(data).to_csv("data.csv", index=False)
print("DONE.")
