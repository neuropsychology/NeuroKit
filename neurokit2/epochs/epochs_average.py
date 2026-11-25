import matplotlib.pyplot as plt
import pandas as pd

from .epochs_to_df import epochs_to_df


def epochs_average(epochs, which=None, indices=None, show=False, **kwargs):
    """**Compute Grand Average**

    Average epochs and returns the grand average, as well as the SD and the confidence interval.

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.
    which : str or list
        The name of the column(s) to compute the average from.
    **kwargs
        Other arguments to pass (not used for now).

    See Also
    ----------
    events_find, events_plot, epochs_create, epochs_to_df

    Examples
    ----------
    * **Example with ECG Peaks**

    .. ipython:: python

      signal = nk.ecg_simulate(duration=10)
      events = nk.ecg_findpeaks(signal)
      epochs = nk.epochs_create(signal, events=events["ECG_R_Peaks"], epochs_start=-0.5,
      epochs_end=0.5)

      @savefig p_epochs_average1.png scale=100%
      grand_av = nk.epochs_average(epochs, which="Signal", show=True)
      @suppress
      plt.close()

    """
    data = epochs_to_df(epochs)

    if "Time" not in data.columns:
        raise ValueError("Something is wrong with the epochs data, could not find a 'Time' column in them.")

    # Select only the first column
    if which is None:
        which = data.columns[0]
    if isinstance(which, str):
        which = [which]
    else:
        which = list(which)

    indices_list = ["mean", "std", "ci"] if indices is None else list(indices)
    has_ci = "ci" in indices_list
    agg_indices = [idx for idx in indices_list if idx != "ci"]

    grouped = data.groupby("Time", sort=True)[which]
    parts = []
    if len(agg_indices) > 0:
        av_main = grouped.agg(agg_indices)
        av_main.columns = [f"{col}_{stat}" for col, stat in av_main.columns]
        parts.append(av_main)

    if has_ci:
        # Vectorized Cython quantile computation across all time points at once
        ci_df = grouped.quantile([0.025, 0.975]).unstack()
        ci_cols = {}
        for col in which:
            ci_cols[f"{col}_CI_low"] = ci_df[(col, 0.025)]
            ci_cols[f"{col}_CI_high"] = ci_df[(col, 0.975)]
        parts.append(pd.DataFrame(ci_cols, index=ci_df.index))

    av = pd.concat(parts, axis=1).reset_index() if len(parts) > 0 else pd.DataFrame({"Time": sorted(data["Time"].unique())})

    # Format column suffixes
    av.columns = av.columns.str.replace("_mean", "_Mean")
    av.columns = av.columns.str.replace("_median", "_Median")
    av.columns = av.columns.str.replace("_std", "_SD")

    # Ensure deterministic per-column ordering matching requested `indices`
    ordered_cols = ["Time"]
    suffix_map = {"mean": ["Mean"], "median": ["Median"], "std": ["SD"], "ci": ["CI_low", "CI_high"]}
    for col in which:
        for idx in indices_list:
            for suf in suffix_map.get(idx, [str(idx)]):
                cname = f"{col}_{suf}"
                if cname in av.columns and cname not in ordered_cols:
                    ordered_cols.append(cname)
    for cname in av.columns:
        if cname not in ordered_cols:
            ordered_cols.append(cname)
    av = av[ordered_cols]

    # Plot
    if show is True:
        for i in which:
            center_col = f"{i}_Mean" if f"{i}_Mean" in av.columns else (f"{i}_Median" if f"{i}_Median" in av.columns else None)
            if center_col is not None:
                plt.plot(av["Time"], av[center_col], label=i)
            if f"{i}_CI_low" in av.columns and f"{i}_CI_high" in av.columns:
                plt.fill_between(
                    av["Time"],
                    av[f"{i}_CI_low"],
                    av[f"{i}_CI_high"],
                    alpha=0.3,
                )
        if any(f"{i}_Mean" in av.columns or f"{i}_Median" in av.columns for i in which):
            plt.legend()

    return av
