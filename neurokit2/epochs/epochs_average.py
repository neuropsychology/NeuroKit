# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from .epochs_to_df import epochs_to_df


def epochs_average(epochs, which=None, show=False, **kwargs):
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

    assert (
        "Time" in data.columns
    ), "Something is wrong with the epochs data, could not find a 'Time' column in them."

    # Select only the first column
    if which is None:
        which = data.columns[0]
    if isinstance(which, str):
        which = [which]

    # Define quantile functions
    def q1(x):
        return x.quantile(0.025)

    def q2(x):
        return x.quantile(0.975)

    # Format which
    what = {i: ["mean", "std", q1, q2] for i in which}

    # Group by and average
    av = data.groupby(["Time"], as_index=False).agg(what).reset_index()
    av.columns = ["%s%s" % (a, "_%s" % b if b else "") for a, b in av.columns]

    # Format
    av.columns = av.columns.str.replace("_mean", "_Mean")
    av.columns = av.columns.str.replace("_std", "_SD")
    av.columns = av.columns.str.replace("_q1", "_CI_low")
    av.columns = av.columns.str.replace("_q2", "_CI_high")

    # Plot
    if show is True:
        for i in which:
            plt.plot(av["Time"], av[f"{i}_Mean"], label=i)
            plt.fill_between(
                av["Time"],
                av[f"{i}_CI_low"],
                av[f"{i}_CI_high"],
                alpha=0.3,
            )
        plt.legend()

    return av
