# -*- coding: utf-8 -*-
import numpy as np


def epochs_to_array(epochs):
    """Convert epochs to an array.

    TODO: make it work with uneven epochs (not the same length).

    Parameters
    ----------
    epochs : dict
        A dict containing one DataFrame per event/trial. Usually obtained via `epochs_create()`.


    Returns
    ----------
    array
        An array containing all signals.


    See Also
    ----------
    events_find, events_plot, epochs_create, epochs_plot

    Examples
    ----------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Get data
    >>> signal = nk.signal_simulate(sampling_rate=100)
    >>>
    >>> # Create epochs
    >>> epochs = nk.epochs_create(signal, events=[400, 430, 460], sampling_rate=100, epochs_end=1)
    >>> X = nk.epochs_to_array(epochs)
    >>> nk.signal_plot(X.T)

    """
    example_array = epochs[list(epochs.keys())[0]].select_dtypes(include=["number"])
    if example_array.shape[1] == 2:
        array = np.full((example_array.shape[0], len(epochs)), np.nan)
        for i, key in enumerate(epochs):
            array[:, i] = epochs[key].select_dtypes(include=["number"]).drop("Index", axis=1).values[:, 0]
    else:
        array = np.full((example_array.shape[0], example_array.shape[1] - 1, len(epochs)), np.nan)
        for i, key in enumerate(epochs):
            array[:, :, i] = epochs[key].select_dtypes(include=["number"]).drop("Index", axis=1).values

    return array
