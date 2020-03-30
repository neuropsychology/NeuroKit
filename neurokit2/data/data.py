# -*- coding: utf-8 -*-
import pandas as pd


def data(dataset="bio_eventrelated_100hz"):
    """Example datasets

    Download and load available example `datasets <https://github.com/neuropsychology/NeuroKit/tree/master/data#datasets>`_.

    Parameters
    ----------
    dataset : str
        The name of the dataset. The list and description is available `here <https://github.com/neuropsychology/NeuroKit/tree/master/data#datasets>`_.

    Returns
    -------
    DataFrame
        The data.


    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> data = nk.data("bio_eventrelated_100hz")
    """
    # TODO: one could further improve this function with like
    # selectors 'ecg=True, eda=True, restingstate=True' that would
    # find the most appropriate dataset

    path = "https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/"

    data = pd.read_csv(path + dataset + ".csv")

    return data
