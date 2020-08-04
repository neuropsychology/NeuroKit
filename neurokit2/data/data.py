# -*- coding: utf-8 -*-
import os
import pandas as pd
import sklearn.datasets


def data(dataset="bio_eventrelated_100hz"):
    """Download example datasets.

    Download and load available `example datasets <https://github.com/neuropsychology/NeuroKit/tree/master/data#datasets>`_.
    Note that an internet connexion is necessary.

    Parameters
    ----------
    dataset : str
        The name of the dataset. The list and description is
        available `here <https://neurokit2.readthedocs.io/en/master/datasets.html#>`_.

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
    if dataset == "iris":
        data = sklearn.datasets.load_iris()
        return pd.DataFrame(data.data, columns=data["feature_names"])

    path = "https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/"

    # Specific case
    if dataset.lower() in ["eeg", "eeg.txt"]:
        df = pd.read_csv(path + "eeg.txt")
        return df.values[:, 0]

    # General case
    file, ext = os.path.splitext(dataset)  # pylint: disable=unused-variable
    if ext == "":
        df = pd.read_csv(path + dataset + ".csv")
    else:
        df = pd.read_csv(path + dataset)

    return df
