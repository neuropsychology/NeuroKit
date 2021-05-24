# -*- coding: utf-8 -*-
import os
import pandas as pd
import sklearn.datasets
import json


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

    dataset = dataset.lower()
    path = "https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/"

    # Specific requests
    if dataset == "iris":
        data = sklearn.datasets.load_iris()
        return pd.DataFrame(data.data, columns=data["feature_names"])

    if dataset in ["eeg", "eeg.txt"]:
        df = pd.read_csv(path + "eeg.txt")
        return df.values[:, 0]

    # Add extension
    if dataset in ["bio_resting_8min_200hz"]:
        dataset += ".json"

    # Specific case for json file
    if dataset.endswith(".json"):
        if 'https' not in dataset:
            data = pd.read_json(path + dataset, orient='index')
        else:
            data = pd.read_json(dataset, orient='index')
        df = {}
        for participant, row in data.iterrows():
            for _, data_string in row.items():
                data_list = json.loads(data_string)
                data_pd = pd.DataFrame(data_list)
                df[participant] = data_pd

        return df

    # General case
    file, ext = os.path.splitext(dataset)  # pylint: disable=unused-variable
    if ext == "":
        if dataset not in ['rsp_200hz']:
            df = pd.read_csv(path + dataset + ".csv")
        else:
            df = pd.read_csv(path + dataset + ".txt")
    else:
        if 'https' not in dataset:
            df = pd.read_csv(path + dataset)
        else:
            df = pd.read_csv(dataset)
    return df
