# -*- coding: utf-8 -*-
import os
import pandas as pd
import sklearn.datasets
import json
import urllib


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
    
    dataset = dataset.tolower()
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
        try:
            df = json.loads(dataset)
        except:
            with urllib.request.urlopen(path + dataset) as url:
                df = json.loads(url.read().decode())
        return df

    # CSV and text
    if dataset.endswith(".txt") or dataset.endswith(".csv"):
        try:
            df = pd.read_csv(dataset)
        except:
            df = pd.read_csv(path + dataset)
    

    return df
