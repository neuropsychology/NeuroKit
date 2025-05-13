# -*- coding: utf-8 -*-
import json
import os
import pickle
import urllib

import pandas as pd
from sklearn import datasets as sklearn_datasets


def data(dataset="bio_eventrelated_100hz"):
    """**NeuroKit Datasets**

    NeuroKit includes datasets that can be used for testing. These datasets are not downloaded
    automatically with the package (to avoid increasing its weight), but can be downloaded via the
    ``nk.data()`` function (note that an internet connection is necessary). See the examples below.

    **Signals**: The following signals (that will return an array) are available:

    * **ecg_1000hz**: Returns a vector containing ECG signal (``sampling_rate=1000``).
    * **ecg_3000hz**: Returns a vector containing ECG signal (``sampling_rate=3000``).
    * **rsp_1000hz**: Returns a vector containing RSP signal (``sampling_rate=1000``).
    * **eeg_150hz**: Returns a vector containing EEG signal (``sampling_rate=150``).
    * **eog_100hz**: Returns a vector containing vEOG signal (``sampling_rate=100``).

    **DataFrames**: The following datasets (that will return a ``pd.DataFrame``) are available:

    * **iris**: Convenient access to the Iris dataset in a DataFrame, exactly how it is in R.
    * **eogs_200hz**: Returns a DataFrame with ``hEOG``, ``vEOG``.

      * Single subject
      * Visual and horizontal electrooculagraphy
      * ``sampling_rate=200``

    * **bio_resting_5min_100hz**: Returns a DataFrame with ``ECG``, ``PPG``, ``RSP``.

      * Single subject
      * Resting-state of 5 min (pre-cropped, with some ECG noise towards the end)
      * ``sampling_rate=100``

    * **bio_resting_8min_100hz**: Returns a DataFrame with ``ECG``, ``RSP``, ``EDA``,
      ``PhotoSensor``.

      * Single subject
      * Resting-state of 8 min when the photosensor is low (need to crop the data)
      * ``sampling_rate=100``

    * **bio_resting_8min_200hz**: Returns a dictionary with four subjects (``S01``, ``S02``,
      ``S03``, ``S04``).

      * Resting-state recordings
      * 8 min (``sampling_rate=200``)
      * Each subject is DataFrame  with ``ECG``, ``RSP`, ``PhotoSensor``, ``Participant``

    * **bio_eventrelated_100hz**: Returns a DataFrame with ``ECG``, ``EDA``, ``Photosensor``,
      ``RSP``.

      * Single subject
      * Event-related recording of a participant watching 4 images for 3 seconds (the condition
        order was: ``["Negative", "Neutral", "Neutral", "Negative"]``)
      * ``sampling_rate=100``

    * **eeg_1min_200hz**: Returns an MNE raw object containing 1 min of EEG
      data (from the MNE-sample dataset).

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
    -------
    DataFrame
        The data.


    Examples
    ---------

    **Single signals and vectors**

    .. ipython:: python

      import neurokit2 as nk

      ecg = nk.data(dataset="ecg_1000hz")
      @savefig p_datasets1.png scale=100%
      nk.signal_plot(ecg[0:10000], sampling_rate=1000)
      @suppress
      plt.close()

    .. ipython:: python

      rsp = nk.data(dataset="rsp_1000hz")
      @savefig p_datasets2.png scale=100%
      nk.signal_plot(rsp[0:20000], sampling_rate=1000)
      @suppress
      plt.close()

    .. ipython:: python

      eeg = nk.data("eeg_150hz")
      @savefig p_data3.png scale=100%
      nk.signal_plot(eeg, sampling_rate=150)
      @suppress
      plt.close()

    .. ipython:: python

      eog = nk.data("eog_100hz")
      @savefig p_data4.png scale=100%
      nk.signal_plot(eog[0:2000], sampling_rate=100)
      @suppress
      plt.close()

    **DataFrames**

    .. ipython:: python

      data = nk.data("iris")
      data.head()

    .. ipython:: python

      data = nk.data(dataset="eogs_200hz")
      @savefig p_datasets5.png scale=100%
      nk.signal_plot(data[0:4000], standardize=True, sampling_rate=200)
      @suppress
      plt.close()

    .. ipython:: python

      data = nk.data(dataset="bio_resting_5min_100hz")
      @savefig p_datasets6.png scale=100%
      nk.standardize(data).plot()
      @suppress
      plt.close()

    .. ipython:: python

      data = nk.data(dataset="bio_resting_8min_100hz")
      @savefig p_datasets7.png scale=100%
      nk.standardize(data).plot()
      @suppress
      plt.close()

    .. ipython:: python

      data = nk.data("bio_resting_8min_200hz")
      data.keys()
      data["S01"].head()

    .. ipython:: python

      data = nk.data("bio_eventrelated_100hz")
      @savefig p_data8.png scale=100%
      nk.standardize(data).plot()
      @suppress
      plt.close()

    .. ipython:: python

      raw = nk.data("eeg_1min_200hz")
      @savefig p_data9.png scale=100%
      nk.signal_plot(raw.get_data()[0:3, 0:2000], sampling_rate=200)
      @suppress
      plt.close()

    """
    # TODO: one could further improve this function with like
    # selectors 'ecg=True, eda=True, restingstate=True' that would
    # find the most appropriate dataset

    dataset = dataset.lower()

    path = "https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/"

    # Signals as vectors =======================
    if dataset in ["eeg", "eeg_150hz", "eeg.txt"]:
        return pd.read_csv(path + "eeg.txt").values[:, 0]

    if dataset in ["rsp", "rsp_1000hz", "rsp_1000hz.txt"]:
        return pd.read_csv(path + "rsp_1000hz.txt", header=None).values[:, 0]

    if dataset in ["ecg", "ecg_1000hz", "ecg_1000hz.csv"]:
        return pd.read_csv(path + "ecg_1000hz.csv")["ECG"].values

    if dataset in ["ecg_3000hz", "ecg_3000hz.csv"]:
        return pd.read_csv(path + "ecg_3000hz.csv")["ECG"].values

    if dataset in ["eog", "veog", "eog_100hz", "eog_100hz.csv"]:
        return pd.read_csv(path + "eog_100hz.csv")["vEOG"].values

    # Dataframes ===============================
    if dataset == "iris":
        info = sklearn_datasets.load_iris()
        data = pd.DataFrame(
            info.data, columns=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        )
        data["Species"] = info.target_names[info.target]
        return data

    if dataset in ["eogs", "eogs_200hz", "eog_200hz", "eog_200hz.csv"]:
        return pd.read_csv(path + "eog_200hz.csv")

    # Add extension
    if dataset in ["bio_resting_8min_200hz"]:
        dataset += ".json"

    # Specific case for json file
    if dataset.endswith(".json"):
        if "https" not in dataset:
            data = pd.read_json(path + dataset, orient="index")
        else:
            data = pd.read_json(dataset, orient="index")
        df = {}
        for participant, row in data.iterrows():
            for _, data_string in row.items():
                data_list = json.loads(data_string)
                data_pd = pd.DataFrame(data_list)
                df[participant] = data_pd

        return df

    # TODO: Add more EEG (fif and edf datasets)
    if dataset in ["eeg_1min_200hz"]:

        return pickle.load(
            urllib.request.urlopen(
                "https://github.com/neuropsychology/NeuroKit/blob/dev/data/eeg_1min_200hz.pickle?raw=true"
            )
        )

    # General case
    file, ext = os.path.splitext(dataset)  # pylint: disable=unused-variable
    if ext == "":
        df = pd.read_csv(path + dataset + ".csv")
    else:
        if "https" not in dataset:
            df = pd.read_csv(path + dataset)
        else:
            df = pd.read_csv(dataset)
    return df
