========
Datasets
========

Here is a list of the datasets that has been used to illustrate Neurokit functions in examples and docstrings:

1. bio_eventrelated_100hz.csv
------------------------------
- Properties:
  - Contains signals ECG, EDA, Photosensor, RSP
  - Event-related signals
  - Sampling rate: 100Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_eventrelated_100hz.csv")

	# Using `nk.data()`
	data = nk.data(dataset="bio_eventrelated_100hz")

- Used in the following docstrings:
```
  - [`bio_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio_analyze)
  - [`ecg_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_analyze)
  - [`ecg_eventrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_eventrelated)
  - [`ecg_rsa()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_rsa)
  - [`ecg_rsp()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_rsp)
  - [`eda_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_analyze)
  - [`eda_eventrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_eventrelated)
  - [`eda_phasic()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_phasic)
  - [`epochs_create()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_create)
  - [`epochs_plot()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_plot)
  - [`epochs_to_df()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_to_df)
  - [`rsp_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_analyze)
  - [`rsp_eventrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_eventrelated)
  - [`signal_power()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_power)
```

- Used in the following examples:
```
  - [Event-related Analysis](https://neurokit2.readthedocs.io/en/dev/examples/eventrelated.html)
  - [Analyze Respiratory Rate Variability (RRV)](https://neurokit2.readthedocs.io/en/dev/examples/rrv.html)
```

2. bio_resting_5min_100hz.csv
------------------------------
- Properties:
  - Contains signals ECG, PPG, RSP
  - Resting-state signals recorded for 5 minutes
  - Sampling rate: 100Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_resting_5min_100hz.csv")

	# Using `nk.data()`
	data = nk.data(dataset="bio_resting_5min_100hz")


- Used in the following docstrings:
```
  - [`bio_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio_analyze)
  - [`ecg_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_analyze)
  - [`ecg_intervalrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_intervalrelated)
  - [`rsp_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_analyze)
  - [`rsp_intervalrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_intervalrelated)
```

- Used in the following examples:
```
  - [Interval-related Analysis](https://neurokit2.readthedocs.io/en/dev/examples/intervalrelated.html)
```

3. bio_resting_8min_100hz.csv
------------------------------
- Properties:
  - Contains signals ECG, RSP, EMG_A, EMG_B, EDA, PhotoSensor
  - Resting-state signals recorded for 8 minutes
  - Sampling rate: 100Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/bio_resting_8min_100hz.csv")

	# Using `nk.data()`
	data = nk.data(dataset="bio_resting_8min_100hz")

- Used in the following docstrings:
  - [`eda_analyze()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_analyze)
  - [`eda_intervalrelated()`](https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_intervalrelated)
```

4. ecg_1000hz.csv
-----------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 1000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_1000hz.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_1000hz")


5. ecg_2000_pandas.csv
----------------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 2000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_2000_pandas.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_2000_pandas")


6. ecg_2000_poly.csv
--------------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 2000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_2000_poly.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_2000_poly")


7. ecg_3000_pandas.csv
-----------------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 3000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_3000_pandas.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_3000_pandas")


8. ecg_3000_poly.csv
-----------------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 3000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_3000_poly.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_3000_poly")

9. ecg_3000hz.csv
-----------------------
- Properties:
  - Contains ECG signal
  - Sampling rate: 3000Hz

- Download it:
.. code-block:: python
	# Reading from the url itself:
	data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/ecg_3000hz.csv")

	# Using `nk.data()`
	data = nk.data(dataset="ecg_3000hz")
