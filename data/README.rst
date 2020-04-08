Datasets
========

NeuroKit includes datasets that can be used for testing. These datasets are not downloaded automatically with the package (to avoid increasing its weight), but can be downloaded via the `nk.data()` function.


Specific signals
------------------------------

ECG *(1000 hz)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Properties:

  - Contains ECG signal
  - Sampling rate: 1000Hz

- Download it:
.. code-block:: python

	# Using nk.data()
	data = nk.data(dataset="ecg_1000hz")['ECG']


ECG - pandas *(3000 hz)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Properties:

  - Contains ECG signal
  - Sampling rate: 3000Hz

- Download it:
.. code-block:: python

	# Using nk.data()
	data = nk.data(dataset="ecg_3000_pandas")['ECG']


Multimodal data
------------------------------

Event-related *(4 events)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


- Properties:

   - Contains signals ECG, EDA, Photosensor, RSP
   - Event-related signals
   - Sampling rate: 100Hz

- Download it:
.. code-block:: python

	# Using nk.data()
	data = nk.data(dataset="bio_eventrelated_100hz")


- Used in the following docstrings:

  - `bio_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio_analyze>`_
  - `ecg_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_analyze>`_
  - `ecg_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_eventrelated>`_
  - `ecg_rsa() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_rsa>`_
  - `ecg_rsp() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_rsp>`_
  - `eda_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_analyze>`_ 
  - `eda_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_eventrelated>`_
  - `eda_phasic() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_phasic>`_
  - `epochs_create() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_create>`_ 
  - `epochs_plot() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_plot>`_
  - `epochs_to_df() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs_to_df>`_
  - `rsp_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_analyze>`_
  - `rsp_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_eventrelated>`_
  - `signal_power() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_power>`_

- Used in the following examples:

  - `Event-related Analysis <https://neurokit2.readthedocs.io/en/dev/examples/eventrelated.html>`_
  - `Analyze Respiratory Rate Variability (RRV) <https://neurokit2.readthedocs.io/en/dev/examples/rrv.html>`_


Resting state *(5 min)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Properties:

  - Contains signals ECG, PPG, RSP
  - Resting-state signals recorded for 5 minutes
  - Sampling rate: 100Hz

- Download it:
.. code-block:: python

	# Using nk.data()
	data = nk.data(dataset="bio_resting_5min_100hz")


- Used in the following docstrings:

  - `bio_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio_analyze>`_
  - `ecg_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_analyze>`_
  - `ecg_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_intervalrelated>`_
  - `rsp_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_analyze>`_
  - `rsp_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_intervalrelated>`_

- Used in the following examples:

  - `Interval-related Analysis <https://neurokit2.readthedocs.io/en/dev/examples/intervalrelated.html>`_


Resting state *(8 min)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Properties:

  - Contains signals ECG, RSP, EMG_A, EMG_B, EDA, PhotoSensor
  - Resting-state signals recorded for 8 minutes
  - Sampling rate: 100Hz

- Download it:
.. code-block:: python

	# Using nk.data()
	data = nk.data(dataset="bio_resting_8min_100hz")


- Used in the following docstrings:

  - `eda_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_analyze>`_
  - `eda_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_intervalrelated>`_


