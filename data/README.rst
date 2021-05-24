Datasets
========

NeuroKit includes datasets that can be used for testing. These datasets are not downloaded automatically with the package (to avoid increasing its weight), but can be downloaded via the `nk.data()` function.




ECG *(1000 hz)*
---------------------

+----------------+-----------+---------+
| Type           | Frequency | Signals |
+================+===========+=========+
| Single-subject | 1000 Hz   | ECG     |
+----------------+-----------+---------+

.. code-block:: python

	data = nk.data(dataset="ecg_1000hz")
    


ECG - pandas *(3000 hz)*
-----------------------------

+----------------+-----------+---------+
| Type           | Frequency | Signals |
+================+===========+=========+
| Single-subject | 3000 Hz   | ECG     |
+----------------+-----------+---------+

.. code-block:: python

	data = nk.data(dataset="ecg_3000_pandas")





Event-related *(4 events)*
------------------------------------


+----------------+-----------+----------------------------+
| Type           | Frequency | Signals                    |
+================+===========+============================+
| Single-subject | 100 Hz    | ECG, EDA, RSP, Photosensor |
| with events    |           |                            |
+----------------+-----------+----------------------------+

.. code-block:: python

	data = nk.data(dataset="bio_eventrelated_100hz")


- Used in the following docstrings:

  - `bio_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio.bio_analyze>`_
  - `ecg_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_analyze>`_
  - `ecg_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_eventrelated>`_
  - `ecg_rsa() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_rsa>`_
  - `ecg_rsp() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg.ecg_rsp>`_
  - `eda_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda.eda_analyze>`_ 
  - `eda_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda.eda_eventrelated>`_
  - `eda_phasic() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda.eda_phasic>`_
  - `epochs_create() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs.epochs_create>`_ 
  - `epochs_plot() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs.epochs_plot>`_
  - `epochs_to_df() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.epochs.epochs_to_df>`_
  - `rsp_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp.rsp_analyze>`_
  - `rsp_eventrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp.rsp_eventrelated>`_
  - `signal_power() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal.signal_power>`_

- Used in the following examples:

  - `Event-related Analysis <https://neurokit2.readthedocs.io/en/dev/examples/eventrelated.html>`_
  - `Analyze Respiratory Rate Variability (RRV) <https://neurokit2.readthedocs.io/en/dev/examples/rrv.html>`_



Resting state *(5 min)*
---------------------------

+----------------+-----------+----------------------------+
| Type           | Frequency | Signals                    |
+================+===========+============================+
| Single-subject | 100 Hz    | ECG, PPG, RSP              |
| resting state  |           |                            |
+----------------+-----------+----------------------------+

.. code-block:: python

	data = nk.data(dataset="bio_resting_5min_100hz")


- Used in the following docstrings:

  - `bio_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.bio_analyze>`_
  - `ecg_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_analyze>`_
  - `ecg_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_intervalrelated>`_
  - `rsp_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_analyze>`_
  - `rsp_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.rsp_intervalrelated>`_

- Used in the following examples:

  - `Interval-related Analysis <https://neurokit2.readthedocs.io/en/dev/examples/intervalrelated.html>`_



Resting state *(8 min)* - Single subject
---------------------------

+----------------+-----------+----------------------------+
| Type           | Frequency | Signals                    |
+================+===========+============================+
| Single-subject | 100 Hz    | ECG, RSP, EDA, Photosensor |
| resting state  |           |                            |
+----------------+-----------+----------------------------+

.. code-block:: python

	data = nk.data(dataset="bio_resting_8min_100hz")


- Used in the following docstrings:

  - `eda_analyze() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_analyze>`_
  - `eda_intervalrelated() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_intervalrelated>`_


Resting state *(8 min)* - Four subjects
---------------------------

+-------------------+-----------+----------------------------------------------------+
| Type              | Frequency | Signals                                            |
+===================+===========+====================================================+
| Multiple-subjects | 200 Hz    | ECG, RSP, Photosensor (with Participant ID labels) |
| resting state     |           |                                                    |
+-------------------+-----------+----------------------------------------------------+

.. code-block:: python

	data = nk.data(dataset="bio_resting_8min_200hz")

