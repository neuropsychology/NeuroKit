Extract and visualize heartbeats (QRS)
========================================
This example shows how to use Neurokit to extract and visualise the QRS complex in electrocardiogram (ECG) signals.

Simulate and process an ECG signal
------------------------------------
You can use `ecg_process() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_process>`_ to process your raw ECG signal. It is a convenient function that automatically processes your ECG signal following neurokit pipeline.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import neurokit2 as nk

	# Simulate 30 seconds of ECG Signal (recorded at 250 samples / second)
	ecg_signal = nk.ecg_simulate(duration=30, sampling_rate=250)
	
	# Automatically processes the (raw) ECG signal
	signals, info = nk.ecg_process(ecg_signal)
	
VIsualize the ECG data 
-------------------------
You can plot all individual heart beats, synchronized by their R peaks by doing the following:

.. code-block:: python

	# Create epochs of heart beat by cutting the signal 0.40 seconds before R-peaks for a duration of 1 second
	epochs = nk.epochs_create(signals['ECG_Clean'], events=info["ECG_R_Peaks"], sampling_rate=250, epochs_duration=1, epochs_start=-0.4)
	
	# Plotting all epochs of heart beat together
	nk.epochs_plot(epochs)

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/example_heartbeatplot.png

To have a more comprehensive overview of your ECG data, you can plot all ECG data (ECG_Raw, ECG_Clean, ECG_Rate and R-peaks) by using `ecg_plot() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_plot>`_ 

.. code-block:: python

	nk.ecg_plot(signals)

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/example_ecgplot.png