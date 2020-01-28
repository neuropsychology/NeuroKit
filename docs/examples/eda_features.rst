Analyze Electrodermal Activity (EDA) features
=============================================

This example shows how to use NeuroKit to analyze the features of an Electrodermal Activity (EDA) signal.


Extract the cleaned EDA signal
-------------------------------

In this example, we will use a simulated EDA signal. However, you can use any signal you have generated (for instance, extracted from the dataframe using <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.read_acqknowledge>`_).

.. code-block:: python
    
    # Load the NeuroKit package
    import neurokit2 as nk

    # Simulate 10 seconds of EDA Signal (recorded at 250 samples / second)
    eda_signal = nk.eda_simulate(duration=10, sampling_rate=250, n_scr=3, drift=0.01)

    
Once you have a raw EDA signal in the shape of a vector (i.e., a one-dimensional array), or a list, you can use `eda_process() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_process>`_ to process it.

.. code-block:: python

    # Process the raw EDA signal
    signals, info = nk.eda_process(eda_signal, sampling_rate=250)

This function outputs two elements, a *dataframe* containing the different signals (e.g., the raw signal, clean signal, SCR samples marking the different features etc.), and a *dictionary* containing information about the Skin Conductance Response (SCR) peaks (e.g., onsets, peak amplitude etc.).

Locate Skin Conductance Response (SCR) features
-----------------------------------------------

The processing function does two important things for our purpose: Firstly, it cleans the signal. Secondly, it detects the location of 1) peak onsets, 2) peak amplitude, and 3) half-recovery time. Let's extract these from the output.

.. code-block:: python

    # Extract clean EDA and SCR features
    cleaned = signals["EDA_Clean"]
    features = [info["SCR_Onsets"], info["SCR_Peaks"], info["SCR_Recovery"]]
    
We can now visualize the location of the peak onsets, the peak amplitude, as well as the half-recovery time points in the cleaned EDA signal, respectively marked by the red dashed line, blue dashed line, and orange dashed line.

.. code-block:: python

    # Visualize SCR features in cleaned EDA signal
    nk.events_plot(features, cleaned, color=['red', 'blue', 'orange'])
    
.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/edafeatures_1.png

Decompose EDA into Phasic and Tonic components
-----------------------------------------------

We can also decompose the EDA signal into its phasic and tonic components, or more specifically, the Phasic Skin Conductance Response (SCR) and the Tonic Skin Conductance Level (SCL) respectively.
The SCR represents the stimulus-dependent fast changing signal whereas the SCL is slow-changing and continuous. Separating these two signals helps to provide a more accurate estimation of the true SCR amplitude.

.. code-block:: python

    # Filter phasic and tonic components
    data = nk.eda_phasic(nk.standardize(eda_signal), sampling_rate=250)
    data["EDA_Raw"] = eda_signal
    data.plot()
    
.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/edafeatures_2.png

    
Quick plot
-----------

You can obtain all of these features by using the `eda_plot() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.eda_plot>`_ function on the dataframe of processed EDA.

.. code-block:: python

    nk.eda_plot(signals)

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/edafeatures_3.png