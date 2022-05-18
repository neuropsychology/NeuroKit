Try the examples in your browser
==================================

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see this tutorial <https://neurokit2.readthedocs.io/en/latest/contributing/contributing.html>`_).

The notebooks in this repo are meant to illustrate what you can do with NeuroKit. It is supposed to reveal how easy it has become to use cutting-edge methods, and still retain the liberty to change a myriad of parameters. These notebooks are organized in different sections that correspond to NeuroKit's modules.

You are free to click on the link below to run everything... **without having to install anything!** There you'll find a Jupyterlab with notebooks ready to fire up. If you need `help figuring out the interface <https://jupyterlab.readthedocs.io/en/stable/user/interface.html>`_. (The secret is ``shift+enter``).

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/neuropsychology/NeuroKit/dev?urlpath=lab%2Ftree%2Fdocs%2Fexamples

1. Analysis Paradigm
====================
Examples dedicated to specific analysis pipelines, such as for event related paradigms and resting state.

Ideas of examples to be implemented:
> Preprocessing feature signals for machine learning Analysis
> EEG + physiological activity during resting state
> Comparing interval related activity from different "mental states" (e.g. meditation, induced emotion vs. neutral)

a) Event-related paradigm
---------------------------
``eventrelated.ipynb``


Description
	This notebook guides you through the initialization of events and epochs creation. It shows you how easy it is to compare measures you've extracted from different conditions.

b) Interval-related paradigm
-----------------------------
``intervalrelated.ipynb``


Description
  Breaks down the step to extract characteristics of physiological activity for epochs of a minimum of a couple minutes

2. Biosignal Processing
=========================
Examples dedicated to processing pipelines, and measure extraction of multiple signals at a time. What's your thing ? How do you do it ?

Ideas of examples to be implemented::

> Batch preprocessing of multiple recordings
> PPG processing for respiration and temperature
> EMG overview (so many muscles to investigate)
> add yours...

a) Custom processing pipeline
------------------------------

``custom.ipynb``


Description
	This notebook breaks down the default NeuroKit pipeline used in ``_process()`` functions. It guides you in creating your own pipeline with the parameters best suited for your signals.


3. Heart rate and heart cycles
===============================
Examples dedicated to the analysis of ECG, PPG and HRV time series. Are you a fan of the `Neurovisceral integration model <https://www.researchgate.net/publication/285225132_Heart_Rate_Variability_A_Neurovisceral_Integration_Model>`_? How would you infer a cognitive or affective process with HRV ? How do you investigate the asymmetry of cardiac cycles ?

Ideas of examples to be implemented::

> Benchmark different peak detection methods
> resting state analysis of HRV
> Comparing resting state and movie watching
> add yours

a) Detecting components of the cardiac cycle
----------------------------------------------
``ecg_delineation.ipynb``

Description
	This notebook illustrate how reliable the peak detection is by analyzing the morphology of each cardiac cycles. It shows you how P-QRS-T components are extracted.

b) Looking closer at heart beats
---------------------------------
``heartbeats.ipynb``

Description
  This notebook gives hints for a thorough investigation of ECG signals by visualizing individual heart beats, interactively.

4. Electrodermal activity
===========================
Examples dedicated to the analysis of EDA signals.

Ideas of examples to be implemented::

> Pain experiments
> Temperature
> add yours

a) Extracting information in EDA
----------------------------------
``eda.ipynb``


Description
	This notebook goes at the heart of the complexity of EDA analysis by break down how Tonic and Phasic components are extracted from the signal.

5. Respiration rate and respiration cycles
===========================================
Examples dedicated to the analysis of respiratory signals, i.e. as given by a belt, or eventually, with PPG.

Ideas of examples to be implemented::

> Meditation experiments
> Stress regulation
> add yours

a) Extracting Respiration Rate Variability metrics
---------------------------------------------------
``rrv.ipynb``


Description
	This notebook breaks down the extraction of variability metrics done by ``rsp_rrv()``

6. Muscle activity
===================
Examples dedicated to the analysis of EMG signals.

Ideas of examples to be implemented::

> Suggestion and muscle activation
> Sleep data analysis
>... nothing yet!
