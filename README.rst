.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/banner.png
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/pyversions/neurokit2.svg
        :target: https://pypi.python.org/pypi/neurokit2
        
.. image:: https://img.shields.io/pypi/dm/neurokit2
        :target: https://pypi.python.org/pypi/neurokit2

.. image:: https://img.shields.io/pypi/v/neurokit2.svg
        :target: https://pypi.python.org/pypi/neurokit2

.. image:: https://travis-ci.org/neuropsychology/NeuroKit.svg?branch=master
        :target: https://travis-ci.org/neuropsychology/NeuroKit

.. image:: https://codecov.io/gh/neuropsychology/NeuroKit/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/neuropsychology/NeuroKit

.. image:: https://api.codeclimate.com/v1/badges/517cb22bd60238174acf/maintainability
        :target: https://codeclimate.com/github/neuropsychology/NeuroKit/maintainability
        :alt: Maintainability


**The Python Toolbox for Neurophysiological Signal Processing (EDA, ECG, PPG, EMG, EEG...)**

This package is the continuation of `NeuroKit 1 <https://github.com/neuropsychology/NeuroKit.py>`_. It's a user-friendly package providing an easy access to complex statistical routines for researchers and clinicians with not much experience in programming, statistics or signal theory, so that they can literally **analyze physiological data with only two lines of code**.


Quick Example
=============

.. code-block:: python

    # Load packages
    import neurokit2 as nk  
    
    # Download an example dataset
    data = nk.data("bio_eventrelated_100hz")
    
    # Preprocess the data (clean signals, filter, etc.)
    processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)
    
    # Compute relevant features
    results = nk.bio_analyze(processed_data, sampling_rate=100)  


Installation
============

To install NeuroKit2, run this command in your terminal:

.. code-block::

    pip install https://github.com/neuropsychology/neurokit/zipball/master

Contribution
============

NeuroKit2 is a collaborative project with a community of contributors with all levels of development expertise. Thus, if you have some ideas for **improvement**, **new features**, or just want to **learn Python** and do something useful at the same time, do not hesitate and check out the `CONTRIBUTION <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_ guide.


Documentation
=============

.. image:: https://readthedocs.org/projects/neurokit2/badge/?version=latest
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/functions-API-orange.svg?colorB=2196F3
        :target: https://neurokit2.readthedocs.io/en/latest/functions.html
        :alt: API
        
.. image:: https://img.shields.io/badge/tutorials-help-orange.svg?colorB=E91E63
        :target: https://neurokit2.readthedocs.io/en/latest/tutorials/index.html
        :alt: Tutorials
        
.. image:: https://img.shields.io/badge/documentation-pdf-purple.svg?colorB=FF9800
        :target: https://neurokit2.readthedocs.io/_/downloads/en/latest/pdf/
        
.. image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/sangfrois/NeuroKit/dev?urlpath=lab%2Ftree%2Fdocs%2Fexamples
        
.. image:: https://img.shields.io/gitter/room/neuropsychology/NeuroKit.js.svg
        :target: https://gitter.im/NeuroKit/community
        :alt: Chat on Gitter
        

Click on the links above and check out our tutorials:

Tutorials
---------

-  `Install Python and NeuroKit <https://neurokit2.readthedocs.io/en/latest/installation.html>`_
-  `How to contribute <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_
-  `Understanding NeuroKit <https://neurokit2.readthedocs.io/en/latest/tutorials/understanding.html>`_
-  `Included datasets <https://neurokit2.readthedocs.io/en/latest/datasets.html>`_

Examples
--------

-  `Simulate Artificial Physiological Signals <https://neurokit2.readthedocs.io/en/latest/examples/simulation.html>`_
-  `Customize your Processing Pipeline <https://neurokit2.readthedocs.io/en/latest/examples/custom.html>`_
-  `Event-related Analysis <https://neurokit2.readthedocs.io/en/latest/examples/eventrelated.html>`_
-  `Analyze Electrodermal Activity (EDA) <https://neurokit2.readthedocs.io/en/latest/examples/eda.html>`_
-  `Analyze Respiratory Rate Variability (RRV) <https://neurokit2.readthedocs.io/en/latest/examples/rrv.html>`_
-  `Extract and Visualize Individual Heartbeats <https://neurokit2.readthedocs.io/en/latest/examples/heartbeats.html>`_
-  `Locate P, Q, S and T waves in ECG <https://neurokit2.readthedocs.io/en/latest/examples/ecg_delineation.html>`_


*You can try out these examples directly in your browser by* `clicking here <https://github.com/neuropsychology/NeuroKit/tree/master/docs/examples#cloud-based-interactive-examples>`_.

**Don't know which tutorial is suited for your case?** Follow this flowchart:


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/workflow.png
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest
        
Citation
=========

.. image:: https://zenodo.org/badge/218212111.svg
   :target: https://zenodo.org/badge/latestdoi/218212111

.. image:: https://img.shields.io/badge/details-authors-purple.svg?colorB=9C27B0
   :target: https://neurokit2.readthedocs.io/en/latest/credits.html
   

.. code-block:: python

    nk.cite()


.. code-block:: tex

    You can cite NeuroKit2 as follows:

    - Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Pham, H., Lesspinasse, F., 
      Schölzel, C., & S H Chen, A. (2020). NeuroKit2: A Python Toolbox for Neurophysiological 
      Signal Processing. Retrieved March 28, 2020, from https://github.com/neuropsychology/NeuroKit

    Full bibtex reference:

    @misc{neurokit2,
      doi = {10.5281/ZENODO.3597887},
      url = {https://github.com/neuropsychology/NeuroKit},
      author = {Makowski, Dominique and Pham, Tam and Lau, Zen J. and Brammer, Jan C. and Pham, Hung and Lespinasse, Fran\c{c}ois and Schölzel, Christopher and S H Chen, Annabel},
      title = {NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing},
      publisher = {Zenodo},
      year = {2020},
    }

..
    Design
    =======

    *NeuroKit2* is designed to provide a **consistent**, **accessible** yet **powerful** and **flexible** API. 

    - **Consistency**: For each type of signals (ECG, RSP, EDA, EMG...), the same function names are called (in the form :code:`signaltype_functiongoal()`) to achieve equivalent goals, such as :code:`*_clean()`, :code:`*_findpeaks()`, :code:`*_process()`, :code:`*_plot()` (replace the star with the signal type, e.g., :code:`ecg_clean()`).
    - **Accessibility**: Using NeuroKit2 is made very easy for beginners through the existence of powerful high-level "master" functions, such as :code:`*_process()`, that performs cleaning, preprocessing and processing with sensible defaults.
    - **Flexibility**: However, advanced users can very easily build their own custom analysis pipeline by using the mid-level functions (such as :code:`*_clean()`, :code:`*_rate()`), offering more control and flexibility over their parameters.


Overview
========

Simulate physiological signals
------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import neurokit2 as nk

    # Generate synthetic signals
    ecg = nk.ecg_simulate(duration=10, heart_rate=70)
    ppg = nk.ppg_simulate(duration=10, heart_rate=70)
    rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
    eda = nk.eda_simulate(duration=10, scr_number=3)
    emg = nk.emg_simulate(duration=10, burst_number=2)

    # Visualise biosignals
    data = pd.DataFrame({"ECG": ecg,
                         "PPG": ppg,
                         "RSP": rsp,
                         "EDA": eda,
                         "EMG": emg})
    nk.signal_plot(data, subplots=True)


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_simulation.png
        :target: https://neurokit2.readthedocs.io/en/latest/examples/simulation.html


Electrodermal Activity (EDA/GSR)
---------------------------------

.. code-block:: python

    # Generate 10 seconds of EDA signal (recorded at 250 samples / second) with 2 SCR peaks
    eda = nk.eda_simulate(duration=10, sampling_rate=250, scr_number=2 drift=0.01)

    # Process it
    signals, info = nk.eda_process(eda, sampling_rate=250)

    # Visualise the processing
    nk.eda_plot(signals, sampling_rate=250)

.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_eda.png
        :target: https://neurokit2.readthedocs.io/en/latest/examples/eda.html


Cardiac activity (ECG)
-----------------------

.. code-block:: python

    # Generate 15 seconds of ECG signal (recorded at 250 samples / second)
    ecg = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70)

    # Process it
    signals, info = nk.ecg_process(ecg, sampling_rate=250)

    # Visualise the processing
    nk.ecg_plot(signals, sampling_rate=250)


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_ecg.png
        :target: https://neurokit2.readthedocs.io/en/latest/examples/heartbeats.html


Respiration (RSP)
------------------

.. code-block:: python

    # Generate one minute of respiratory (RSP) signal (recorded at 250 samples / second)
    rsp = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)

    # Process it
    signals, info = nk.rsp_process(rsp, sampling_rate=250)

    # Visualise the processing
    nk.rsp_plot(signals, sampling_rate=250)


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_rsp.png
        :target: https://neurokit2.readthedocs.io/en/latest/examples/rrv.html


Electromyography (EMG)
-----------------------

.. code-block:: python

    # Generate 10 seconds of EMG signal (recorded at 250 samples / second)
    emg = nk.emg_simulate(duration=10, sampling_rate=250, burst_number=3)

    # Process it
    signals = nk.emg_process(emg, sampling_rate=250)

    # Visualise the processing
    nk.emg_plot(signals, sampling_rate=250)


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_emg.png


Photoplethysmography (PPG/BVP)
-------------------------------

.. code-block:: python

    # Generate 15 seconds of PPG signal (recorded at 250 samples / second)
    ppg = nk.ppg_simulate(duration=15, sampling_rate=250, heart_rate=70)



Electrogastrography (EGG)
--------------------------

Consider `helping us develop it <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_!


Electrooculography (EOG)
--------------------------

Consider `helping us develop it <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_!


Alternative Packages
=====================

Here's a list of great alternative packages that you should check out:


General
--------

- `BioSPPy <https://github.com/PIA-Group/BioSPPy>`_
- `PySiology <https://github.com/Gabrock94/Pysiology>`_
- `PsPM <https://github.com/bachlab/PsPM>`_
- `pyphysio <https://github.com/MPBA/pyphysio>`_


ECG
----

- `biopeaks <https://github.com/JohnDoenut/biopeaks>`_
- `hrv <https://github.com/rhenanbartels/hrv>`_
- `hrv-analysis <https://github.com/Aura-healthcare/hrvanalysis>`_
- `py-ecg-detectors <https://github.com/berndporr/py-ecg-detectors>`_
- `HeartPy <https://github.com/paulvangentcom/heartrate_analysis_python>`_
- `ECG_analysis <https://github.com/marianpetruk/ECG_analysis>`_
- `pyedr <https://github.com/jusjusjus/pyedr>`_
- `Systole <https://github.com/embodied-computation-group/systole>`_

EDA
---

- `eda-explorer <https://github.com/MITMediaLabAffectiveComputing/eda-explorer>`_
- `cvxEDA <https://github.com/lciti/cvxEDA>`_
- `Pypsy <https://github.com/brennon/Pypsy>`_
- `BreatheEasyEDA <https://github.com/johnksander/BreatheEasyEDA>`_ *(matlab)*
- `EDA <https://github.com/mateusjoffily/EDA>`_ *(matlab)*

EEG
----

- `MNE <https://github.com/mne-tools/mne-python>`_
- `unfold <https://github.com/unfoldtoolbox/unfold>`_ *(matlab)*
  
  
Eye-Tracking
-------------

- `PyGaze <https://github.com/esdalmaijer/PyGaze>`_
- `PyTrack <https://github.com/titoghose/PyTrack>`_


  
Notes
============

*Note: The authors do not give any warranty. If this software causes your keyboard to blow up, your brain to liquefy, your toilet to clog or a zombie plague to leak, the authors CANNOT IN ANY WAY be held responsible.*