.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/banner.png
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest
       
.. image:: https://img.shields.io/pypi/v/neurokit2.svg
        :target: https://pypi.python.org/pypi/neurokit2

.. image:: https://img.shields.io/travis/neuropsychology/NeuroKit.svg
        :target: https://travis-ci.org/neuropsychology/NeuroKit

.. image:: https://codecov.io/gh/neuropsychology/NeuroKit/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/neuropsychology/NeuroKit
  
.. image:: https://img.shields.io/pypi/dm/neurokit2
        :target: https://pypi.python.org/pypi/neurokit2
        
.. image:: https://api.codeclimate.com/v1/badges/517cb22bd60238174acf/maintainability
   :target: https://codeclimate.com/github/neuropsychology/NeuroKit/maintainability
   :alt: Maintainability
   

  
  
**The Python Toolbox for Neurophysiological Signal Processing (EDA, ECG, PPG, EMG, EEG...)**

This is a work in progress project meant as a continuation of `NeuroKit.py <https://github.com/neuropsychology/NeuroKit.py>`_. We are looking to build a **community of people** around this collaborative project. If you're interested by getting involved, do `let us know! <https://github.com/neuropsychology/NeuroKit/issues/3>`_.


Installation
------------

To install NeuroKit, run this command in your terminal:

.. code-block::

    pip install https://github.com/neuropsychology/neurokit/zipball/master


Documentation
--------------

.. image:: https://readthedocs.org/projects/neurokit2/badge/?version=latest
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/functions-NeuroKit-orange.svg?colorB=2196F3
        :target: https://neurokit2.readthedocs.io/en/latest/functions.html
        :alt: API
        

Click on the links above and check out our tutorials:

-  `Intall Python and NeuroKit <https://neurokit2.readthedocs.io/en/latest/installation.html>`_
-  `How to contribute <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_

Examples
-------------

Simulate biosignals
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import pandas as pd
    import neurokit2 as nk

    # Generate synthetic signals
    ecg = nk.ecg_simulate(duration=10, heart_rate=70)
    emg = nk.emg_simulate(duration=10, n_bursts=3)

    # Visualise biosignals
    pd.DataFrame({"ECG": ecg, "EMG": emg}).plot(subplots=True, layout=(2, 1))


.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/README_simulation.png

Signal processing
^^^^^^^^^^^^^^^^^^

NeuroKit includes functions to facilitate signal processing:

- `signal_binarize() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_binarize>`_
- `signal_findpeaks() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_findpeaks>`_
- `signal_resample() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_resample>`_
- `signal_interpolate() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_interpolate>`_
- `signal_detrend() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_detrend>`_
- `signal_filter() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.signal_filter>`_
        
Alternatives
-------------

Here's a list of great alternative packages that you should check-out:


Multi
^^^^^^

- `BioSPPy <https://github.com/PIA-Group/BioSPPy>`_
- `PySiology <https://github.com/Gabrock94/Pysiology>`_

ECG
^^^^


- `hrv <https://github.com/rhenanbartels/hrv>`_
- `biopeaks <https://github.com/JohnDoenut/biopeaks>`_
- `py-ecg-detectors <https://github.com/berndporr/py-ecg-detectors>`_
- `HeartPy <https://github.com/paulvangentcom/heartrate_analysis_python>`_
- `pyphysio <https://github.com/MPBA/pyphysio>`_


EDA
^^^^

- `eda-explorer <https://github.com/MITMediaLabAffectiveComputing/eda-explorer>`_
- `cvxEDA <https://github.com/lciti/cvxEDA>`_
- `BreatheEasyEDA <https://github.com/johnksander/BreatheEasyEDA>`_ *(matlab)*

EEG
^^^^

- `MNE <https://github.com/mne-tools/mne-python>`_
- `unfold <https://github.com/unfoldtoolbox/unfold>`_ *(matlab)*
  
  
Eye-Tracking
^^^^^^^^^^^^

- `PyGaze <https://github.com/esdalmaijer/PyGaze>`_
- `PyTrack <https://github.com/titoghose/PyTrack>`_
