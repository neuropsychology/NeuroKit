What software for physiological signal processing
==================================================

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see this tutorial <https://neurokit2.readthedocs.io/en/latest/tutorials/contributing.html>`_).
   
   
If you're here, it's probably that you have (or plan to have) some **physiological data** (*aka* biosignals, e.g., ECG for cardiac activity, RSP for respiration, EDA for electrodermal activity, EMG for muscle activity etc.), that you plan to **process and analyze these signals** and that **you don't know where to start**. Whether you are an undergrad, a master or PhD student, a postdoc or a full professor, you're at the right place.

So let's discuss a few things to consider to best decide on your options.

Software *vs.* programming language (packages)
-----------------------------------------------

In this context, a software would be a program that you download, install (through some sort of `.exe` file), and start similarly to most of the other programs installed on our computers. 

They are appealing because of their **(apparent) simplicity and familiarity** of usage: you can click on icons and menus and you can *see* all the available options, which makes it easy for exploration. In a way, it also feels *safe*, because you can always close the software, press *"do not save the changes"*, and start again. 

Unfortunately, when it comes to science, this comes with a set of **limitations**; they are in general quite expensive, are limited to the set of included features (it's not easy to use one feature from one software, and then another from another one), have a slow pace of updates (and thus often don't include the most recent and cutting-edge algorithms, but rather well-established ones), and are not open-source (and thus prevent to run fully reproducible analyses).

- **Software for biosignals processing**

  - `AcqKnowledge <https://www.biopac.com/product/acqknowledge-software/>`_: General physiological analysis software (ECG, PPG, RSP, EDA, EMG, ...).
  - `Kubios  <https://www.kubios.com/>`_: Heart-rate variability (HRV).
  
Unfortunately, it's the prefered option for many researchers. *Why?* For PIs, it's usually because they are established tools backed by some sort of company behind them, with experts, advertisers and sellers that do their job well. The companies also offer some guaranty in terms of training, updates, issues troubleshooting, etc. For younger researchers starting with physiological data analysis, it's usually because they don't have much (or any) **experience with programming languages**. They feel like there is already a lot of things to learn on the theorethical side of physiological signal processing, so they don't want to add on top of that, **learning a programming language**.

However, it is important to understand that you don't necessarily have to **know how to code** to use some of the packages. Moreover, some of them include a GUI (see below), which makes them very easy to use and a great alternative to the software mentioned above.


.. note::
   **TLDR**; Closed proprietary software, even though seemlingly appealing, might not a good investement of time or money. 

GUI *vs.* code
-------------

*TODO*.



- **Packages with a GUI**

  - `Ledalab <http://www.ledalab.de/>`_: EDA *(Matlab)*.
  - `PsPM <https://bachlab.github.io/PsPM/>`_: Primarly EDA *(Matlab)*.
  - `biopeaks <https://github.com/JanCBrammer/biopeaks>`_: ECG, PPG *(Python)*.
  - `mnelab <https://github.com/cbrnr/mnelab>`_: EEG *(Python)*.

.. note::
   **TLDR**; While GUIs can be good alternatives and a first step to dive into programming language-based tools, coding will provide you with more freedom, incredible power and the best fit possible for your data and issues. 


Matlab *vs.* Python *vs.* R *vs.* Julia
----------------------------------------

What is the best programming language for physiological data analysis?

**Matlab** is the historical main contender. However... *TODO*.




- **Python-based packages**
  
  - `NeuroKit2 <https://github.com/neuropsychology/NeuroKit>`_: ECG, PPG, RSP, EDA, EMG.
  - `BioSPPy <https://github.com/PIA-Group/BioSPPy>`_: ECG, RSP, EDA, EMG.
  - `PySiology <https://github.com/Gabrock94/Pysiology>`_: ECG, EDA, EMG.
  - `pyphysio <https://github.com/MPBA/pyphysio>`_: ECG, PPG, EDA.
  - `HeartPy <https://github.com/paulvangentcom/heartrate_analysis_python>`_: ECG.
  - `hrv <https://github.com/rhenanbartels/hrv>`_: ECG.
  - `hrv-analysis <https://github.com/Aura-healthcare/hrvanalysis>`_: ECG.
  - `pyhrv <https://github.com/PGomes92/pyhrv>`_: ECG.
  - `py-ecg-detectors <https://github.com/berndporr/py-ecg-detectors>`_: ECG.
  - `Systole <https://github.com/embodied-computation-group/systole>`_: PPG.
  - `eda-explorer <https://github.com/MITMediaLabAffectiveComputing/eda-explorer>`_: EDA.
  - `Pypsy <https://github.com/brennon/Pypsy>`_: EDA.
  - `MNE <https://github.com/mne-tools/mne-python>`_: EEG.
  - `tensorpac <https://github.com/EtienneCmb/tensorpac>`_: EEG.
  - `PyGaze <https://github.com/esdalmaijer/PyGaze>`_: Eye-tracking.
  - `PyTrack <https://github.com/titoghose/PyTrack>`_: Eye-tracking.
  
  
- **Matlab-based packages**

  - `BreatheEasyEDA <https://github.com/johnksander/BreatheEasyEDA>`_: EDA.
  - `EDA <https://github.com/mateusjoffily/EDA>`_: EDA.
  - `unfold <https://github.com/unfoldtoolbox/unfold>`_: EEG.
  