Complexity Analysis of Physiological Signals
============================================

A *complex system*, can be loosely defined as one that comprises of many components that interact with each other.
In science, this approach is used to investigate how relationships between a system's parts results in
its collective behaviours and how the system interacts and forms relationships with its environment.

In recent years, there has been an increase in the use of complex systems to model *physiological systems*, 
such as for medical purposes where the dynamics of physiological systems can distinguish which systems and healthy
and which are impaired. One prime example of how complexity exists in physiological systems is heart-rate variability (HRV),
where higher complexity (greater HRV) has been shown to be an indicator of health, suggesting enhanced ability to adapt to
environmental changes.

Although complexity is an important concept in health science, it is still foreign to many health scientists.
This tutorial aims to provide a simple guide to the main tenets of compexity analysis in relation to physiological signals.

Basic Concepts
---------------

Definitions
""""""""""""

Complex systems are examples of *nonlinear dynamical systems*.

A dynamical system can be described by a vector of numbers, called its *state*, which can be represented by a point in a phase space.
In terms of physiology, this vector might include values of *variables* such as lung capacity, heart rate and blood pressure. This state aims to provide a complete description of the system (in this case, health) at some point in time.

The set of all possible states is called the *state space* and the way in which the system evolves over time (i.e. change in person's health state over time)
can be referred to as *trajectory*. 

After a sufficiently long time, different trajectories may evolve or converge to a common subset of state space called an *attractor*.
The presence and behavior of attractors gives intuition about the underlying dynamical systems. This attractor can be a fixed-point
where all trajectories converge to a single point (i.e., homoeostatic equilibrium) or it can be periodic where
trajectories follow a regular path (e.g., cyclic path).


Time-delay embedding
"""""""""""""""""""""

Nonlinear time-series analysis is used to understand, characterize and predict dynamical information about human physiological systems.
This is based on the concept of *state-space reconstruction*, which allows one to reconstruct the full dynamics of
a nonlinear system from a single time series (a signal). 

One standard method for state-space reconstruction is *time-delay embedding* (or also known as delay-coordinate embedding).
It aims to identify the state of a system at a particular point in time by searching the past histrory of observations
for similar states, and by studying how these similar states evolve, in turn predict the future course of the time series.



Embedding Parameters
""""""""""""""""""""

Two basic parameters are needed to be determined for time-delayed embedding: embedding dimension *m*, and time delay *tau (τ)* (also known as embedding lag).
Vectors in a new space in this phase space reconstruction are formed from time delayed values of the scalar measurements: 1) the number m of elements
(embedding dimension), and 2) time (delay or lag). 

There are different methods to guide the choice of *tau*. In this code example below,
optimal tau is determined for which the delayed and non-delayed time series share the least *mutual information*,
and optimal dimension is determined based on *false nearest neighbours (FNN)*.


.. code-block:: python

    ecg = nk.ecg_simulate(duration=60*6, sampling_rate=150)
    signal = nk.ecg_rate(nk.ecg_peaks(ecg, sampling_rate=150), sampling_rate=150)

    delay = nk.embedding_delay(signal, delay_max=1000, method="fraser1986", show=True)
    >>> 179

    values = nk.embedding_dimension(signal, delay=delay, dimension_max=20, show=True)


More of average mutual information and false nearest neighbours can be read about in this `chapter <https://personal.egr.uri.edu/chelidz/documents/mce567_Chapter_7.pdf>`_ here.
    


Entropy as measures of Complexity
----------------------------------

The complexity of physiological signals can be represented by the entropy of these non-linear, dynamic physiological systems.

Entropy can be defined as the measure of *disorder* in a signal. 


Shannon Entropy (ShEn)
""""""""""""""""""""""

*Examples of use*
- `Atrial Fibrillation synchronization <https://www.researchgate.net/profile/Marco_Scaglione/publication/7458451_Quantification_of_synchronization_during_atrial_fibrillation_by_Shannon_entropy_Validation_in_patients_and_computer_model_of_atrial_arrhythmias/links/0912f50f18df072f4b000000/Quantification-of-synchronization-during-atrial-fibrillation-by-Shannon-entropy-Validation-in-patients-and-computer-model-of-atrial-arrhythmias.pdf>`_ 

Approximate Entropy (ApEn)
""""""""""""""""""""""""""
- Quantifies the amount of regularity and the unpredictability of fluctuations over time-series data.
- Advantages of ApEn: lower computational demand (can be designed to work for small data samples i.e. less than 50 data points and can be applied in real time)
and less sensitive to noise.
- Smaller values indicate that the data is more regular and predictable, and larger values corresponding to more complexity or irregularity in the data.

*Examples of use*
- `Respiratory patterns in Panic Disorder <https://ajp.psychiatryonline.org/doi/pdf/10.1176/appi.ajp.161.1.79`_ using parameters **m** and **r**
- `Respiration and EEG in during Eye-closed Waking and Sleep Stages` <https://pdfs.semanticscholar.org/22f2/759ffc80534e17d3461cd2ade678b7fb7468.pdf>`_ using parameters **m** and **r**
- `RR intervals and QT intervals of Heart Rate during Exercise` <https://pdfs.semanticscholar.org/9d33/d84ec1d554e330e445e94ce275790cc06c23.pdf>`_ using parameters **m** and **r**

Sample Entropy (SampEn)
"""""""""""""""""""""""
- A modification of approximate entropy
- Advantages over ApEn: data length independence and a relatively trouble-free implementation.
- Large values indicate high complexity whereas smaller values characterize more self-similar and regular signals.

*Examples of use*
- `Neonatal Heart Rate Variability <https://journals.physiology.org/doi/full/10.1152/ajpregu.00069.2002>`_ using parameters **m** and **r**
- `Atrial Fibrillation Detection in short RR intervals <https://journals.physiology.org/doi/full/10.1152/ajpheart.00561.2010?utm_source=TrendMD&utm_medium=cpc&utm_campaign=American_Journal_of_Physiology_-_Heart_and_Circulatory_Physiology_TrendMD_0>`_ using parameters **m** and **r**
- `EMG-derived Respiratory Rate  <https://upcommons.upc.edu/bitstream/handle/2117/83487/EMBC2015_Estrada_fsampen.pdf>`_ using parameters **m** and **r**
- `RR intervals and Respiratory Signal across Age <https://s3.amazonaws.com/academia.edu.documents/39126886/55c3317408aeca747d5de622.pdf?response-content-disposition=inline%3B%20filename%3DNonlinear_properties_of_cardiac_rhythm_a.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIATUSBJ6BAHW76RWJW%2F20200513%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200513T111747Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBMaCXVzLWVhc3QtMSJHMEUCIQDj1RmVbtrPNCxvjvGiWnIVc3%2F8yyj6GXT01g47DlltZwIgE7LKd3JQRn241WXE00nqev92Td7x9VHQMRZLEuNFRoQqtAMIWxAAGgwyNTAzMTg4MTEyMDAiDNSgE7SQz9ABezi7uyqRAwpQAsqcIu23Gv2IZ0CZw6sWY%2F8NeFl7vvl3tJB872Wjucyiih3b6yhr6av4LvE9H%2BbCbzH3uD0z31xVa31akaxBWkR3OEeJc%2FNwHxi0fYRTvQZpf2TPHPgghnYFufvEoPfjpGoBhUz%2FRmlWoTJ7khtHvbqhzRgwOBZhWvi9f9P%2BH%2BgHf5oRa0MS%2FfNBEiRX1vWHW7R0NIZ40tDWwC6xPTkH7GWt8XXwaSEMrrmIbKXlUKdna2tnIEzaDkfFDWnV1ueHTo3cm5W9EqccfK9%2FDp%2Fs4gnHWPmjgc%2BBCDKPbr8fLHF1ha1ia%2F8Xmg4PAxWHG7MmtGRuODYZZLETsGeKUq9rIW4ummGTKWu3vbvJHl8%2F1KhQnil%2BgoEDlgEuAyR2nTkKxBLRii%2FUjx03p0HIMllZCEsbLswFtsxEGi4pmWFIrvRqb8AFZ8Mw3IjP%2BAg9LcQK17KzvcfT6FCDVIXhW03fg7XsgkD%2BAqwthsfN9ZSz9Uc5dH69tJT%2BN1XgejR6op3SYG%2BPTk1bK8xkz98zDoqeMLmO7%2FUFOusBZUAu2AvxTyNJB3TI5xkpc9iiD9T0d%2BfGUOkoet9nrUHYUFv5AMneiLIrdmAsb1f%2FjKgUUuXwMLuIJyNZ%2FQpDO0fISOMhvg5CFB00wr6DZsbaynLfw0OP%2Biq%2Bq0IpZ2HRMQ2pSEau9tkhhcmK0yyhTChrhrv9Yt9wsDixtYIWBwv1TwgWxtm3jYApSU4siI%2F0b1V%2B%2BrhAZGX7sN4kcIPb7xIJnFeoKozoT6s8CQ7Gu4TG6YgizF6XC2HRS8jTwVz6nqn8URUH7mfjgjZhe6yj4u%2BGSnfQ7zDrIhOHhMWEZIBD2Oq%2Bk4iPnzxh%2Fg%3D%3D&X-Amz-Signature=2395ef88397eaf6a21c92f1794ac6556c88ce127a1bd43764b736710cf2bee60>`_ using parameters **m** and **r**

Fuzzy Entropy (FuzzyEn)
""""""""""""""""""""""""
- Similar to ApEn and SampEn



Multiscale Entropy (MSE)
""""""""""""""""""""""""
- Expresses different levels of either ApEn or SampEn by means of multiple factors for generating multiple time series
- Captures more useful information than using a scalar value produced by ApEn and SampEn

*Examples of use*
- `Heart Rate Variability in Rats <https://journals.physiology.org/doi/full/10.1152/ajpregu.00076.2016?utm_source=TrendMD&utm_medium=cpc&utm_campaign=American_Journal_of_Physiology_-_Regulatory%252C_Integrative_and_Comparative_Physiology_TrendMD_0>`_ using parameters **m** and **r**
