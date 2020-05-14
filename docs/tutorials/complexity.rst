Complexity Analysis of Physiological Signals
============================================

A **complex system**, can be loosely defined as one that comprises of many components that interact with each other.
In science, this approach is used to investigate how relationships between a system's parts results in
its collective behaviours and how the system interacts and forms relationships with its environment.

In recent years, there has been an increase in the use of complex systems to model **physiological systems**, 
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

Complex systems are examples of **nonlinear dynamical systems**.

A dynamical system can be described by a vector of numbers, called its **state**, which can be represented by a point in a phase space.
In terms of physiology, this vector might include values of **variables** such as lung capacity, heart rate and blood pressure. This state aims to provide a complete description of the system (in this case, health) at some point in time.

The set of all possible states is called the **state space** and the way in which the system evolves over time (e.g., change in person's health state over time)
can be referred to as **trajectory**. 

After a sufficiently long time, different trajectories may evolve or converge to a common subset of state space called an **attractor**.
The presence and behavior of attractors gives intuition about the underlying dynamical systems. This attractor can be a fixed-point
where all trajectories converge to a single point (i.e., homoeostatic equilibrium) or it can be periodic where
trajectories follow a regular path (i.e., cyclic path).


Time-delay embedding
"""""""""""""""""""""

Nonlinear time-series analysis is used to understand, characterize and predict dynamical information about human physiological systems.
This is based on the concept of **state-space reconstruction**, which allows one to reconstruct the full dynamics of
a nonlinear system from a single time series (a signal). 

One standard method for state-space reconstruction is *time-delay embedding* (or also known as delay-coordinate embedding).
It aims to identify the state of a system at a particular point in time by searching the past history of observations
for similar states, and by studying how these similar states evolve, in turn predict the future course of the time series.



Embedding Parameters
""""""""""""""""""""

Two basic parameters are needed to be determined for time-delayed embedding: embedding dimension **m**, tolerance threshold **r**, and time delay **tau (τ)** (also known as embedding lag).
Vectors in a new space in this phase space reconstruction are formed from time delayed values of the scalar measurements. The parameter **m** determines the length of the vectors (i.e., number of elements)
to be compared where these vectors consist of time delayed values of **tau**, and **r** is the tolerance for accepting similar patterns between two vectors.

There are different methods to guide the choice of parameters. In **NeuroKit**, you can use `nk.complexity_optimize()` to estimate the optimal parameters.
.. code-block:: python

    import neurokit2 as nk

    signal = nk.signal_simulate(duration=10, frequency=1, noise=0.01)
    parameters = nk.complexity_optimize(signal)
    
    parameters
    >>> {'delay': 20, 'dimension': 5, 'r': 0.014156214774197567}
    
You can then visualize the reconstructed attractor by entering the parameters into `nk.complexity_embeddding()`.

.. code-block:: python

    embedded = nk.complexity_embedding(signal, delay=20, dimension=5, show=True)

.. image:: https://raw.github.com/neuropsychology/Neurokit/dev/docs/img/attractor.png



More of these methods can be read about in this `chapter <https://personal.egr.uri.edu/chelidz/documents/mce567_Chapter_7.pdf>`_ here.
    


Entropy as measures of Complexity
----------------------------------

The complexity of physiological signals can be represented by the entropy of these non-linear, dynamic physiological systems.

Entropy can be defined as the measure of *disorder* in a signal. 


Shannon Entropy (ShEn)
""""""""""""""""""""""
- call `nk.entropy_shannon()`

*Examples of use*


Approximate Entropy (ApEn)
""""""""""""""""""""""""""
- Quantifies the amount of regularity and the unpredictability of fluctuations over time-series data.
- Advantages of ApEn: lower computational demand (can be designed to work for small data samples i.e. less than 50 data points and can be applied in real time)
and less sensitive to noise.
- Smaller values indicate that the data is more regular and predictable, and larger values corresponding to more complexity or irregularity in the data.
- call `nk.entropy_approximate()`

*Examples of use*

| Reference | Signal | Parameters | Findings |
| --- | --- | --- | --- |
|  `Caldirola et al. (2004) <https://ajp.psychiatryonline.org/doi/pdf/10.1176/appi.ajp.161.1.79>`_ | 17min breath-by-breath recordings of respiration parameters | *m*=1, *r*=0.2 of SD of original time series | Panic disorder patients showed higher ApEn indexes in baseline RSP patterns (all parameters) than healthy subjects |
|  `Burioka et al. (2003) <https://pdfs.semanticscholar.org/22f2/759ffc80534e17d3461cd2ade678b7fb7468.pdf>`_ | 30 mins of Respiration, 20s recordings of EEG | *m*=2, *r*=0.2 of SD of original time series, *τ*=1.1s for respiration, 0.09s for EEG| Lower ApEn of respiratory movement and EEG in stage IV sleep than other stages of consciousness |
|  `Boettger et al. (2009) <https://pdfs.semanticscholar.org/9d33/d84ec1d554e330e445e94ce275790cc06c23.pdf>`_ | 64s recordings of QT and RR intervals | *m*=2, *r*=0.2 of SD of original time series | Higher ratio of ApEn(QT) to ApEn(RR) for higher intensities of exercise, reflecting sympathetic activity |  


Sample Entropy (SampEn)
"""""""""""""""""""""""
- A modification of approximate entropy
- Advantages over ApEn: data length independence and a relatively trouble-free implementation.
- Large values indicate high complexity whereas smaller values characterize more self-similar and regular signals.
- call `nk.entropy_sample()`

*Examples of use*=
| Reference | Signal | Parameters | Findings |
| --- | --- | --- | --- |
|  `Lake et al. (2002) <https://journals.physiology.org/doi/full/10.1152/ajpregu.00069.2002>`_ | 25min recordings of RR intervals | *m*=3, *r*=0.2 of SD of original time series | SampEn is lower in the course of neonatal sepsis and sepsislike illness |
| `Lake et al. (2011) <https://journals.physiology.org/doi/full/10.1152/ajpheart.00561.2010?utm_source=TrendMD&utm_medium=cpc&utm_campaign=American_Journal_of_Physiology_-_Heart_and_Circulatory_Physiology_TrendMD_0>`_ | 24h recordings of RR intervals | *m*=1, *r*= to vary | In patients over 4o years old, SampEn has high degrees of accuracy in distinguishing atrial fibrillation from normal sinus rhythm in 12-beat calculations performed hourly. | 
| `Estrada et al. (2015) <https://upcommons.upc.edu/bitstream/handle/2117/83487/EMBC2015_Estrada_fsampen.pdf>`_ | EMG diaphragm signal | *m*=1, *r*=0.3 of SD of original time series | fSampEn (fixed SampEn) method to extract RSP rate from respiratory EMG signal |
| ` Kapidzic et al. (2014) <https://s3.amazonaws.com/academia.edu.documents/39126886/55c3317408aeca747d5de622.pdf?response-content-disposition=inline%3B%20filename%3DNonlinear_properties_of_cardiac_rhythm_a.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIATUSBJ6BAHW76RWJW%2F20200513%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200513T111747Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBMaCXVzLWVhc3QtMSJHMEUCIQDj1RmVbtrPNCxvjvGiWnIVc3%2F8yyj6GXT01g47DlltZwIgE7LKd3JQRn241WXE00nqev92Td7x9VHQMRZLEuNFRoQqtAMIWxAAGgwyNTAzMTg4MTEyMDAiDNSgE7SQz9ABezi7uyqRAwpQAsqcIu23Gv2IZ0CZw6sWY%2F8NeFl7vvl3tJB872Wjucyiih3b6yhr6av4LvE9H%2BbCbzH3uD0z31xVa31akaxBWkR3OEeJc%2FNwHxi0fYRTvQZpf2TPHPgghnYFufvEoPfjpGoBhUz%2FRmlWoTJ7khtHvbqhzRgwOBZhWvi9f9P%2BH%2BgHf5oRa0MS%2FfNBEiRX1vWHW7R0NIZ40tDWwC6xPTkH7GWt8XXwaSEMrrmIbKXlUKdna2tnIEzaDkfFDWnV1ueHTo3cm5W9EqccfK9%2FDp%2Fs4gnHWPmjgc%2BBCDKPbr8fLHF1ha1ia%2F8Xmg4PAxWHG7MmtGRuODYZZLETsGeKUq9rIW4ummGTKWu3vbvJHl8%2F1KhQnil%2BgoEDlgEuAyR2nTkKxBLRii%2FUjx03p0HIMllZCEsbLswFtsxEGi4pmWFIrvRqb8AFZ8Mw3IjP%2BAg9LcQK17KzvcfT6FCDVIXhW03fg7XsgkD%2BAqwthsfN9ZSz9Uc5dH69tJT%2BN1XgejR6op3SYG%2BPTk1bK8xkz98zDoqeMLmO7%2FUFOusBZUAu2AvxTyNJB3TI5xkpc9iiD9T0d%2BfGUOkoet9nrUHYUFv5AMneiLIrdmAsb1f%2FjKgUUuXwMLuIJyNZ%2FQpDO0fISOMhvg5CFB00wr6DZsbaynLfw0OP%2Biq%2Bq0IpZ2HRMQ2pSEau9tkhhcmK0yyhTChrhrv9Yt9wsDixtYIWBwv1TwgWxtm3jYApSU4siI%2F0b1V%2B%2BrhAZGX7sN4kcIPb7xIJnFeoKozoT6s8CQ7Gu4TG6YgizF6XC2HRS8jTwVz6nqn8URUH7mfjgjZhe6yj4u%2BGSnfQ7zDrIhOHhMWEZIBD2Oq%2Bk4iPnzxh%2Fg%3D%3D&X-Amz-Signature=2395ef88397eaf6a21c92f1794ac6556c88ce127a1bd43764b736710cf2bee60>`_ | RR intervals and its corresponding RSP signal | *m*=2, *r*=0.2 of SD of original time series | During paced breathing, significant reduction of SampEn(Resp) and SampEn(RR) with age in male subjects, compared to smaller and nonsignificant SampEn decrease in females |
| `Abásolo et al. (2006) <http://epubs.surrey.ac.uk/39603/6/Abasolo_et_al_PhysiolMeas_final_version_2006.pdf>`_ | 5min recordings of EEG in 5 second epochs | *m*=1, *r*=0.25 of SD of original time series | Alzheimer's Disease patients had lower SampEn than controls in parietal and occipital regions |


Fuzzy Entropy (FuzzyEn)
""""""""""""""""""""""""
- Similar to ApEn and SampEn
- call `nk.entropy_fuzzy()`


Multiscale Entropy (MSE)
""""""""""""""""""""""""
- Expresses different levels of either ApEn or SampEn by means of multiple factors for generating multiple time series
- Captures more useful information than using a scalar value produced by ApEn and SampEn
- call `nk.entropy_multiscale()`
