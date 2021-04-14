Heart Rate Variability (HRV) Analysis Methods
======================================


Time-Domain Analysis
---------------------


**Deviation-based Indices**

- **SDNN** (ms): Standard deviation of NN intervals
- **SDANN** (ms): Standard deviation of average NN intervals extracted from 5 minute segments of time series data
- **SDNNI** (ms): Mean of the standard deviations of NN intervals per 5 minute segments of time series data

**Difference-based Indices**

- **RMSSD** (ms): Square root of the mean squared differences between successive NN intervals

.. math::
    $RMSSD = \sqrt{\frac{1}{N-1}\sum\limits_{n=1}^{N-1}({RR}_{n+1}-{RR}_{n})^2}$

- **SDSD** (ms): Standard deviation of the successive NN intervals differences
- **pNN20** (%): Proportion of successive NN intervals larger than 20ms
- **pNN50** (%): Proportion of successive NN intervals larger than 50ms


**Geometric Indices**

****

Frequency-Domain Analysis
---------------------

- Power spectrum divied into four frequency bands (units in Hz)

   - **ULF**: ultra-low frequency ( ≤0.003 Hz)
   - **VLF** very low frequency (0.0033--0.04 Hz)
   - **LF**: low frequency (0.04--0.15 Hz)
   - **HF**: high frequency (0.15--0.4 Hz)

- Power in normalized units (ms^2): 
 - **LFn**
 - **HFn**

- Natural logarithm of absolute powers of VLF, LF, and HF bands

- **LF/HF ratio**

****

Non-linear Dynamics
---------------------

**Poincaré Plot Anlysis**

The Poincaré plot is a graphical representation of each NN interval plotted against its preceding NN interval. The ellipse that emerges is a visual quantification of the correlation between successive NN intervals.

- **SD1**: Standard deviation perpendicular to the line of identity
   - Index of short-term and rapid HRV changes

- **SD2**: Standard deivation parallel to the line of identity
   - Index of long-term HRV changes

- **SD1/SD2**: ratio of *SD1* to *SD2*
   - Describes the ratio of short term to long term variations in HRV

Other indices computed based on the relationship between the short-term and long-term HRV changes are **Cardiac Sympathetic Index (CSI)**, which is a measure of cardiac sympathetic function independent of vagal activity and conversely, the **Cardiac Vagal Index (CVI)**, an index of cardiac parasympathetic function (vagal activity unaffected by sympathetic activity).

**Entropy Measures**

Entropy-based methods are measures of orderliness in contiguous events. Greater entropy in the HR signal implies that there is higher randomness and unpredictability while lower entropy implies greater regularity and predictability.

- **Approximate Entropy (ApEn)**: Logarithmic likelihood that incremental comparisons of successive NN interval differences are minimal
   - Quantify complexity based on a single time scale

- **Sample Entropy (SampEn)**
   - Quantify complexity based on a single time scale

- **Multiscale Entropy (MSE)**: 
   - The calculation methodology first involves constructing multiple coarse-grained time series, where data points are averaged in non-overlapping windows increasing in length (i.e., scale = 1, 2...). Secondly, entropy (can be *SampEn* or *ApEn*) is then computed for each coarse-grained time series by plotting its values as a function of the timescale. The area under the *MSE* curve then represents the complexity index.
   - Recent improved derivatives include composite MSE (**CMSE**) and refined composite MSE (**RCMSE**).

For a more comprehensive step-by-step guideline on the computation of *SampEn* and *ApEn*, see this `tutorial <https://www.mdpi.com/1099-4300/21/6/541>`_ and for *MSE*, see `here <http://physionet.cps.unizar.es/physiotools/mse/tutorial/tutorial.pdf>`_


**Fractal Methods**

**Detrended Fluctuation Analysis (DFA)**
  - A measure of fractal-like correlations in the HR signal
  

**Correlation Dimension (CD)**


****

NeuroKit2 *vs.* Other Packages
---------------------
*NeuroKit2* is the most comprehensive software for computing HRV indices, and the list of features is available below:

+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
| Domains           | Indices        |     NeuroKit    |     heartpy     |       HRV       |       pyHRV     |
+===================+================+=================+=================+=================+=================+
| Time Domain       |   CVNN         |        ✔️       |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   CVSD         |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |    MAD         |                 |    ✔️           |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |    MHR         |                 |                 |      ✔️         |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |    MRRI        |                 |                 |       ✔️        |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   | NNI parameters |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |ΔNNI parameters |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   MadNN        |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   MeanNN       |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   MedianNN     |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   MCVNN        |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   pNN20        |         ✔️      |       ✔️        |                 |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   pNN50        |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   RMSSD        |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SDANN        |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SDNN         |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SDNN_index   |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SDSD         |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   TINN         |         ✔️      |                 |                 |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
| Frequency Domain  |   ULF          |        ✔️       |                 |                 |        ✔️       |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   VLF          |         ✔️      |                 |       ✔️        |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   LF           |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   LFn          |         ✔️      |                 |       ✔️        |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   LF Peak      |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   LF Relative  |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   HF           |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   HFnu         |         ✔️      |                 |       ✔️        |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |  HF Peak       |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |  HF Relative   |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   LF/HF        |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
| Non-Linear Domain |   SD1          |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |  SD2           |         ✔️      |       ✔️        |          ✔️     |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   S            |         ✔️      |       ✔️        |                 |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SD1/SD2      |         ✔️      |       ✔️        |                 |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   SampEn       |         ✔️      |                 |                 |         ✔️      |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |  DFA           |                 |                 |                 |  ✔️             |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   CSI          |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   Modified CSI |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+
|                   |   CVI          |         ✔️      |                 |                 |                 |
+-------------------+----------------+-----------------+-----------------+-----------------+-----------------+



 *Note*: This table of indices will be continually updated as the different packages develop.


