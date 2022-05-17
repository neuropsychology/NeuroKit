**The Structure of Chaos: An Empirical Comparison of Fractal Physiology
Complexity Indices using NeuroKit2**
================

<!-- # Benchmarking and Analysis of Complexity Measures -->
<!-- # Measuring Chaos: Complexity and Fractal Physiology using NeuroKit2 -->
<!-- # Measuring Chaos with NeuroKit2: An Empirical Comparison of Fractal Physiology Complexity Indices -->
<!-- # The Structure of Chaos: An Empirical Comparison of Fractal Physiology Complexity Indices using NeuroKit2 -->

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

# Introduction

Complexity is an umbrella term for concepts derived from information
theory, chaos theory, and fractal mathematics, used to quantify
unpredictability, entropy, and/or randomness. Using these tools to
characterize signals (a subfield commonly referred to as “fractal
physiology,” Bassingthwaighte, Liebovitch, & West, 2013) has shown
promising results in physiology in the assessment and diagnostic of the
state and health of living systems Ehlers (1995).

There has been a large and accelerating increase in the number of
complexity indices in the past few decades. These new procedures are
usually mathematically well-defined and theoretically promising.
However, few empirical evidence exist to understand their differences
and similarities. Moreover, some can be very expensive in terms of
computation power and thus, time, which can become an issue in some
applications such as high sampling-rate techniques (e.g., M/EEG) or
real-time settings (brain-computer interface). As such, having a general
view depicting the relationship between the indices with information
about their computation time would be useful, for instance to guide the
indices selection in settings where time or computational power is
limited.

One of the contributing factor of this lack of empirical comparison is
the lack of free, open-source, unified, and easy to use software for
computing various complexity indices. Indeed, most of them are described
mathematically in journal articles, and reusable code is seldom made
available, which limits their further application and validation.
*NeuroKit2* (Makowski et al., 2021) is a Python package for
physiological signal processing that aims at providing the most
comprehensive, accurate and fast pure Python implementations of
complexity indices.

Leveraging this tool, the goal of this study is to empirically compare a
vast number of complexity indices, inspect how they relate to one
another, and extract some recommendations for indices selection, based
on their added-value and computational efficiency. Using *NeuroKit2*, we
will compute more than a hundred complexity indices on various types of
signals, with varying degrees of noise. We will then project the results
on a latent space through factor analysis, and report the most
interesting indices in regards to their representation of the latent
dimensions.

## Methods

![Different types of simulated signals, to which was added 5 types of
noise (violet, blue, white, pink, and brown) with different intensities.
For each signal type, the first row shows the signal with a minimal
amount of noise, and the last with a maximal amount of noise. We can see
that adding Brown noise turns the signal into a Random-walk (i.e., a
Brownian
motion).](../../studies/complexity_benchmark/figures/fig1_signals-1.png)

The script to generate the data can be found at
**github.com/neuropsychology/NeuroKit/studies/complexity_benchmark**

We started by generating 5 types of signals, one random-walk, two
oscillatory signals made (one made of harmonic frequencies that results
in a self-repeating - fractal-like - signal), and two complex signals
derived from Lorenz systems (with parameters
(![\\sigma = 10, \\beta = 2.5, \\rho = 28](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csigma%20%3D%2010%2C%20%5Cbeta%20%3D%202.5%2C%20%5Crho%20%3D%2028 "\sigma = 10, \beta = 2.5, \rho = 28"));
and
(![\\sigma = 20, \\beta = 2, \\rho = 30](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csigma%20%3D%2020%2C%20%5Cbeta%20%3D%202%2C%20%5Crho%20%3D%2030 "\sigma = 20, \beta = 2, \rho = 30")),
respectively). Each of this signal was iteratively generated at …
different lengths (). The resulting vectors were standardized and each
were added 5 types of
![(1/f)^\\beta](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%281%2Ff%29%5E%5Cbeta "(1/f)^\beta")
noise (namely violet
![\\beta=-2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D-2 "\beta=-2"),
blue
![\\beta=-1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D-1 "\beta=-1"),
white
![\\beta=0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D0 "\beta=0"),
pink
![\\beta=1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D1 "\beta=1"),
and brown
![\\beta=2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D2 "\beta=2")
noise). Each noise type was added at 48 different intensities (linearly
ranging from 0.1 to 4). Examples of generated signals are presented in
**Figure 1**.

The combination of these parameters resulted in a total of 6000 signal
iterations. For each of them, we computed 128 complexity indices, and
additionally basic metric such as the standard deviation (*SD*), the
*length* of the signal and its dominant *frequency*. The parameters used
(such as the time-delay
![\\tau](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau "\tau")
or the embedding dimension) are documented in the data generation
script. For a complete description of the various indices included,
please refer to NeuroKit’s documentation
(<https://neuropsychology.github.io/NeuroKit>).

## Results

The data analysis script, the data and the code for the figures is fully
available at **ADD LINK**. The analysis was performed in R using the
*easystats* collection of packages Makowski, Lüdecke, Ben-Shachar, &
Patil (2020/2022).

### Computation Time

Despite the relative shortness of the signals considered (a few thousand
points at most), the fully-parallelized data generation script took 12h
to run on a 48-cores machine. After summarizing and sorting the indices
by computation time, the most striking feature are the orders of
magnitude of difference between the fastest and slowest indices. Some of
them are also particularly sensitive to the data length, a property
which combined with computational expensiveness leads to indices being
100,000 slower to compute than other basic metrics.

Multiscale indices are among the slowest, due to their iterative nature
(a given index is computed multiple times on coarse-grained subseries of
the signal). Indices related to Recurrence Quantification Analysis (RQA)
are also relatively slow and don’t scale well with signal length.

## References

<div id="refs" class="references csl-bib-body hanging-indent"
line-spacing="2">

<div id="ref-bassingthwaighte2013fractal" class="csl-entry">

Bassingthwaighte, J. B., Liebovitch, L. S., & West, B. J. (2013).
*Fractal physiology*. Springer.

</div>

<div id="ref-ehlers1995chaos" class="csl-entry">

Ehlers, C. L. (1995). Chaos and complexity: Can it help us to understand
mood and behavior? *Archives of General Psychiatry*, *52*(11), 960–964.

</div>

<div id="ref-lau2021brain" class="csl-entry">

Lau, Z. J., Pham, T., Annabel, S., & Makowski, D. (2021). *Brain
entropy, fractal dimensions and predictability: A review of complexity
measures for EEG in healthy and neuropsychiatric populations*.

</div>

<div id="ref-parametersArticle" class="csl-entry">

Lüdecke, D., Ben-Shachar, M., Patil, I., & Makowski, D. (2020).
Extracting, computing and exploring the parameters of statistical models
using R. *Journal of Open Source Software*, *5*(53), 2445.
<https://doi.org/10.21105/joss.02445>

</div>

<div id="ref-seeArticle" class="csl-entry">

Lüdecke, D., Patil, I., Ben-Shachar, M. S., Wiernik, B. M., Waggoner,
P., & Makowski, D. (2021). <span class="nocase">see</span>: An R package
for visualizing statistical models. *Journal of Open Source Software*,
*6*(64), 3393. <https://doi.org/10.21105/joss.03393>

</div>

<div id="ref-correlationArticle" class="csl-entry">

Makowski, D., Ben-Shachar, M., Patil, I., & Lüdecke, D. (2020). Methods
and algorithms for correlation analysis in R. *Journal of Open Source
Software*, *5*(51), 2306. <https://doi.org/10.21105/joss.02306>

</div>

<div id="ref-modelbasedPackage" class="csl-entry">

Makowski, D., Lüdecke, D., Ben-Shachar, M. S., & Patil, I. (2022). <span
class="nocase">modelbased</span>: Estimation of model-based predictions,
contrasts and means (Version 0.7.2.1). Retrieved from
<https://CRAN.R-project.org/package=modelbased> (Original work published
2020)

</div>

<div id="ref-Makowski2021neurokit" class="csl-entry">

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F.,
Pham, H., … Chen, S. H. A. (2021). NeuroKit2: A python toolbox for
neurophysiological signal processing. *Behavior Research Methods*,
*53*(4), 1689–1696. <https://doi.org/10.3758/s13428-020-01516-y>

</div>

</div>
