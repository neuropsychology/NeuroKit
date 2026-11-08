News
=====

0.3.1
-------------------
Compatibility
+++++++++++++

* Python 3.15 is now supported.
* ``setuptools`` is now only installed with the ``full`` extra, removing its ``<82`` pin.

Changes in Results
++++++++++++++++++

* ``fractal_correlation()`` with the default ``radius``, and therefore ``HRV_CD``, can differ.
* ``expspace()`` returns different values when ``start`` is above 0.
* ``entropy_approximate()`` (and ``entropy_fuzzy(approximate=True)``, ``entropy_range(approximate=True)``,
  ``entropy_multiscale(method="MSApEn")`` and the ``complexity_tolerance(method="maxApEn")`` curve)
  now returns the signed difference Phi(m) - Phi(m+1) as defined by Pincus (1991), instead of its
  absolute value. It can therefore be slightly negative when the embedded vectors match only
  themselves (small tolerance, high dimension or short signal), where the absolute value used to
  make this degenerate estimate look like a small positive entropy.
* ``entropy_sample()`` (and ``entropy_fuzzy()``, ``entropy_range()`` and ``entropy_multiscale()``
  with ``"MMSEn"`` or ``"IMSEn"``) can differ slightly when ``delay`` is above 1: the matches at
  dimension *m* are now counted among the same templates as at *m* + 1, as defined by Richman &
  Moorman (2000). Before, the last ``delay - 1`` vectors at *m* had no counterpart at *m* + 1.
  Results with ``delay=1`` are unchanged.
* ``complexity_lempelziv(permutation=True)`` (and ``entropy_multiscale(method="LZC", permutation=True)``)
  with ``dimension`` of 4 or more now gives the correct number of words. Before, the ordinal
  patterns were joined into a string of their indices, so patterns 10 and above were written with
  two digits and parsed as two symbols, which inflated the count and the sequence length used for
  the normalization. Binary LZC (the default) is unchanged.
* ``complexity_delay()`` with the autocorrelation methods (``"theiler1990"``, ``"casdagli1991"``,
  ``"rosenstein1993"``) returned the delay plus one: the autocorrelation at lag 0 was labelled as
  delay 1. They now return the first lag at which the autocorrelation is at or below the
  threshold (1/e, 0, and 1 - 1/e of its initial value), and they honour a custom sequence of
  delays (a list ``delay_max`` was ignored).
* ``complexity_delay(method="rosenstein1994")`` now returns the delay at which the slope of the
  average displacement has decreased to 40% of its initial value, as in the paper. It used to look
  for a slope of 40 degrees on an arbitrarily scaled curve.
* ``complexity_delay(method="gautama2003")`` now uses the number of delay vectors, rather than the
  length of the signal, in the description-length penalty of the entropy ratio.
* ``complexity_dimension(method="fnn")`` now returns the first dimension at which the proportion
  of false nearest neighbours drops below 2% (or within 1.85 times its minimum when it stays
  above). It used to accept the dimension before the drop when the proportion reached exactly 0
  afterwards (e.g., 2 instead of 3 for the Lorenz attractor). Its test II now uses the same
  (Euclidean) distance in both dimensions.
* ``complexity_dimension(method="afnn")`` with a custom list of dimensions returned the index of
  the optimal dimension plus one rather than the dimension itself.
* ``complexity_tolerance(method="chon2009")`` was wrong by a factor of 4 or more (the fourth root
  of N / 1000 was computed as a division by N / 4000), used the extrapolated coefficients instead
  of the published ones for dimensions 3 to 7, and was not scaled by the SD of the signal (the
  formula gives r in units of the SD, so the same tolerance was returned whatever the amplitude
  of the signal).
* ``complexity_optimize()`` now calls ``complexity_delay()``, ``complexity_dimension()`` and
  ``complexity_tolerance()`` instead of its own copies of them, which tested one delay less than
  ``delay_max`` and mapped ``"rosenstein1993"`` to the average displacement method.
* ``fractal_petrosian()`` now uses the number of samples as *N* (Petrosian, 1995) rather than the
  number of consecutive differences, which is one less with the default ``symbolize="C"`` (and
  ``"D"`` and ``"R"``). The change is small (about 1e-4 for 500 samples) but systematic.
* ``complexity_symbolize(method="C")`` (used by default in ``fractal_petrosian()``) no longer
  treats a zero difference as an increase: a flat segment takes the sign of the previous nonzero
  difference, so it is not counted as a sign change. Before, a flat segment within a decrease
  counted as two sign changes. This only affects signals with repeated consecutive values (e.g.,
  quantized or low-resolution recordings).
* ``fractal_dfa()`` with ``overlap=True`` (the default, and therefore the ``DFA`` and ``MFDFA``
  indices of ``hrv_nonlinear()`` and ``rsp_rrv()``) now includes the last window that fits in the
  signal. Before, the window starting at ``len(signal) - window`` was left out, so the last
  samples were never used (e.g., the last 25 of 2,000 samples with windows of 50). The effect on
  alpha is small (up to about 0.01) but systematic. Results with ``overlap=False`` are unchanged.
* ``fractal_dfa()`` and ``fractal_hurst()`` with their default ``scale`` (and therefore
  ``complexity()`` and ``fractal_tmf()``; the ``hrv_nonlinear()`` and ``rsp_rrv()`` indices set
  their own scales) now use evenly log-spaced scales, about 10% apart (e.g., 25 scales instead of
  70 for 1,000 samples, 73 instead of 4,236 for 100,000). Before, the number of scales was a tenth
  of the signal length, and the scales were truncated to whole samples and deduplicated, which
  left them crowded at the large scales (about 90% in the upper half of the log range for long
  signals), where the fluctuations are the noisiest. This contradicted the documented equal
  weighting. The new scales reduce the standard deviation of DFA alpha and of the Hurst exponent
  by 10 to 15% for 2,000 samples and by 30% for 10,000 samples (white and fractional Gaussian
  noise), with similar means (within 0.01). When ``scale`` is an integer (the number of scales),
  the scales are now rounded rather than truncated to whole samples.
* ``fractal_dfa(multifractal=True)`` (``fractal_mfdfa()``), and therefore the ``MFDFA`` indices of
  ``hrv_nonlinear()``, ``rsp_rrv()`` and ``complexity()``, and ``fractal_tmf()``:

  * Detrended segments are now discarded only when their variance is numerically zero (at the
    rounding error of the signal, e.g., runs of identical values), instead of when it is below
    1e-8 times the largest variance at that scale. That cutoff discarded genuinely small
    fluctuations, which dominate the negative ``q`` moments: for the binomial cascade of
    Kantelhardt et al. (2002), h(-5) was 1.18 instead of 1.74 (theory: 1.80). Results only change
    for signals whose segment variances span more than 8 orders of magnitude, and for signals
    whose segments are all exactly linear at some scales (h was then fitted on rounding noise,
    e.g., about 20). Such scales are now left out of the fit.
  * ``Asymmetry`` is now the asymmetric ratio of Orozco-Duque et al. (2015, eq. 16),
    (H(q_min) - Peak) / (H(q_min) - H(q_max)), the share of the spectrum's width on the side of
    the negative ``q``, which is 0.5 for a symmetric spectrum as documented. It was
    (min(H) - Peak) / Width, between -1 and 0 (the new value is the old one plus 1 when H
    decreases with q).
  * ``Fluctuation`` (the h-fluctuation index of Orozco-Duque et al., 2015, eq. 17) is now the sum
    of the squared second differences of h(q) between consecutive ``q``, divided by
    2 * max(abs(q)) + 2. It applied ``np.gradient()`` twice, i.e., second differences over steps
    of 2 ``q`` divided by 4, with one-sided differences at the ends. With unevenly spaced ``q``,
    the second derivative of h(q) is now estimated from its spacing.
  * ``Increment`` (Faini et al., 2021) is now the sum of the squared differences of h(q) between
    consecutive ``q`` (divided by the step of ``q``, which is 1 in the HRV and RRV indices). It was
    computed from central differences (``np.gradient()``), which skip every other ``q`` and count
    the end points twice.
  * The singularity exponents ``H`` now take the spacing of ``q`` into account
    (``np.gradient(Tau, q)``). With unevenly spaced ``q`` (e.g., the default
    ``[-5, -3, -1, 0, 1, 3, 5]``), this changes ``H`` and ``D`` at q = -1 and 1, but none of the
    returned indices.

* ``fractal_higuchi()`` and ``complexity_k()`` now return NaN for signals with missing values.
  Before, the increments touching a missing value were silently left out of the curve lengths,
  while their normalization still counted them.
* The ``"Intercept"`` of ``fractal_higuchi()`` and the ``"Intercepts"`` of ``complexity_k()`` are
  now those of the fitted line (log L(k) against log(1/k)); their sign was flipped before.
* ``complexity_k()``, and therefore ``fractal_higuchi()`` with its default ``k_max="default"``,
  now selects the first k_max from which the HFD stays within 0.01 of its value for the next 10
  tested k_max values. Before, ``find_plateau()`` kept the bend of the HFD curve with the highest
  HFD among the 10% sharpest ones (and returned the k_max just before it): the selected k_max was
  erratic (e.g., from 2 to 118 for 10 white noises of 500 samples, now from 4 to 14) and grew with
  the length of the signal (a median of 122 for pink noise of 500 samples and 756 for 2,000, now
  23 and 19). A list of k_max values is now tested in increasing order. ``find_plateau()`` itself
  is unchanged.
* ``fractal_hurst()`` now uses the population standard deviation in the R/S statistic, for which
  the expected value of Anis & Lloyd (1976), subtracted with ``corrected=True`` (the default), is
  exact. With the sample (n - 1) SD, the corrected exponent was biased upward (e.g., 0.525 instead
  of 0.50 for white noise of 500 samples, 0.513 for 2,000 samples). The expected value is now also
  computed exactly for windows above 340 samples (an asymptotic approximation was used).
  Uncorrected exponents are slightly lower.
* ``fractal_sda()`` split the signal into ``scale`` bins (of n / ``scale`` points each) instead of
  bins of ``scale`` points, so the FD was around 0.6-0.75 whatever the signal (e.g., 0.66 for white
  noise and 0.71 for fractional Gaussian noise with H = 0.7, for 2,000 samples). It now gives 1.54
  and 1.37 (2 - H: 1.5 and 1.3). The SD of the bin means now uses the sample (n - 1) formula, as in
  Hasselman's *casnet*, which reduces the bias of the largest scales (with only 2 bins).
* ``fractal_psdslope()`` with the default ``method="voss1988"`` computed FD = (5 - beta) / 2 with
  the slope instead of beta (the PSD decays as 1/f^beta, so beta = -slope): e.g., 3.5 instead of
  1.5 for a Brownian motion. The fit now also includes the lowest frequencies (``signal_psd()``'s
  default ``min_frequency`` dropped the first three), which reduces the SD of the slope by about
  10% (this also affects ``method="hasselman2013"``).
* ``fractal_nld()`` now normalizes the signal as Kalauzi et al. (2009, eqs. 8-10): centered and
  divided by its mean absolute deviation (instead of its standard deviation), with the sum of the
  absolute differences divided by the number of samples (instead of the number of differences).
  The calibration curve of ``corrected=True`` was fitted on this definition: it now gives an FD of
  2.00 for white noise (30-sample windows) instead of 1.92. Raw NLD values are about 1.25 times
  larger for Gaussian-like signals.
* ``fractal_tmf()`` had its sign flipped: ``scipy.stats.ttest_1samp(surrogates, original)``
  returns (mean(surrogates) - original) / SE, while tMF is (original - mean(surrogates)) / SE
  (Bell et al., 2019), positive when the signal is more multifractal than its linear surrogates.
  Its ``**kwargs`` are now passed on to ``fractal_dfa()``, as documented (they were ignored).
* ``fractal_linelength()`` (and the ``LL`` index of ``complexity()``) no longer bridges missing
  values: the differences involving a NaN are ignored, instead of dropping the NaNs and taking the
  difference between the samples around the gap. Results without missing values are unchanged.
* ``fractal_density()`` now uses its ``tolerance`` argument (it always used 0.2 SD) and accepts
  pandas Series.
* ``entropy_bubble()`` (and the ``BubbEn`` index of ``complexity()``) now uses the Rényi entropy
  of order ``alpha`` (2 by default, as in Manis et al., 2017). The ``alpha`` argument was not
  passed on, so the Shannon entropy was used whatever its value (e.g., 0.656 instead of 0.582 for
  white noise with ``dimension=3``).
* ``entropy_increment()`` (and ``entropy_multiscale(method="MSIncrEn")``) now keeps the sign and
  the size of each increment as two letters of its word, as in Liu et al. (2016). They were
  multiplied, which merged a rise or a decline of size 0 with no change: values are higher (e.g.,
  4.06 instead of 3.86 bits for white noise with ``dimension=3`` and ``q=4``), the worked example
  of the paper now gives its 3 bits (instead of 2.75), and ``q=0`` gives the entropy of the signs
  of the increments, as described in the paper, instead of always 0.
* ``entropy_phase()`` is now between 0 and 1, as defined by Rohila & Sharma (2019): the entropy in
  bits was divided by the natural logarithm of ``k``, which made every value 1.4427 (1 / ln(2))
  times too large (e.g., 1.166 instead of 0.808 for white noise with ``k=4``). The points on the
  axes of the second order difference plot (e.g., two equal successive RR intervals in ms) and on
  the boundaries between sectors are now counted; they were left out of all the sectors.
* ``entropy_differential()`` now uses the default estimator of ``scipy.stats.differential_entropy()``
  (``method="auto"``), instead of always ``"vasicek"``, which is more biased for short signals
  (e.g., -0.13 instead of -0.04 bits on average for 100 samples of a normal distribution). Results
  are unchanged for more than 1000 samples.
* ``entropy_attention()`` (and the ``AttEn`` index of ``complexity()``) is now in bits by default,
  like ``entropy_shannon()`` and the other Shannon-based entropies, so values are 1.4427
  (1 / ln(2)) times larger (e.g., 1.657 instead of 1.149 for white noise); the new ``base``
  argument gives the previous values with ``base=np.e``. With arguments for
  ``scipy.signal.find_peaks()`` (e.g., ``distance`` or ``height``), which make the maxima and
  minima no longer alternate, the MaxMin and MinMax intervals now go from each key pattern to the
  next one of the other type. Before, every other interval between successive key patterns was
  taken, which mixed in MaxMax and MinMin intervals (e.g., an AttEn of 3.01 bits, instead of 2.64,
  for white noise with ``height=0.5``). AttEn and its sub-entropies are now NaN (with a warning)
  when a type of interval is missing, e.g., with fewer than two maxima; they used to be 0, which
  lowered the average.
* ``entropy_multiscale()`` with ``"CMSEn"``, ``"RCMSEn"``, ``"CMSPEn"`` and ``"CMSWPEn"`` (and
  therefore the ``CMSEn`` and ``RCMSEn`` indices of ``hrv_nonlinear()``), and
  ``complexity_coarsegraining(method="timeshift")``, now average the non-overlapping windows that
  start at each of the Tau offsets, as defined by Wu et al. (2013). The "time-shift"
  coarse-graining used to keep every Tau-th sample from each offset, without averaging, so there
  was no low-pass filtering and the composite entropies stayed flat (or rose) with the scale
  instead of decreasing like MSEn (e.g., 2.17 instead of 1.10 at scale 10 for white noise with
  ``dimension=2``) (#1071). ``complexity_coarsegraining(method="timeshift")`` returns a list of the
  Tau series (their lengths can differ by one) instead of a 2D array.
* ``entropy_multiscale(method="RCMSEn")`` now sums the numbers of template matches of the Tau
  series before taking their ratio, as in Wu et al. (2014). It used to average their proportions,
  which is the same only when the series have the same length (#1071). ``fuzzy=True`` (and
  ``complexity_fuzzyrcmse()``) now applies at every scale; it was only applied at scale 1.
* ``entropy_multiscale(method="CMSEn")`` is now undefined (infinite or NaN) at a scale where the
  entropy of one of its Tau series is undefined, as defined by Wu et al. (2013). The undefined
  series used to be silently left out of the average, which biased the value downwards (the
  refined ``"RCMSEn"`` exists to handle these cases) (#1071).
* ``entropy_multiscale()`` (every method, and therefore the ``MSEn``, ``CMSEn`` and ``RCMSEn``
  indices of ``hrv_nonlinear()`` and ``MSPEn`` of ``complexity()``) now uses by default the scale
  factors 1 to 20 (Costa et al., 2005), capped so that the coarse-grained series at the largest
  scale keeps at least 100 samples (``min(20, len(signal) // 100)`` scales, and scale 1 alone
  below 200 samples). The default used to go up to ``len(signal) / (dimension + 10)``, i.e., until
  the coarse-grained series had only about 12 samples whatever the length of the signal: for 300
  RR intervals (a 5-minute recording) it gave 24 scales, of which those above 10 were undefined in
  25% to 80% of the realizations of 1/f noise and the others scattered by about 30%; for 2,000
  samples with the default ``dimension=3``, 152 scales. They are now 3 and 20. The sample entropy
  is reasonably stable from about 100 samples (Richman & Moorman, 2000; Yentes et al., 2013). The
  previous scales can be obtained with ``scale=int(len(signal) / (dimension + 10)) - 1``. The
  default is also much faster to compute.
* The index returned by ``entropy_multiscale()`` (every method, and therefore the ``MSEn``,
  ``CMSEn`` and ``RCMSEn`` indices of ``hrv_nonlinear()`` and ``MSPEn`` of ``complexity()``) and
  by ``entropy_hierarchical()`` is now the area under the curve of the entropy values against the
  scale factors divided by the range of the scales, i.e., the average height of the curve. It was
  the area under the values (assumed evenly spaced) divided by their number, which was 0 for a
  single scale, c × (n - 1) / n for a flat curve at c over n scales (e.g., 1.5 for 4 values of 2),
  and ignored the spacing of uneven scales or of the scales left out as undefined. For the default
  (consecutive) scales, values are now n / (n - 1) times larger, e.g., by about 1% for the 82
  scales of 1,000 samples with ``dimension=2`` and 4% for the 24 scales of 300 RR intervals.
* ``entropy_shannon_joint()`` now returns the joint entropy H(X, Y), i.e., the Shannon entropy of
  the pairs of values (x_i, y_i) (Cover & Thomas, 2006), as documented. It returned the
  Kullback-Leibler divergence between the distributions of ``x`` and of ``y``
  (``scipy.stats.entropy(p_x, p_y)``), which ignores how the two sequences are paired: e.g.,
  1.459 instead of 0.119 bits for its docstring example, and H(X) instead of 0 for ``x`` with
  itself. ``x`` and ``y`` can now take different values (they had to have the same labels, checked
  with an ``assert``), and must have the same length (sequences of different lengths were
  accepted).
* ``entropy_cumulativeresidual()`` (and its alias ``complexity_cren()``) now uses the survival
  function P(X > x), as defined by Rao et al. (2004) and given in its docstring, instead of the
  cumulative distribution P(X <= x), which gives the "cumulative entropy" (the CREn of -X). Values
  change for asymmetric distributions, e.g., 1.432 instead of 0.921 bits for 10,000 samples of an
  exponential distribution of mean 1 (whose CREn is 1 nat, 1.4427 bits), or 0.5 instead of 0.311
  for ``[0, 0, 0, 1]``; they are unchanged for symmetric ones (e.g., its docstring example) and
  close for roughly symmetric ones (1.282 instead of 1.295 for 1,000 samples of white noise).
  Signals with missing values now return NaN: the missing values were counted in the
  probabilities, but their interval was left out of the sum.
* ``entropy_kl()`` with the default ``norm="euclidean"`` (and therefore
  ``complexity_delay(method="gautama2003")``) used the volume of the Euclidean ball of radius 1
  instead of diameter 1 in the Kozachenko-Leonenko estimator (Kraskov et al., 2004), so KLEn was
  ``dimension`` × log(2) nats too high (e.g., 4.28 instead of 2.90 nats for 1,000 samples of white
  noise with ``dimension=2``, whose differential entropy is 2.84 nats). ``norm="max"`` was correct
  and is unchanged. As ``complexity_delay(method="gautama2003")`` takes the ratio of the KLEn of
  the signal and of its surrogates, the delay and dimension it selects can change.
* ``entropy_kl()`` now leaves the delay vectors that have an identical neighbour (at a distance of
  0, e.g., in quantized signals) out of the average of the log-distances. They were left out of
  the sum but still counted in the number of vectors, i.e., as a distance of 1 in the units of the
  signal, so that the result depended on these units: e.g., for 300 RR intervals in integer ms
  with ``dimension=1``, converting them to seconds changed KLEn by 1.17 instead of log(1000) =
  6.91. Results for signals without repeated delay vectors are unchanged (apart from the
  Euclidean constant above). KLEn is now NaN, with a warning, for a constant signal (it returned
  6.31 for 100 samples with ``dimension=2``) or a single delay vector (it returned infinity).
* ``entropy_renyi()`` and ``entropy_tsallis()`` given a ``freq`` that contains zero
  probabilities now leave out the states that do not occur. Before, the default ``alpha=1`` and
  ``q=1``, and ``q`` below 1, returned NaN (0 × log(0)), and ``alpha=0`` (Hartley entropy) counted
  them (e.g., log(3) instead of log(2) for ``freq=[0.5, 0.5, 0]``). This can affect
  ``entropy_permutation(weighted=True)`` with ``algorithm=nk.entropy_renyi`` or
  ``nk.entropy_tsallis`` when a pattern only occurs in constant vectors (weight 0). Results
  computed from a signal are unchanged (all its states occur).
* ``entropy_shannon()`` (and its alias ``complexity_se()``), ``entropy_renyi()``,
  ``entropy_tsallis()`` and ``entropy_cumulativeresidual()`` now pass ``c`` and ``random_state``
  on to ``complexity_symbolize()``. They were silently ignored, so that e.g.
  ``entropy_shannon(signal, symbolize="MEP", c=5)`` used 3 symbols (1.585 instead of 2.322 bits for
  white noise). Results without these arguments are unchanged, including those of the indices
  that use ``entropy_shannon()`` (e.g., ``ShanEn`` of ``hrv_nonlinear()``, ``AttEn`` and
  ``MSPEn`` of ``complexity()``) and ``entropy_renyi()`` (``BubbEn``).
* ``entropy_maximum()`` now treats a string as a sequence of characters, as
  ``entropy_shannon()`` does: it returned 0 for ``"ABBA"`` (one string), below the 1 bit of
  ``entropy_shannon("ABBA")`` that it is meant to bound. It now raises an error for
  multidimensional inputs, like ``entropy_shannon()`` (their values were pooled).
* ``entropy_dispersion()`` is now between 0 and 1, as defined by Rostaghi & Azami (2016): the
  entropy in bits was divided by the natural logarithm of the number of patterns, which made every
  value 1.4427 (1 / ln(2)) times too large (e.g., 0.976 instead of 1.409 for white noise with the
  default parameters, and 0.790 instead of 1.139 for the worked example of Azami & Escudero, 2018,
  whose DispEn is 1.7351 nats, i.e., 0.790 once divided by ln(9)). With ``fluctuation=True``, it
  is now normalized by the logarithm of the (2c - 1)^(m - 1) possible fluctuation patterns, as in
  Azami & Escudero (2018), instead of c^m (e.g., 0.891 instead of 1.147 for white noise with the
  default parameters, and 0.710 instead of 1.082 for the worked example of the paper). The
  Reverse Dispersion Entropy (``info["RDEn"]``) now sums over all the possible patterns, including
  those that do not occur, as defined by Li et al. (2019), and uses the (2c - 1)^(m - 1)
  fluctuation patterns with ``fluctuation=True``: it is higher when some patterns are missing
  (e.g., 0.083 instead of 0.042 for the worked example of DispEn, and 0.0048 instead of 0.0030 for
  200 samples of white noise with the default parameters).
* ``entropy_dispersion(symbolize="finesort")`` now computes the fine-sorting factor from the
  differences between the NCDF values of each vector, as defined by Fu et al. (2019). It was
  computed from the classes of the vector, i.e., from its pattern itself, so it never split a
  pattern and the fine-sorted DispEn was always equal to DispEn (e.g., for white noise with
  ``c=3`` and ``rho=1``, it is now 1.042, against 0.996 for DispEn). With ``fluctuation=True``, the
  factor is now added after the differences between the classes, instead of being differenced
  with the last class.
* ``entropy_permutation()`` with ``corrected=True`` (the default), and therefore
  ``complexity_pe()``, ``complexity_wpe()`` and the PEn methods of ``entropy_multiscale()``
  (``"MSPEn"``, ``"CMSPEn"``, ``"MMSPEn"``, ``"IMSPEn"`` and their ``"WPEn"`` counterparts), is now
  divided by the maximum entropy in the unit of the entropy that is computed. It was always divided
  by log2(m!), which is only right for the default Shannon entropy in bits, so that the values were
  wrong with another logarithmic ``base`` (e.g., 0.408 instead of 0.589 for the example of Bandt &
  Pompe, 2002, with ``base=np.e``), with ``algorithm=nk.entropy_renyi`` (in nats: 0.691 instead of
  0.997 for white noise with ``alpha=2``) or with ``algorithm=nk.entropy_tsallis`` (0.322 instead
  of 0.999 with ``q=2``). Results with the default Shannon entropy in bits are unchanged (for CPEn,
  see the next bullet).
* ``entropy_permutation(conditional=True)`` (CPEn), and therefore the ``CWPEn`` index of
  ``complexity()``, is now the conditional entropy that it is documented to be, between 0 and 1
  with ``corrected=True`` (the default). The correction divided it by log((m + 1)!), the maximum
  of the PEn of dimension m + 1, instead of its own maximum, log(m + 1) (the next value can take
  m + 1 ranks), so that it was at most 0.613 for ``dimension=2`` (the default of ``complexity()``)
  and 0.436 for ``dimension=3``. The entropy of the patterns of m values is now computed from the
  first m values of the vectors of m + 1 values and, for the weighted CPEn, with their weights (the
  variance of the m + 1 values). It was computed from all the vectors of m values, weighted by the
  variance of their m values, so that the difference was not a conditional entropy and could be
  negative (e.g., -0.082 bits for ``[4, 1, 3, 0, 2]`` with ``dimension=2``, whose patterns of 2
  values each determine the rank of the third value: it is now 0). For 1000 samples of white
  noise, CPEn is now 0.995 instead of 0.434 with ``dimension=3``, and ``CWPEn`` 0.990 instead of
  0.607 (0.019 instead of 0.012 for 10 s of ``ecg_simulate(sampling_rate=250, random_state=1)``).
  Without weights, the uncorrected CPEn only changes for short signals (e.g., 0.8 instead of 0.604
  bits for the 7 samples of the example of Bandt & Pompe, 2002, with ``dimension=2``, and 0.2147
  instead of 0.2148 bits for the ECG with ``dimension=3``), but it changes more with weights (0.088
  instead of 0.136 bits for the ECG). With ``algorithm=nk.entropy_renyi``, CPEn is divided by
  log(m + 1) in nats (0.991 instead of 0.432 for white noise with ``alpha=2`` and
  ``dimension=3``), and with ``algorithm=nk.entropy_tsallis`` by the maximum of the difference,
  ln_q(m + 1) max(1, (m!)^(1 - q)), which white noise does not reach for q above 1 (0.167 instead
  of 0.131 with ``q=2``, and 0.996 instead of 0.626 with ``q=0.5``).
* ``entropy_slope()`` (and ``entropy_multiscale(method="MSSlopEn")``) now uses all the vectors of
  the signal: the last one was left out (e.g., 3.539 instead of 3.536 bits for the 31 vectors of
  the worked example of Cuesta-Frau, 2019). With a ``delay`` above 1, the vectors are now made of
  samples spaced by the delay, as in the paper (where a delay is a downsampling), instead of
  successive differences between samples spaced by the delay (e.g., 3.875 instead of 4.033 bits for
  white noise with ``delay=2``). A single threshold now separates the ties from the rises and
  declines; all the slopes were mapped to the same symbol, which always gave 0.
* ``entropy_symbolicdynamic()`` (and ``entropy_multiscale()`` with ``"MSSyDyEn"`` and
  ``"MMSyDyEn"``) now computes the state transition entropy as the entropy of the symbol that
  follows each state conditional on that state, -ΣΣ P(q) P(σ|q) ln P(σ|q) (the definition of Li
  et al., 2017, as corrected by Wei et al., 2023), so that SyDyEn is the joint entropy of the
  states and of their transitions, and its normalization by ln(c^(m + 1)) bounds it between 0 and
  1 (1 for white noise). The transition term was -ΣΣ P(q) ln(P(q) P(σ|q)) (as in EntropyHub),
  which is not an entropy and grew with the number of different states and transitions (e.g.,
  0.902 instead of 4.117 for 1,000 samples of white noise with the default parameters). The normalization uses the number of possible symbols, which is 2 for the binary
  symbolizations (e.g., ``symbolize="A"``: 0.999 instead of 0.580 for white noise) and the number
  of bins when ``symbolize`` is an integer, instead of ``c``. The state transitions of all the
  vectors that are followed by a symbol (up to the sample N - m × delay) are now counted; those of
  the last (m - 1) × delay vectors were left out, which mattered with large delays, and therefore
  at the large scales of ``"MMSyDyEn"``. With symbolizations whose symbols start at 0 (e.g.,
  ``symbolize="A"``, ``"binning"`` or a number of bins), the transitions to the symbol 0 are now
  counted; they were dropped (``symbolize="binning"`` now gives the same values as the equal-width
  bins of ``"linear"``).
* ``entropy_coalition(method="synchrony")`` (SCE) is now reproducible: the random binary sequences
  used for its normalization were drawn without a seed, so that every call gave a different value
  (e.g., 0.8533, 0.8539 and 0.8531 for the same signals). They are now drawn from the new
  ``random_state`` argument (30 by default).
* ``entropy_power()`` now returns the entropy power exp(2H) / (2 pi e) of Shannon (1948) and Dembo
  et al. (1991), i.e., the variance of the Gaussian variable with the same differential entropy H,
  in the squared units of the signal. 2H was divided by the number of samples (the *n* of the
  formula is the dimension of the random vector, 1 for a signal), so that PowEn was about
  1 / (2 pi e) = 0.0586 whatever the signal (e.g., 0.0586 instead of 1.035 for 10,000 samples of
  a standard normal distribution, and instead of 102.8 with an SD of 10). The kernel density
  estimate is also normalized to integrate to 1 over the range where it is evaluated (from the
  minimum to the maximum of the signal), so that multiplying the signal by *a* multiplies PowEn by
  exactly a^2: the mass of the tails missing from this range made the entropy depend on the units
  (e.g., PowEn was on average 12% lower than expected after multiplying 100 samples of noise by 10).
  The ``"Density"`` returned in ``info`` is this normalized density.
* ``entropy_angular()`` now computes the ComEDA index of Nardelli et al. (2022) as in the code
  released by the authors, which it claims to implement: the signal is normalized to [0, 1]
  before the embedding, the angular distance between two embedded vectors is the cosine of their
  angle (as in the paper) rather than the angle itself, and the PDF of the distances is estimated
  with the diffusion kernel density estimator of Botev et al. (2010) on B points (the Sturges
  number of bins rounded up to a power of 2) spread over the range of the distances extended by
  half of it on each side. Before, the raw signal was embedded, so that AngEn depended on its
  offset (e.g., 0.98 for white noise, but 0.68 after adding 10), the PDF was a Gaussian kernel
  estimate with an ad hoc bandwidth on bins spanning only the range of the angles, and the
  quadratic Rényi entropy was divided by the log of the number of bin edges (one more than the
  number of bins), so it could not reach 1. Values are therefore different and lower (e.g., 0.32
  instead of 0.98 for white noise of 1,000 samples with the default ``dimension=2``, and 0.37
  instead of 0.92 with ``dimension=3``). The ``"bins"`` of the returned ``info`` are now the B
  points at which the PDF is estimated (instead of the B + 1 bin edges).
* ``entropy_grid()`` now divides both axes of the Poincaré plot into the same *k* sections of the
  [0, 1] range of the normalized signal (Yan et al., 2019), as drawn with ``show=True``. Before,
  each axis was divided over the range of its own values (the present or the delayed ones),
  which is narrower when the minimum or the maximum of the signal is among its first or last
  ``delay`` samples (e.g., a GridEn of 2.183 instead of 2.322 bits, and a GDR of 0.778 instead of
  0.889, for 100 samples of white noise whose first sample is the maximum). The values that fall
  exactly on a grid line (e.g., RR intervals in integer milliseconds) are now always counted in
  the section above it; rounding errors in the bin edges could put them in the section below
  (with ``k=5`` or ``k=10``, about 18% of 5-minute series of RR intervals in ms changed, by about
  0.01 bits and up to 0.06; the default ``k=3`` was not affected).
* ``complexity_lyapunov()`` with ``separation="auto"`` (the default, for all the methods) now
  computes the mean period from the whole power spectrum, as in Rosenstein et al. (1993). The
  spectrum went through ``signal_psd()``, whose default ``min_frequency`` left out the three lowest
  nonzero frequencies, which dominate for signals with slow fluctuations, so the separation was too
  short: e.g., 171 instead of 135 samples for the x-component of the Lorenz system sampled every
  0.01 time unit, and 291 instead of 98 for a random walk of 2,000 samples. It is unchanged for
  white noise, chaotic maps or an ECG (e.g., 4 for the logistic map).
* ``complexity_lyapunov()`` now uses ``method="makowski"`` by default (it was
  ``"rosenstein1993"``), which chooses the part of the divergence curve to fit automatically (see
  below). Rosenstein's method, with its default ``len_trajectory=20``, depends on the length of the
  signal and on its sampling rate: e.g., 0.43 to 0.81 times the LLE of chaotic maps (from 500 to
  20,000 samples) and 2.5 times that of the Lorenz system, against 0.98 to 1.00 and 0.94 times
  with ``"makowski"``. Use ``method="rosenstein1993"`` to get the previous results.
* ``complexity_lyapunov(method="makowski")`` now chooses the part of the divergence curve to which
  the line is fitted from the local slope of the curve. The trajectories are followed until the
  curve levels off (at the level of unrelated vectors, or when it stops rising, e.g., for
  quasi-periodic signals) instead of for 10 × ``delay`` samples, and the line is fitted to the
  first of the longest runs of steps (from step 1) over which the local slope varies by less than
  a factor of 1.5, instead of from the first step to the knee of the curve (found with
  ``find_knee()``). This leaves out the initial transient of flows and the saturation, so that the
  LLE no longer depends on the length of the signal or on its sampling rate: e.g., 0.92 instead of
  2.42 times the LLE of the Lorenz system and 0.90 instead of 1.40 for the Rössler system (medians
  over sampling steps of 0.005 to 0.05 and 0.05 to 0.2 time units, with 5,000 samples), 0.99 to
  1.00 instead of 0.93 to 0.96 for chaotic maps (500 to 5,000 samples), and 0.24 and 0.25 per
  second instead of 8.9 and 10.5 for a simulated respiration signal of 120 s sampled at 50 and
  200 Hz. The nearest neighbour of each vector is now found among the vectors more than
  ``separation`` samples away in time, as in Rosenstein's method: it used to be the
  (``separation`` + 1)-th nearest neighbour whatever its distance in time (e.g., the 136th for the
  Lorenz system, 9% of which were within the separation window). It returns NaN, with a warning,
  when the curve has no linear part because the neighbours become unrelated within a few steps,
  e.g., for white noise (0.32 per sample before), and the slope of the curve, close to 0 (with a
  warning), when the trajectories do not diverge, e.g., for periodic signals with or without
  noise. With 5% of observational noise, the LLE of chaotic maps is underestimated by about 25%
  (0.76 instead of 0.82 before). ``max_length`` (by default, 20 times the ``separation``) is now
  an upper bound, and can be any integer of at least 2 (it had to be at least 6). In ``info``, the
  divergence curve now starts at step 0, and ``"Changepoint"`` is replaced by ``"Fit_Range"`` (the
  first and last steps of the fit), with ``"R2"``, ``"Plateau"`` (the level at which the curve
  levels off) and ``"Pairs"`` (the number of pairs of neighbours at each step); ``show=True``
  shades the fit range.
* ``complexity_lyapunov(method="eckmann1986")`` now uses the ``delay``: the neighbours are found
  among the delay vectors, and the tangent maps send each vector ``m * delay`` samples ahead (with
  m = (``dimension`` - 1) / (``matrix_dim`` - 1)). The delay was ignored (always 1), so oversampled
  signals were embedded over a few samples: e.g., for the Lorenz system (sampled every 0.01 time
  unit, whose exponents are 0.906, 0 and -14.57 per time unit) with ``delay=10`` and
  ``dimension=4``, the LEs are now 0.96, 0.07, -3.19 and -9.72 per time unit instead of 1.35,
  -0.08, -6.48 and -11.96, the values for ``delay=1`` (and 0.86, -0.01 and -11.07 with
  ``dimension=3``). Results with ``delay=1`` and ``dimension`` = ``matrix_dim`` (e.g., 4) are
  unchanged (apart from the separation above).
* ``complexity_lyapunov(method="eckmann1986")`` with ``dimension`` larger than ``matrix_dim``
  (e.g., ``dimension=7`` with the matrix dimension of 4, i.e., m = 2) now multiplies the tangent
  maps every m steps along the trajectory (T_1, T_1+m, T_1+2m, ..., as in Eckmann et al., 1986;
  the m interleaved products, starting at each of the first m vectors, are averaged): each tangent
  map sends a vector m steps ahead, but they were multiplied at every step, i.e., each one was
  applied to the frame of the vector m - 1 steps further. E.g., 1.04, 0.06, -6.1 and -12.07
  instead of 3.17, 1.21, -8.37 and -13.08 per time unit for the Lorenz system with
  ``dimension=7`` and ``delay=1``; for the Hénon map
  (exponents of 0.419 and -1.62, plus spurious ones) with ``dimension=5``, ``matrix_dim=3`` and
  ``separation=0``, 0.837, 0.416 and -1.568 (0.999, 0.328 and -1.642 with nolds' ``lyap_e()``,
  which multiplies them at every step too).
* ``complexity_lyapunov(method="rosenstein1993")`` (the previous default) now computes the distances
  between the vectors exactly. They were computed with the dot-product formula of
  ``sklearn.metrics.pairwise.euclidean_distances()``, whose rounding error is about 1e-8 times the
  amplitude of the signal. Values are unchanged (to about 1e-10), except for noise-free signals
  that repeat themselves exactly (e.g., a sine whose period is a whole number of samples), whose
  neighbours are copies at the rounding error, so that the estimate is meaningless either way
  (e.g., 0.024 instead of -0.0009 per sample for a sine with a period of 50 samples).
* ``complexity_ordinalpatterns()``, and therefore ``entropy_permutation()`` (and the ``CWPEn``
  index of ``complexity()`` and the PEn methods of ``entropy_multiscale()``) and
  ``complexity_lempelziv(permutation=True)``, now ranks the equal values of a vector (ties) in
  their order of appearance, as defined by Bandt & Pompe (2002). The vectors were sorted with
  numpy's ``"quicksort"``, which does not keep the order of the ties for vectors of 4 values or
  more (with numpy 2), and whose order of the ties can depend on the CPU: e.g., for 1,000
  intervals of about 800 ms rounded to 10 ms, 28 of the 316 vectors with ties had a different
  pattern with ``dimension=4``, and 172 of 621 with ``dimension=6`` (an uncorrected PEn of 8.853
  instead of 8.805 bits). Results for signals without ties are unchanged.
* ``fishershannon_information()`` now computes the Fisher information with the derivative of the
  density with respect to the values of the signal. ``np.gradient()`` was given no spacing, so the
  derivative was taken with respect to the index of the 100 points at which the density is
  evaluated, which made the Fisher information about the squared spacing of these points too
  small (e.g., 0.0055 instead of 0.97 for 5,000 samples of a standard normal distribution, whose
  Fisher information is 1). Together with the fix of ``entropy_power()``, the Fisher-Shannon
  complexity is now about 1 for a normal distribution (1.01 instead of 0.00027 for 2,000 samples),
  its theoretical minimum, and above for other distributions (e.g., 1.36 for a Laplace
  distribution).
* ``entropy_range()`` (RangEn B, the default ``approximate=False``) no longer removes the
  self-matches twice: the range distance of a vector with itself is undefined (0 / 0), so the
  vector was already left out of its own count, from which the SampEn computation then removed
  one more match. Values are lower and no longer depend on the length of the signal as much
  (e.g., on average 1.52 instead of 1.58 for 1,000 samples of white noise, and 1.52 instead of
  1.77 for 300 samples, with ``tolerance=0.2``), and match the definition of Omidvarnia et al.
  (2018).
* ``entropy_range()`` (RangEn A and B) no longer multiplies the tolerance by the SD of the signal:
  the range distance is normalized (between 0 and 1), so the default ``tolerance="sd"`` now gives
  0.2 (Omidvarnia et al., 2018) whatever the amplitude of the signal, and the other methods of
  ``complexity_tolerance()`` are applied to the standardized signal. RangEn used to depend on the
  amplitude of the signal, which it is designed not to: e.g., for RR intervals in ms (SD of 50 ms)
  the tolerance was 10, all the vectors matched and RangEn was 0 (it is now 1.52 for such a
  random series of 300 intervals). Results for signals with an SD of 1 only change as described in
  the previous point. Two vectors now also match when their range distance equals the tolerance
  (d <= r, Eq. 19-20 of Omidvarnia et al., 2018), which only matters for signals with
  repeated values (e.g., integers), and makes RangEn 0 at ``tolerance=1`` as described in the
  paper.
* ``entropy_cosinesimilarity()`` (and ``entropy_multiscale(method="MSCoSiEn")``) is now in bits,
  as defined by Chanwimalueang & Mandic (2017), and therefore between 0 and 1: it used the natural
  logarithm, so values were ln(2) = 0.69 times smaller (e.g., 0.25 instead of 0.37 for white noise
  with ``dimension=2`` and ``tolerance=0.07``, the value given in the paper). The cosine
  similarities are no longer rescaled to [-1, 1] (together with the zeros of the lower triangle of
  their matrix), which distorted the angles of short signals (e.g., by 2% for 30 samples); they
  are only clipped to [-1, 1] to remove rounding errors.
* ``entropy_cosinesimilarity()`` no longer multiplies the tolerance by the SD of the signal: the
  angular distance is between 0 and 1, so the default ``tolerance="sd"`` now gives 0.2 whatever
  the amplitude of the signal (and the other methods of ``complexity_tolerance()`` are applied to
  the standardized signal). CoSiEn used to depend on the amplitude of the signal, which it is
  designed not to: e.g., for RR intervals in ms (SD of 50 ms) the tolerance was 10, all the pairs
  of vectors were similar and CoSiEn was 0 (it is now 0.44 for such a random series of 300
  intervals). For signals with an SD of 1, only the change of logarithm applies (e.g., 0.45
  instead of 0.31 for white noise).
* ``entropy_kolmogorov()`` now counts the pairs of identical embedded vectors (at a distance of 0)
  in the correlation integrals; they were left out (e.g., 1.27 instead of NaN for 600 samples of
  white noise rounded to integers, with ``dimension=2`` and ``tolerance=0.5``). Results for
  signals without repeated vectors are unchanged.
* The ``"Excess_Entropy"`` returned by ``entropy_rate()`` is now the intercept of the line fitted
  to the block entropies, i.e., the excess entropy E in H(k) = E + h × k for long histories
  (Crutchfield & Feldman, 2003). It was the block entropy at k = 1 (the Shannon entropy of the
  symbols, which is E + h for a Markov chain): e.g., 0.24 instead of 0.81 bits for a
  binary Markov chain whose excess entropy is 0.24, and 0.00 instead of 1.00 for a random binary
  sequence. ``"MaxRatEn"`` is now NaN when the knee is the second history length (it was a slope
  fitted on a single point).
* ``entropy_multiscale(method="MSCoSiEn")`` now uses the same tolerance as
  ``entropy_cosinesimilarity()`` (0.2 with the default ``tolerance="sd"``, whatever the amplitude
  of the signal), so that its value at scale 1 is CoSiEn. It used 0.2 times the SD of the signal,
  as for the methods based on SampEn, as a tolerance on the angular distance (between 0 and 1).
* ``entropy_sample()`` and ``entropy_approximate()`` with ``distance="range"`` (and therefore
  ``entropy_range(approximate=True, corrected=True)``) now count each vector as a match of itself,
  and the pairs at a range distance equal to the tolerance, as for the other distances. The range
  distance of a vector with itself is undefined (0 / 0), so the vector was left out of its own
  count, from which ``entropy_sample()`` then removed one more match; ``entropy_sample()`` with
  ``distance="range"`` is now RangEn (B) (``entropy_range()``, with the same tolerance).
* ``entropy_fuzzy()`` (and ``complexity_fuzzyen()``, ``complexity_fuzzyapen()`` and
  ``entropy_multiscale(fuzzy=True)``) now passes on the exponent ``n`` of the membership function
  exp(-d^n / r), now documented: it was silently ignored, so that ``n=2`` (as in Chen et al., 2007)
  gave the values of ``n=1`` (e.g., 1.350 instead of 1.645 for 300 samples of white noise). Values
  with the default ``n=1`` are unchanged.
* ``complexity_rqa()`` (and ``hrv_rqa()``) now counts the vertical lines (``Laminarity``,
  ``TrappingTime``, ``VMax``, ``VEn``) in every column of the recurrence matrix, above and below
  the line of identity, as defined by Marwan et al. (2007). Only the part of each column below
  the line of identity was used, i.e., only the returns to a state that happen after it, which
  differs from the full matrix for time-irreversible signals (e.g., ``Laminarity`` of 0.043
  instead of 0.050 for the HRV of ``bio_resting_5min_100hz``; 1-4% differences on simulated
  signals). The white vertical lines (``W``, ``WMax``, ``WEn``) are now counted in the full matrix
  as well, and only between two recurrence points: the ones that reach the top or bottom border
  of the matrix are cut by the end of the recording, and set ``WMax`` to nearly the length of the
  signal (e.g., ``W`` of 85 instead of 129 samples and ``WEn`` of 4.41 instead of 5.41 for the
  same HRV), as for the recurrence times of RecurrenceAnalysis.jl (pyunicorn counts them).
* ``complexity_rqa()`` (and ``hrv_rqa()``) now fits ``DiagRec`` (Tomashin et al., 2022, approach
  3) over the first 64 diagonals, as in the authors' code, instead of all of them. The far
  diagonals, which hold few points, outnumbered the others in the fit, so the slope barely
  separated white from pink noise (-0.03 ± 0.03 vs. 0.00 ± 0.01, for 1,026 samples) and was noisy
  for brown noise (-0.20 ± 0.14); it is now -0.12 ± 0.03, 0.00 ± 0.01 and -0.30 ± 0.06. The other
  diagonal indices (``RecurrenceRate``, ``Determinism``, ``L``, ``Divergence``, ``LEn``) are
  unchanged.
* ``hrv_rqa()`` with the default ``tolerance="zimatore2021"`` now sets the radius to half the mean
  distance between all pairs of points of the embedded RR intervals (with the ``dimension`` and
  ``delay`` of the analysis), as in Zimatore et al. (2021), i.e., in the same space as the
  distances of the recurrence matrix. It was computed on the RR intervals alone (stacked twice,
  so sqrt(2) times their distances), a radius too small for the default 7-dimensional embedding:
  e.g., 41 instead of 97 ms for ``bio_resting_5min_100hz``, which left 0.3% of recurrences
  (now 12.7%) and nearly no vertical lines (``VMax`` of 2, now 24). All the default ``hrv_rqa()``
  indices change.
* ``complexity_rqa()``'s default ``tolerance`` is now ``"recurrence"`` (the radius that gives a
  recurrence rate of 1%) instead of ``"sd"`` (0.2 times the SD of the signal). The latter is
  designed for SampEn and ApEn, which use the Chebyshev distance (the largest difference between
  the coordinates), while the recurrence matrix uses the Euclidean distance, which is larger: it
  left about 0.1% of recurrences in 3 dimensions (the default) and none in 5 or more, so that
  most indices were undefined or degenerate. All the default ``complexity_rqa()`` indices change
  (``hrv_rqa()`` sets its own radius).
* ``complexity_tolerance(method="recurrence")`` now returns the smallest tolerance that gives a
  recurrence rate of 1% (the 1% quantile of the distances between the embedded points), instead
  of the closest of 50 values evenly spaced up to the largest distance, which gave recurrence
  rates between about 0.4% and 2.3%. With a custom ``r_range``, the value of the range whose rate
  is the closest to 1% is still returned. It is also about 7 times faster (0.6 s instead of 4.4 s
  for 5,000 samples).
* ``information_gain()`` with ``delay`` above 1 now follows each word by the word starting
  ``delay`` samples later, which adds one new symbol, so that the mean information gain (MIG) is
  the conditional entropy of that symbol given the word (the entropy of the words of
  ``dimension + 1`` symbols minus that of the words of ``dimension`` symbols), as with
  ``delay=1``. Each word was followed by the one starting one sample later, whose ``dimension``
  symbols are all new when ``delay`` is above 1: MIG jumped from 1 bit for white noise with
  ``delay=1`` to about ``dimension`` bits (e.g., 3.99 instead of 1.00 with the default
  ``dimension=4``), and the fluctuation complexity (``FC``) changed accordingly.
* ``information_gain()`` now compares the words symbol by symbol. They were joined into strings,
  so that with symbols of more than one character (e.g., ``symbolize=None`` with 10 or more
  values, or negative values) different words could be merged (e.g., 1 followed by 11, and 11
  followed by 1), which changed MIG and FC depending on the labels of the symbols; non-integer
  symbols (with ``symbolize=None``) were truncated to integers, which merged them (e.g., 0 and
  0.5). The probability of the next word given the current one is now that of the transitions
  that start from the current word (instead of all its occurrences, including the last one, which
  has no successor), which changes MIG by about 1e-6 for 2,000 samples. It returns NaN with a
  warning for signals with missing values (they were symbolized as 0) and for signals too short to
  have a transition between words.
* ``complexity_decorrelation()`` returns NaN with a warning for constant signals and for signals
  with missing values; it returned 1 (the autocorrelation was NaN). Its undocumented -1 for an
  autocorrelation that never crosses zero is now NaN, but this cannot happen for non-constant
  signals (the autocovariances of a demeaned signal at all lags sum to zero). Other values are
  unchanged.
* ``entropy_wiener()`` (Wiener entropy, or spectral flatness) is rewritten. It returned the
  flatness of the first window of a short-time Fourier transform only, ignoring the rest of the
  signal, and gave about 0.15 for white noise instead of a value close to 1. It is now the ratio of
  the geometric to the arithmetic mean of the power spectrum of the whole signal, estimated by
  ``signal_psd()`` with the Welch method on windows of a sixteenth of the signal by default: the
  flatness is biased downwards by the fluctuations of the estimated spectrum, and averaging about
  30 windows gives about 0.98 for white noise (0.84 with the default windows of ``signal_psd()``,
  and 0.56 with a single periodogram), 0.73 for an AR(1) process with a coefficient of 0.5 (whose
  unpredictable proportion of the variance is 0.75), and nearly 0 for sinusoids. The power is no
  longer clipped at 1e-10 (which depended on the units of the signal). The logarithmic Wiener
  entropy, from 0 to minus infinity as the docstring described, is returned in ``info["Log"]``
  together with the ``"PSD"``. It returns NaN with a warning for constant signals, signals with
  missing values and spectra with fewer than 2 frequency bins, gains a ``show`` argument, its
  ``**kwargs`` are passed to ``signal_psd()`` (instead of ``signal_timefrequency()``), and it is
  now listed in the documentation, with a docstring that gives the formula, its relationship with
  linear predictability (Dubnov, 2004) and references (Johnston, 1988; Tchernichovski et al.,
  2000).
* ``signal_changepoints()`` (and therefore ``eda_changepoints()`` and
  ``emg_activation(method="pelt")``) now returns the optimal changepoints (most were dropped or
  wrong), ``change="mean"`` uses the correct cost, constant segments no longer crash, and constant
  signals no longer return changepoints. The default ``penalty`` now follows the Schwarz Information
  Criterion (``2 * log(n)`` for ``change="var"`` and ``3 * log(n)`` for ``"meanvar"``).
* ``emg_activation(method="pelt")`` now labels bursts as active (activity and rest were inverted).

New Features
+++++++++++++

* ``complexity_delay(method="kim1999")`` (C-C method) is rewritten: it now splits the signal into
  the disjoint subseries of the paper (they were embedded with the delay a second time) and counts
  the pairs with vectorised distances, which makes it usable (a few seconds instead of minutes).
* ``complexity_dimension()`` works on quantized signals (e.g., RR intervals in integer
  milliseconds): for ``"fnn"``, identical vectors are accepted as neighbours and judged by
  Kennel's second test (the first one, a ratio, is undefined for them; they used to be silently
  counted as true neighbours); for ``"afnn"``, they are skipped instead of turning the whole
  estimate into NaN. Both methods return NaN with a warning,
  instead of raising, when no dimension can be evaluated.
* ``fractal_correlation()`` gains a ``theiler`` argument (Theiler window): pairs of points closer
  than that number of samples in time are left out of the correlation sum, which removes the bias
  of the autocorrelation of oversampled signals (e.g., EEG) at small radii. Defaults to 0 (all
  pairs), so results are unchanged.
* ``fractal_tmf()`` gains a ``random_state`` argument (for reproducible surrogates), and returns the
  multifractal width of the signal and of its surrogates in ``info``.
* ``complexity_rqa()``, ``recurrence_matrix()`` and ``hrv_rqa()`` gain a ``distance`` argument
  (``"euclidean"``, the default, ``"chebyshev"`` or ``"manhattan"``), which
  ``complexity_tolerance(method="recurrence")`` also takes, so that the default tolerance still
  gives 1% of recurrences (and ``hrv_rqa()``'s radius is in the same distance). With the Chebyshev
  distance of the entropy measures (and a delay of 1), the recurrence matrix in dimension *m* + 1
  is the one of dimension *m* where the diagonal lines continue, and SampEn is exactly minus the
  log of the probability that a recurrence point is followed by another one on its diagonal: this
  allows separating what RQA and the entropy measures share by definition from what they share
  empirically.
* ``complexity_lyapunov()`` gains a ``sampling_rate`` argument: the exponents are then per second
  instead of per sample (with ``show=True``, the time axis is in seconds).
* ``entropy_symbolicdynamic()`` gains a ``fluctuation`` argument for the fluctuation-based
  Symbolic Dynamic Entropy (FSDE; Shen et al., 2025), computed on the differences between the
  neighbouring symbols (and its multiscale version, MFSDE, with
  ``entropy_multiscale(method="MSSyDyEn", fluctuation=True)``). It is normalized by
  ln((2c - 1)^m), the maximum of the joint entropy of the states and of their transitions, so
  that it lies between 0 and 1; the paper divides it by ln((2c - 1)^(m - 1)), which gives values
  m / (m - 1) times larger.
* New ``complexity_jensenshannon()``: the Jensen-Shannon statistical complexity (C_JS, or MPR
  complexity; Rosso et al., 2007), the product of the normalized permutation entropy and of the
  Jensen-Shannon divergence between the distribution of the ordinal patterns and the uniform one.
  Unlike permutation entropy alone, it distinguishes chaos from coloured noise: with
  ``show=True``, the signal is drawn in the complexity-entropy plane, with the curves of minimum
  and maximum complexity (Martin et al., 2006). It also returns the proportion of missing
  (forbidden) ordinal patterns (Amigó et al., 2007).
* New ``complexity_irreversibility()``: time-irreversibility indices, which are 0 for reversible
  (e.g., linear Gaussian) processes and change sign when the signal is reversed: Trev, the
  skewness of the increments (Schreiber & Schmitz, 2000; the ``trev`` feature of hctsa), and the
  up-down balance and rotational asymmetry ordinal contrasts (Bandt, 2023), which leave out the
  patterns with ties. Unlike the heart rate asymmetry indices of ``hrv_nonlinear()``, they apply
  to any signal.
* New ``entropy_correctedconditional()``: the corrected conditional entropy (CCE; Porta et al.,
  1998), the minimum over the length of the patterns of the conditional entropy of a sample given
  the previous ones, corrected for the patterns found only once, a standard regularity index for
  short cardiovascular series. It also returns the normalized CCE and the regularity index
  (Porta et al., 2001).

Fixes
+++++++++++++

* ``signal_autocor()``: ``unbiased=True`` now returns 1 at lag 0 (values were divided by their
  maximum), integer signals no longer overflow with ``demean=False``, invalid lags and
  multidimensional inputs raise a ``ValueError``, and the plot's lag axis starts at 0.
* ``eda_autocor()`` accepts fractional lags and float sampling rates.
* ``signal_cyclesegment()``, ``ppg_segment()``, ``rsp_segment()`` and ``ecg_segment()`` accept the
  documented peak dict (and the tuple or DataFrame returned by the ``*_peaks()`` functions) and
  raise a clear error with too few peaks. Any ``signal_name`` can now be plotted by
  ``signal_cyclesegment()``.
* ``signal_detrend()`` accepts ``(n, 1)`` column arrays (it could return an ``(n, n)`` matrix).
* ``rsp_process()`` no longer raises ``IndexError`` when short recordings leave no complete
  trough-to-peak pair after detection. Peak finding returns empty arrays, and phase and amplitude
  return ``NaN`` values when peaks or troughs are missing (#1242).
* ``hrv(show=True)`` no longer raises ``KeyError`` when ``peaks`` is the info dict returned by
  ``ppg_peaks()``; the plot now uses the same peaks as the computed indices.
* ``complexity_rqa()`` (and ``hrv_rqa()``) now returns the documented ``LamiDet`` (laminarity /
  determinism): it was computed but never stored, so the column was missing (it only appeared,
  as NaN, when the determinism was 0 or undefined). The lines of the recurrence matrix are now
  counted with vectorised run lengths instead of a Python loop over the diagonals and rows: e.g.,
  0.6 s instead of 21 s and a sixth of the memory for 5,000 samples.
* ``signal_merge()`` now correctly adds list and Series inputs.
* ``complexity_delay(method="fraser1986")`` no longer fails on a noise-free periodic signal when
  the tested delays reach its period (the default number of bins of ``mutual_information()``
  diverged for perfectly correlated variables).
* The event-related ECG, PPG, RSP and EOG codebooks now clarify that rate maxima, minima and
  means are changes from the sample nearest stimulus onset, while SD is an absolute population
  standard deviation. Calculations and output names are unchanged (#1240).
* ``complexity_tolerance()`` accepts numpy integers as ``dimension`` (e.g., the output of
  ``complexity_dimension()``), which used to raise "method not recognized" for ``"nolds"``,
  ``"chon2009"`` and ``"makowski"``; plotting the ``"recurrence"`` and ``"neighbours"`` methods no
  longer converts the returned scores to percentages in place, and the plots show the 1% and 2%
  targets; ``show=True`` works with the ``"makowski"`` method; the docstring now lists all the
  methods and which ones use ``delay`` and ``dimension``.
* ``complexity_lempelziv()`` is about 10 times faster, no longer fails on a one-sample signal, and
  with ``permutation=True`` its ``info`` keeps the ``"Permutation"``, ``"Dimension"`` and
  ``"Delay"`` entries (they were overwritten by those of ``complexity_ordinalpatterns()``).
* ``fractal_katz()``'s docstring now describes the version computed (absolute differences between
  successive values, ignoring time) and no longer claims a 1-1.5 range: KFD is not bounded by 2 and
  grows with the length of the signal for noise.
* ``eog_findpeaks()`` and ``eog_clean()`` now identify the correct function in missing-MNE errors,
  rather than reporting ``signal_filter()``. ``eog_process()`` documents its default blink detector's
  optional MNE dependency and the ``method="neurokit"`` alternative (#1241).
* ``fractal_correlation()`` is about 3 times faster and uses half the memory: the distances are
  computed once per pair (``scipy.spatial.distance.pdist()``) and the correlation sums of all the
  radii are counted in one pass. The radii are returned sorted. Results are unchanged, except with
  ``radius="Corr_Dim"`` for signals with exactly repeated points, whose smallest nonzero distance
  (which sets the radii) was rounding noise.
* ``fractal_correlation()``'s docstring now cites Grassberger & Procaccia (1983) and Theiler
  (1986), and notes that the default radii can reach beyond the scaling region of an attractor.
* ``fractal_dfa()`` is 5 to 15 times faster for the same scales (e.g., 0.2 s instead of 2.7 s for
  200,000 samples and 40 scales): the polynomial trends of all the windows of a scale are computed
  at once, from their coefficients, instead of with a loop over the windows. With the new default
  scales (see above), it is about 75 times faster (0.02 s instead of 1.5 s for 20,000 samples).
* ``fractal_dfa(multifractal=True)``: several ``q`` values within 0.1 of 0 raised an
  ``IndexError`` or silently swapped the fluctuations of two ``q`` values (e.g., with
  ``q=np.linspace(-5, 5, 101)``); unsorted ``q`` values gave a wrong singularity spectrum (they are
  now sorted).
* ``fractal_dfa(maxdfa=True)`` kept too few scales (the knee was truncated to an integer power of
  2, e.g., scales up to 64 instead of 87 were kept), and ``info["scale"]`` still listed all of them.
* ``fractal_dfa()`` accepts a number of scales given as a NumPy integer.
* ``fractal_dfa()``'s docstring no longer says that ``q = 0`` is removed or asks for help with it:
  it uses the logarithmic average of Kantelhardt et al. (2002).
* ``fractal_dfa()``'s docstring now describes alpha correctly (0.5 for white noise, below 0.5 for
  anti-correlations, 1.5 for a Brownian motion, equal to the Hurst exponent only for stationary
  signals), gives the right minimum of the default scales (10 samples, not 4), and cites Peng et
  al. (1994).
* ``fractal_petrosian()``'s docstring now gives the right default for ``symbolize`` (``"C"``, the
  sign of consecutive differences, not the mean).
* ``fractal_sevcik()``'s docstring now describes the method and explains the ln(2) term found in
  some papers: it comes from Sevcik's derivation and vanishes as N grows, and his eq. 6 (without
  it, as used here) is the one that reproduces the values of his Table 2. Results are unchanged.
* ``fractal_nld()``'s docstring now says that the calibration curve was fitted on Weierstrass
  functions (not on EEG), gives the parameters that Kalauzi et al. (2009) adjusted on EEG, and
  that a float (not a DataFrame) is returned.
* ``fractal_hurst()``'s docstring now says that H = 0.5 corresponds to uncorrelated noise (the
  increments of a Brownian motion, not the Brownian motion itself) and describes the correction.
* ``fractal_psdslope()``'s docstring now gives both conversion formulas and the range of validity of
  Voss' (fractional Brownian motions; it returns values above 2 for noise-like signals), and no
  longer says that the PSD ``method`` can be changed (it is the function's own ``method``
  argument).
* ``fractal_sda()``'s docstring now describes the bins (``scales`` are bin sizes, which can be
  given as a list) and the relationship with H (FD = 2 - H for fractional Gaussian noise).
* ``complexity_k()`` (and therefore ``fractal_higuchi(k_max="default")``) is several hundred times
  faster (e.g., 0.04 s instead of 19 s for 1,600 samples): the curve lengths are computed once
  instead of again for each tested k_max, and the least-squares fits of all the k_max values are
  done at once. The HFD values are unchanged.
* ``complexity_k()`` and ``fractal_higuchi()`` now accept lists and pandas Series (with any
  index), and ``complexity_k()`` accepts numpy integers as ``k_max`` (it raised an
  ``UnboundLocalError``). ``complexity_k()`` raises an error for k_max values that are not between
  2 and half the length of the signal, and ``fractal_higuchi()`` warns before returning NaN. The
  k_max returned by ``complexity_k()`` is an int, and its warning when no plateau is found no
  longer says that the optimal k_max is 2 or less.
* ``complexity_hjorth()``'s docstring now gives the right formulas of mobility and complexity (the
  one given for complexity was the mobility, and mobility was defined through complexity), and
  the units of mobility.
* ``entropy_bubble()`` and ``complexity_ordinalpatterns(algorithm="bubblesort")`` are up to a few
  hundred times faster: the swaps of bubble sort are counted as the number of inversions of each
  vector (pairs of elements in the wrong order), with array operations. The ``"Embedded"`` vectors
  returned by ``complexity_ordinalpatterns(algorithm="bubblesort")`` are no longer sorted (they
  were sorted in place). ``entropy_bubble()`` raises an error when ``dimension`` is below 2 (instead
  of a ``ZeroDivisionError``), and its docstring describes the method, no longer lists a
  ``tolerance`` argument, and gives the right default of ``alpha``.
* ``entropy_increment()`` is up to 50 times faster (the words are counted with ``np.unique()``),
  raises an error when ``dimension`` is below 2 (instead of returning NaN), and its docstring
  describes the words and the normalization by ``dimension - 1``.
* ``entropy_phase()`` now accepts lists and pandas Series: a Series returned NaN (its differences
  were aligned on the index) and a list raised a ``TypeError``. With ``show=True``, the lines of
  the sectors are drawn once.
* ``entropy_differential()`` has an explicit ``method`` argument, now documented, works on arrays
  instead of converting them to lists, raises an error for strings (instead of returning NaN), and
  its docstring notes that it depends on the units of the signal.
* ``entropy_attention()`` now accepts lists (they raised a ``TypeError``) and pandas Series with
  any index when ``show=True`` (they raised a ``KeyError``), and is up to 3 times faster for long
  signals (the intervals are counted with ``np.bincount()`` instead of a histogram with one bin per
  sample).
* ``signal_formatpeaks()`` now drops every missing index together with its value. Before, only the
  first one was dropped, so with two or more unrecovered SCRs ``eda_peaks()`` put ``SCR_RecoveryTime``
  values at the wrong recovery points.
* ``entropy_multiscale()`` now accepts lists (they raised an ``AttributeError``), numpy integers
  and arrays as ``scale`` (they raised a ``TypeError`` or a ``ValueError``), and ``show=True`` with
  a list of scales (it raised a ``TypeError``); ``complexity_mspe()`` works (it passed ``"MSPEn"``
  as the ``scale`` and always raised a ``TypeError``). The entropy is now NaN at the scales where
  the coarse-grained series is too short to embed (fewer than (``dimension`` + 1) × delay + 1
  samples) instead of raising an error, so ``scale="max"`` works with the SampEn-based methods.
  The error for an unknown ``method`` names it and lists all the methods. The docstring describes
  the index and notes that ``delay`` is ignored; its FuzzyRCMSEn example printed the FuzzyCMSEn
  value.
* ``entropy_sample()``, ``entropy_approximate()`` and ``entropy_multiscale()`` are 2 to 4 times
  faster on short series (up to 50 embedded vectors, e.g., the coarse-grained series at large
  scales): the neighbours are counted with a dense distance matrix instead of a KDTree, whose
  fixed overhead dominated. The scikit-learn version is no longer parsed at every call, and the
  ratio of the counts no longer goes through ``np.isclose()``. Values are unchanged.
* ``complexity_coarsegraining()`` accepts lists (a list was returned unchanged at ``scale=1``) and
  numpy integers as ``scale``; its docstring describes the time-shift series, and its benchmarking
  example uses ``method="rolling"`` (it passed a ``rolling=True`` argument that does not exist).
* ``entropy_shannon()``, ``entropy_renyi()``, ``entropy_tsallis()`` and
  ``entropy_cumulativeresidual()`` accept numpy integers as ``symbolize`` (e.g., ``np.int64(3)``
  for 3 bins), which raised an ``UnboundLocalError``. The docstring of ``entropy_shannon()`` names
  the ``symbolize`` argument (it said ``method``), gives the formula with the logarithmic ``base``
  and documents the ``**kwargs``.
* ``entropy_renyi()`` with ``alpha=np.inf`` now returns the min-entropy, -log(max p), instead of
  NaN. Its docstring now gives the formula with the natural logarithm that it uses (it showed
  log2; values are in nats, so that ``alpha=1`` is ``entropy_shannon(base=np.e)``), says that it
  returns the Rényi entropy (not the Tsallis entropy), and its ``info`` contains ``"Alpha"``.
* ``entropy_tsallis()``'s docstring notes that ``q=1`` gives the Shannon entropy in nats, and
  cites Tsallis (1988). The ``"Values"`` in its ``info`` are 0 for the states that do not occur
  (they were NaN for ``q`` of 1 or below).
* ``entropy_cumulativeresidual()`` works with the ``freq`` argument, taken as the probabilities of
  the values 0, 1, 2, etc. (it always raised an ``UnboundLocalError``), raises a clear error for
  non-numeric values such as strings (instead of a ``TypeError`` on subtracting them), and is
  about 30 times faster for 100,000 samples (the terms are computed with array operations instead
  of a loop over the values). Its docstring describes how the survival function of the empirical
  distribution is integrated, what is done for signals with negative values, and its units.
* ``entropy_kl()``'s docstring now gives the formula of the estimator, its unit (nats), its
  dependence on the units of the signal and the defaults of ``delay`` and ``dimension``, and cites
  Kraskov et al. (2004); an unknown ``norm`` raises an error that lists the valid ones.
* ``entropy_permutation()`` (and ``complexity_pe()`` and ``complexity_wpe()``) returns NaN with a
  warning for signals with missing values (they were ranked as the largest values) and, with
  ``weighted=True``, when all the vectors are constant (the weights are all 0); it raises an error
  when ``dimension`` is below 2 without ``conditional=True``, or below 1 with it (instead of
  returning NaN after a division by zero). The weighted frequencies of WPEn are counted with ``np.bincount()``, which is
  up to 70 times faster for large dimensions (e.g., 10 ms instead of 0.7 s for 10,000 samples with
  ``dimension=7``). The docstring documents the ``conditional`` argument and the ``algorithm``
  keyword argument (and no longer names ``entropy_reyni``), gives the formulas, and describes CPEn
  as the sorting entropy of Bandt & Pompe (2002), with its bounds. It described it as the
  conditional entropy of ordinal patterns of Unakafov & Keller (2014), the entropy of the next
  pattern given the current one, which is lower (e.g., about 0.66 instead of 0.89 nats for the
  logistic map with r = 4 and ``dimension=5``, whose Kolmogorov-Sinai entropy is ln(2) = 0.693).
* ``entropy_dispersion()`` returns NaN with a warning for signals with missing values (they were
  put in the highest class), raises an error when ``c`` is below 2 (instead of returning NaN) and
  when ``dimension`` is below 2 with ``fluctuation=True`` (instead of returning 0) or
  ``symbolize="finesort"`` (instead of a cryptic error), and passes ``random_state`` on to
  ``complexity_symbolize()`` (e.g., for ``symbolize="kmeans"``, whose values changed at every call).
  Its docstring describes DispEn, FDispEn, the fine-sorted DispEn and RDEn, documents the ``base``
  keyword argument, and cites Azami & Escudero (2018), Li et al. (2019) and Fu et al. (2019).
* ``entropy_slope()`` now accepts lists (they raised a ``TypeError``) and pandas Series with any
  index (a Series with a non-default index returned 0, as its differences were aligned on the
  index), returns NaN with a warning for signals with missing values, and raises an error when
  ``dimension`` is below 2 (it returned 0) or when the ``thresholds`` are not increasing angles
  between 0 and 90 degrees (unsorted thresholds gave meaningless symbols). Its docstring describes
  the symbols and the thresholds (the default ``[0.1, 45]`` corresponds to differences of 0.0017
  and 1), and notes that Cuesta-Frau (2019) divides the counts by the number of different patterns
  rather than by the number of vectors (5.29 instead of 3.54 bits for the worked example of the
  paper).
* ``entropy_symbolicdynamic()`` is about 20 times faster (the states and transitions are counted
  with ``np.unique()`` and ``np.bincount()``), returns NaN with a warning for signals with missing
  values, raises a clear error for signals too short to have a state transition (it raised an
  ``IndexError``), and passes ``random_state`` on to ``complexity_symbolize()`` (e.g., for
  ``symbolize="kmeans"``). Its docstring gives the formula and its normalization.
* ``entropy_coalition()`` accepts ``method="ACE"`` and ``method="SCE"``, as documented (they raised
  an ``UnboundLocalError``, like any unknown method, which now raises a ``ValueError``), raises an
  error for fewer than 2 channels, returns NaN with a warning for signals with missing values, and
  no longer reseeds numpy's global random number generator (it called ``np.random.seed(30)``);
  ACE values are unchanged. It is 5 to 25 times faster (e.g., 0.11 s instead of 2.9 s for SCE with
  16 channels of 20,000 samples): the coalitions are mapped to integers with a matrix product and
  the synchrony of all the pairs of a channel is computed at once. Its docstring describes the
  computation and documents that arrays are taken as (n_channels, n_samples), unlike DataFrames.
* ``entropy_power()`` raises a clear error for constant signals (instead of a ``LinAlgError`` from
  the kernel density estimation) and for signals with missing values, and its docstring gives the
  formula and its units and no longer warns that the implementation is uncertain.
* ``entropy_spectral()`` accepts numpy integers as ``bins`` (e.g., ``np.int64(10)``), which were
  silently ignored (the spectrum was not binned, e.g., 0.751 instead of 0.592 for sinusoids at 5,
  6 and 10 Hz with 10 bins), and raises an error when ``bins`` is not an integer of at least 2
  (``bins=1`` returned NaN, and floats were ignored). It returns NaN with a warning for constant
  signals, signals with missing values, and when the spectrum has fewer than 2 frequency bins (it
  returned -0.0 or NaN, without a warning, for signals of fewer than 10 samples). Its docstring
  gives the normalization by the logarithm of the number of frequency bins, describes the binning
  and the default frequency range, and cites Inouye et al. (1991).
* ``entropy_svd()`` (and the ``SVDEn`` index of ``complexity()``) returns 0 instead of NaN when a
  singular value is null (a rank-deficient embedding, e.g., ``[1, 0, 0, 0, 0]`` with
  ``dimension=2``), and NaN with a warning (instead of a ``LinAlgError``) for signals with missing
  values. Its docstring gives the formula (Roberts et al., 1999), the range of the values
  (0 to log2(``dimension``) bits), notes that it depends on the mean of the signal, and describes
  the returned value correctly (it was described as the singular value decomposition).
* ``entropy_distribution()`` is about 4 times faster (the Chebyshev distances are computed with
  ``scipy.spatial.distance.pdist()``); values are unchanged. The ``"Doane"`` method of ``bins``,
  as documented, works (only ``"doanes"`` was recognized). It returns NaN with a warning for
  signals with missing values (instead of a ``ValueError`` from the histogram) and for signals too
  short to give 2 distances (instead of NaN without a warning or an ``OverflowError``), and raises
  an error for fewer than 2 bins. A non-integer number of bins is now truncated for the
  normalization as it was for the histogram. Its docstring describes the method (Li et al., 2015).
* ``entropy_ofentropy()`` is up to a few hundred times faster (e.g., 3 ms instead of 1.4 s for
  100,000 samples): the samples of all the windows are counted at once. Values are unchanged.
  It has an explicit ``base`` argument (the other keyword arguments, which were ignored, now raise
  an error), returns NaN with a warning (instead of 0) for signals with missing values or shorter
  than ``scale``, and raises an error when ``scale`` or ``bins`` is below 1. Its docstring
  describes the steps of Hsu et al. (2017).
* ``entropy_quadratic()``'s docstring explains the ln(2r) correction of SampEn and the units of QSE,
  and gives the right year of Lake (2006) and the reference of Lake & Moorman (2011).
* ``entropy_angular()`` is 20 to 30 times faster for 1,000 to 4,000 samples (e.g., 0.08 s instead
  of 1.9 s for 2,000 samples) and uses much less memory: the cosines are computed once per pair
  with ``scipy.spatial.distance.pdist()``, instead of from two full matrices of all the pairs, and
  the density is estimated from the binned distances instead of a Gaussian for each distance and
  bin. It returns NaN with a warning for constant signals, signals with missing values, and
  signals too short to give two distinct distances (it returned NaN silently, or with a
  ``RuntimeWarning``), and leaves out the embedded vectors at the origin, whose angle is undefined
  (a single one made AngEn NaN). Its ``info`` now includes the ``Dimension`` and the ``Delay``,
  and its docstring describes each step, gives the formula and cites Botev et al. (2010).
* ``entropy_grid()`` returns NaN with a warning for constant signals and signals with missing
  values (they raised a ``ValueError`` about a non-finite range), and raises a clear error when
  ``k`` or ``delay`` is not a positive integer or the signal is not longer than ``delay`` (a
  ``delay`` of 0 raised an unrelated ``ValueError``, and a one-sample signal returned 0). With
  ``show=True``, the number of points in each grid is now drawn in the same orientation as the
  Poincaré plot (it was mirrored) and with axis labels (they were set on the Poincaré plot a
  second time). Its docstring describes the normalization, the *k* × *k* grids and the range of
  GridEn (0 to log(k²)), calls GDR a proportion (not a percentage), and its examples no longer
  name the result ``phasen``.
* ``complexity_lyapunov(method="eckmann1986")`` works with the default ``dimension=2`` and with
  ``dimension=3`` (it raised "slice step cannot be zero"): the default ``matrix_dim`` is now
  ``min(dimension, 4)`` (it was always 4), and ``matrix_dim`` and ``min_neighbors`` are now passed
  on (they were silently ignored). An error explains the valid values of ``matrix_dim`` when
  ``dimension - 1`` is not a multiple of ``matrix_dim - 1`` (e.g., ``dimension=5`` failed with a
  shape error). It also accepts lists and pandas Series (they raised a ``TypeError`` or a
  ``KeyError``). The exponent is -inf instead of +inf when all the diagonal elements of R are 0.
* ``complexity_lyapunov()`` is much faster and uses much less memory for long signals: the nearest
  neighbours are found with a KDTree instead of a full distance matrix (8 × N² bytes, e.g., 3.2 GB
  for 20,000 samples). E.g., 0.02 s instead of 1.2 s for 10,000 samples with
  ``"rosenstein1993"``, and 0.6 s instead of 3.3 s with ``"eckmann1986"``. The neighbours are the
  same.
* ``complexity_lyapunov()`` returns NaN (with a warning) for constant signals and signals with
  missing values (they raised "cannot convert float NaN to integer"), raises an error explaining
  the minimum length when the signal is too short to find neighbours outside of the separation
  window (it used to warn and return a value, or fail with "attempt to get argmin of an empty
  sequence"), and accepts numpy integers as ``separation`` (they raised an ``AssertionError``).
* ``complexity_lyapunov()`` now raises an error for unknown arguments (``"rosenstein1993"``
  silently ignored them), and ignores the arguments of the other methods (e.g.,
  ``len_trajectory`` with ``"makowski"`` raised a ``TypeError``). ``show`` is an explicit argument.
* ``complexity_lyapunov(method="rosenstein1993")`` now fits the line at the steps of the divergence
  curve where it is defined (they were renumbered after leaving out the steps where all the
  distances were 0), and its ``info`` contains the whole curve and the ``"Intercept"`` of the line
  (also for ``"makowski"``).
* ``complexity_lyapunov()``'s docstring now describes the algorithms with their formulas, the
  units of the exponents (per sample, in natural logarithm), the dependence of Rosenstein's method
  on ``len_trajectory`` (e.g., 0.35 with the default of 20 samples instead of 0.69 with 8 for the
  logistic map, whose LLE is ln(2)), the ``max_length`` argument of ``"makowski"``, the
  ``separation`` argument, and the spurious exponents of Eckmann's method when ``matrix_dim`` is
  larger than the dimension of the attractor (Sauer et al., 1998). It no longer claims that the
  ``**kwargs`` are passed to ``signal_psd()``, nor that Eckmann's method does not work.
* ``fishershannon_information()``'s docstring gives the formula of the Fisher-Shannon complexity
  and its expected values, notes that its ``**kwargs`` are passed to ``density()``, and no longer
  warns that the implementation is uncertain; its example no longer passes a ``method`` argument
  that was ignored.
* ``entropy_range()`` is 4 to 6 times faster (the range distances are computed by blocks with
  array operations instead of a loop over the vectors), returns NaN with a warning for constant
  signals (whose range distances are all undefined) and for signals with missing values (instead
  of 0 or -0), and raises an error for ``corrected=True`` without ``approximate=True`` (it was
  ignored). ``corrected`` is now an explicit argument, and other arguments raise an error instead
  of being ignored. Its docstring gives the range distance and the definitions of RangEn (A) and
  (B), and explains why RangEn (A) is often undefined.
* ``entropy_cosinesimilarity()`` is 3 to 4 times faster and uses much less memory (the angles are
  computed by blocks of pairs instead of with several N x N matrices), raises an error when
  ``dimension`` is below 2, and returns NaN with a warning for signals with missing values
  (instead of 0). Its docstring gives the formulas and the tolerances recommended by
  Chanwimalueang & Mandic (2017), and no longer lists a ``**kwargs`` argument that does not exist.
* ``entropy_kolmogorov()`` is about 7 times faster and uses half the memory (the distances are
  computed once per pair with ``scipy.spatial.distance.pdist()``), and returns NaN with a warning
  for signals with missing values. Its docstring describes the estimator (Grassberger & Procaccia,
  1983), its units (nats per sample) and its convergence.
* ``entropy_fuzzy()`` (and ``complexity_fuzzyen()``) raises an error for ``corrected=True``
  without ``approximate=True`` (it was ignored). Its docstring now describes FuzzyEn (Chen et al.,
  2007) and FuzzyApEn: the removal of the baseline of each vector, the membership function
  exp(-d / r) (Chen et al., 2007, used exp(-d^2 / r)) and the treatment of the self-matches, cites
  Chen et al. (2007, 2009) and Xie et al. (2010), and its example is rendered (the directive was
  missing a space). Values (and therefore the ``HRV_FuzzyEn`` index of ``hrv_nonlinear()``) are
  unchanged.
* ``entropy_rate()`` accepts a list of history lengths with ``show=True`` (it raised a
  ``TypeError``) and sequences of any symbols (e.g., letters, with ``symbolize=None``), raises an
  error for fewer than two history lengths (the slope was fitted on a single point), and returns
  NaN with a warning for signals with missing values (they were symbolized as 0). Its docstring
  describes the block entropies (in bits), the excess entropy and the returned values, and cites
  Crutchfield & Feldman (2003); the plot shows the excess entropy at k = 0.
* ``entropy_angular()``, ``entropy_distribution()``, ``entropy_cosinesimilarity()``,
  ``entropy_kolmogorov()`` and ``entropy_coalition()`` are now listed in the documentation.
* ``expspace()`` now spans the full range and no longer depends on the units of the values.
* ``fractal_correlation()`` now accepts a list of radiuses and returns 0 for constant signals.
* ``signal_resample()`` no longer sets the last sample to 0 for some signal lengths.
* ``ecg_simulate()`` no longer hangs for some settings.
* ``complexity_dimension(method="fnn")`` (and ``complexity_optimize(dimension_method="fnn")``)
  now return the dimension at which the proportion of false nearest neighbours meets the
  criterion, rather than its index in the tested sequence. The result was one too low with the
  default sequence (e.g., 2 instead of 3 for the Lorenz attractor) and arbitrary for a custom
  ``dimension_max`` sequence; the marker drawn with ``show=True`` was off accordingly.
* ``complexity_optimize(dimension_method="fnn", show=True)`` no longer raises a shape error when
  plotting the false nearest neighbour fractions.
* ``information_gain()`` is 50 to 250 times faster (e.g., 8 ms instead of 0.4 s for 100,000
  samples, and 1.4 ms instead of 0.38 s for 10,000 samples with ``dimension=10``): the words are
  numbered as integers and the transitions counted with ``np.unique()``, instead of joining the
  symbols into strings and looping over all the pairs of words. Its docstring gives the formulas
  of MIG and FC and describes the words with a ``delay``.
* ``complexity_relativeroughness()`` is about 25 times faster for long signals (e.g., 5 ms instead
  of 0.13 s for 1,000,000 samples): only the autocovariances at lag 0 and 1 are computed, instead
  of the whole autocorrelation function. Values are unchanged. ``demean`` and ``unbiased`` are now
  explicit arguments (the other arguments of ``signal_autocor()``, which were passed through
  ``**kwargs``, do not change RR and are no longer accepted). It returns NaN with a warning for
  constant signals and signals with missing values (instead of NaN with a ``RuntimeWarning``), and
  its docstring gives the formula and its expected values for white noise (2) and fractional
  Gaussian noise (4 - 2^(2H)).
* ``complexity_decorrelation()``'s plot (``show=True``) now normalizes the autocorrelation by its
  value at lag 0 (the unbiased autocovariance could be larger at long lags, which shrank the
  curve), and its docstring defines DT as the first lag at which the autocorrelation is zero or
  negative.
* ``complexity_embedding()`` raises a readable error when ``dimension * delay`` exceeds the length
  of the signal (the message was a tuple of two strings), raises an error when ``dimension`` is
  below 1 (it returned an array with no columns and more rows than samples), and accepts NumPy
  floats such as ``np.float32`` as ``delay`` and ``dimension`` (with the same warning as Python
  floats; they raised a ``TypeError``). Its docstring gives the shape of the output, which has
  ``dimension`` columns.


0.3.0
-------------------
Compatibility
+++++++++++++

* pandas 3 is now supported (the ``pandas<3`` pin was removed). Python 3.10 to 3.14 are tested.

Changes in Results
++++++++++++++++++

Several fixes below change computed values compared to 0.2.x:

* ``HRV_PSS`` is now weighted by the number of RR differences in short segments (e.g., from
  0.74 to 0.55 on ``bio_resting_5min_100hz``) (#1214).
* ``RSA_PorgesBohrer`` is now computed on heart period and is positive, in ln(ms^2); it was
  previously negative (#1231).
* ``HRV_SDANN*`` and ``HRV_SDNNI*`` now include the last interval of each window and use the
  actual interval timestamps when available (#1213, #1215).
* ``HRV_TINN`` no longer returns 0 when the interval histogram has large counts (#1216).
* ``ecg_delineate(method="dwt")`` P and T peaks (and the onsets/offsets derived from them) can
  differ for signals in their original units (#1220).
* ``epochs_create(events=None)`` now returns the final complete block, i.e., often one more
  epoch (#1222).
* ``rsp_rvt()`` differs at sampling rates other than 1000 Hz (#1233).
* ``standardize(robust=True)`` on multi-column data now scales each column by its own MAD (#1229).
* ``fractal_dfa()`` with negative ``q`` can shift slightly (#1209).

New Features
+++++++++++++

* ``read_xdf()`` returns richer metadata: ``info`` now contains ``stream_names``,
  ``stream_types``, ``channel_names``, ``channel_counts``, ``durations`` and a per-stream
  ``streams`` list (name, type, channels, nominal and effective sampling rates, start/end
  times, uid, source id, hostname). Metadata is computed from the streams that actually
  contribute to the returned data.
* ``ecg_peaks()``, ``ecg_findpeaks()``, ``ecg_clean()`` and ``ecg_process()`` support
  ``method="elgendi2013"``, the two-moving-average QRS detector of Elgendi (2013). ``"elgendi"``
  still refers to the 2010 detector (#1202, #1232).

Fixes
+++++++++++++

* ``hrv_nonlinear()`` weights short acceleration/deceleration segments by their
  number of RR differences when calculating ``HRV_PSS`` (#1214).
* ``hrv_time()``: ``SDANN`` and ``SDNNI`` dropped the last RR interval of each window, and
  ignored the actual interval timestamps (e.g., recordings with gaps) (#1213, #1215).
* ``hrv_time()``: ``TINN`` returned 0 when every candidate triangle fit had a squared error
  above a fixed ceiling, which happens with large interval counts (#1216).
* ``rsp_rvt()`` passes ``sampling_rate`` to breath detection in the ``"power2020"`` and
  ``"birn2006"`` methods; at lower sampling rates many breaths were skipped (#1233).
* ``entropy_spectral()`` forwards ``sampling_rate`` to ``signal_psd()`` instead of raising a
  ``TypeError``, so ``min_frequency`` and ``max_frequency`` are applied in Hz of the actual
  signal (#1219).
* ``epochs_create()`` retains the final complete block during automatic chunking
  and aligns blocks with the data when relative epoch starts are nonzero (#1222).
* ``signal_interpolate(method="monotone_cubic")`` applies boundary fill values to every
  out-of-range sample, including irregular or unordered output grids (#1227).
* ``standardize(window=...)`` accepts named pandas columns and preserves their labels and
  sample index when performing rolling standardization (#1228).
* ``standardize(robust=True)`` scales each input column by its own MAD, so columns
  measured in different units are standardized independently (#1229).
* ``hrv_rsa()``: ``RSA_PorgesBohrer`` is now computed on heart period (ms) instead of heart
  rate (bpm), so it is in ln(ms^2) as documented and no longer comes out negative (#1230,
  #1231).
* ``ecg_process()``, ``ecg_delineate()``, ``ecg_phase()`` and ``ecg_quality()`` no longer
  crash when no R-peak is detected; they return NaN/empty outputs with a warning.
* ``read_acqknowledge()`` returns the sampling rate as an ``int`` (previously a NumPy float,
  which broke downstream functions), and the file-reading logic was updated for pandas 3.
* ``epochs_create()`` and ``signal_recompose()`` were updated for pandas 3 and scipy 1.18.
* ``signal_recompose()`` now applies ``keep_sd`` to reconstructed components instead of
  discarding the reconstruction and filtering the original components (#1221).
* ``eda_process()`` now forwards a custom ``amplitude_min`` to ``eda_peaks()``
  instead of hardcoding ``0.1`` (#1197, #1203).
* ``rsp_intervalrelated()`` computes inspiration/expiration durations from sample positions
  rather than index labels, fixing wrong durations for time-indexed epochs (#1196, #1201).
* ``hrv()`` and related functions accept ``(peaks, sampling_rate)`` tuples as well as the
  ``(signals, info)`` tuples returned by processing functions (#1205).
* ``signal_detrend(method="tarvainen2002")``: the second-difference operator was truncated at
  the end of the signal, leaving the last samples undetrended. The linear system is now solved
  with a sparse solver, which is also orders of magnitude faster for long signals (#1198).
* ``ecg_intervalrelated()``, ``rsp_intervalrelated()``, ``eda_intervalrelated()``,
  ``emg_intervalrelated()`` and ``eog_intervalrelated()`` no longer carry features over from
  a previous call (e.g., ``HRV_DFA_alpha2`` from a long recording reappearing in the results
  of a short one). ``ecg_intervalrelated()`` also returns HRV indices as floats instead of
  nested arrays (#1226).
* ``ecg_delineate(method="prominence")`` no longer crashes when an R-peak falls at the boundary
  of a beat segment (#1181).
* Microstates: ``method="ica"`` was routed to PCA and ``method="aahc"`` was not recognized;
  ``train="all"`` selected only the first ``n_channels`` samples; precomputed GFP passed to
  ``microstates_peaks()`` was ignored; EEG standardization was performed across channels
  instead of across time; MNE ``Epochs`` input failed; microstate labels and
  ``GEV_per_microstate`` were incorrectly remapped after reordering; ``criterion="cv"``
  raised a ``TypeError``; backfitting is now invariant to the scale of the maps; the first
  microstate lifetime was overcounted by one sample (#1199).
* ``hrv_nonlinear()`` docstring now points to :func:`hrv_rqa` (#1207).
* ``fractal_dfa()``: the variance threshold used to discard numerically-zero detrended
  segments is now relative to the largest segment variance at each scale instead of an
  absolute ``1e-8`` cutoff, restoring invariance of ``h(q)`` under constant rescaling of the
  signal (#1208). Note: results computed with negative ``q`` on physiological signals in
  their natural units may shift slightly; the new values are the more correct ones.
* ``ecg_delineate(method="dwt")``: the choice among candidate T and P peaks no longer depends on the
  units or gain of the signal (e.g., mV vs. uV). The amplitude term of the candidate score is now
  scale-free and weighted against the latency penalty (``amplitude_weight``), which also changes the
  detected peaks (and the onsets/offsets derived from them) for signals in their original units
  (#1210, #1220).
* ``epochs_create()`` applies baseline correction when ``epochs_start`` contains
  a list of per-event start times (#1223).

Maintenance
+++++++++++++

* Source distributions no longer ship the tests, docs and data folders.
* Broken image links in the README and documentation were fixed.

0.2.8
-------------------
New Features
+++++++++++++

* New feature `events_find()`: is now able to combine multiple digital input channels,
  retrieve events from the combined events channel and differentiate between the inputs that
  occur simultaneously.



0.2.4
-------------------
Fixes
+++++++++++++

* `eda_sympathetic()` has been reviewed: low-pass filter and resampling have been added to be in
  line with the original paper
* `eda_findpeaks()` using methods proposed in nabian2018 is reviewed and improved. Differentiation
  has been added before smoothing. Skin conductance response criteria have been revised based on
  the original paper.



0.2.1
-------------------
New Features
+++++++++++++

* Allow for input with NaNs and extrapolation in `signal_interpolate()`
* Add argument `method` in `find_outliers()`
* A lot (see https://github.com/neuropsychology/NeuroKit/pull/645)




0.2.0
-------------------
New Features
+++++++++++++

* Add new time-domain measures in `hrv_time()`: `Prc20NN`, `Prc80NN`, `MinNN`, and `MaxNN`





0.1.6
-------------------

Breaking Changes
+++++++++++++++++

* Argument `type` changed to `out` in `expspace()`


New Features
+++++++++++++

* Add new time-domain measures in `hrv_time()`: `Prc20NN`, `Prc80NN`, `MinNN`, and `MaxNN`
* Allow `fix_peaks()` to account for larger intervals

Fixes
+++++++++++++





0.1.5
-------------------

Breaking Changes
+++++++++++++++++

* Argument `r` changed to `radius` in `fractal_correlation()`
* Argument `r` changed to `tolerance` in entropy and complexity utility functions
* Argument `r_method` changed to `tolerance_method` in `complexity_optimize()`
* `complexity_lempelziv()`, `fractal_higuchi()`, `fractal_katz()`, `fractal_correlation()`, `fractal_dfa()`, `entropy_multiscale()`, `entropy_shannon()`, `entropy_approximate()`, `entropy_fuzzy()`, `entropy_sample()` now return a tuple consisting of the complexity index, and a dictionary comprising of the different parameters specific to the measure. For `fractal_katz()` and `entropy_shannon()`, the parameters dictionary is empty.
* Restructure `complexity` submodules with optimization files starting with `optim_*`, such as `optim_complexity_delay()`, `optim_complexity_dimension()`, `optim_complexity_k()`, `optim_complexity_optimize()`, and `optim_complexity_tolerance()`.
* `mutual_information()` moved from `stats` module to `complexity` module.

New Features
+++++++++++++

* Added various complexity indices: `complexity_hjorth()`, `complexity_hurst()`, `complexity_lyapunov()`, `complexity_rqa()`, `complexity_rr()`, `entropy_coalition()`, `entropy_permutation()`, `entropy_range()`, `entropy_spectral()`, `fractal_nld()`, `fractal_psdslope()`, `fractal_sda()`, `fractal_sevcik()`
* Added `mne_templateMRI()` as a helper to get MNE's template MRI.
* Added `eeg_source()` as a helper to perform source reconstruction.
* Added `eeg_source_extract()` to extract the activity from a brain region.
* Added `parallel_run()` in `misc` as a parallel processing utility function.
* Added `find_plateau()` in `misc` to find the point of plateau in an array of values.
* Added `write_csv()` in `data` to facilitate saving dataframes into multiple parts.
* Added more complexity-related functions, `entropy_cumulative_residual()`, `entropy_differential()`, `entropy_svd()`, `fractal_petrosian()`, and `information_fisher()`.
* Updates logic to find `kmax` in `fractal_higuchi()`
* Add RSP_Amplitude_Baseline in event-related analysis
* Add argument `add_firstsamples` in `mne_channel_extract()` to account for first sample attribute in mne raw objects
* Allow plotting of `mne.Epochs` in `epochs_plot()`
* Add `mne_crop()` to crop `mne` Raw objects with additional flexibility to specify first and last elements
* Plotting function in `eeg_badchannels()` to visualize overlay of individual EEG channels and highlighting of bad ones
* Add `eog_peaks()` as wrapper for `eog_findpeaks()`
* Allow `ecg_delineate()` to account for different heart rate


Fixes
+++++++++++++

* Ensure detected offset in `emg_activation()` is not beyond signal length
* Raise ValueError in `_hrv_sanitize_input()` if RRIs are detected instead of peaks
* Ensure that multifractal DFA indices returned by `fractal_mdfa()` is not Nan when array of slopes contains Nan (due to zero fluctuations)
* Documentation of respiration from peak/trough terminology to inhale/exhale onsets
* Change labelling in `rsp_plot()` from "inhalation peaks" and "exhalation troughs" to "peaks (exhalation onsets)" and "troughs (inhalation onsets)" respectively.
* Change RSP_Amplitude_Mean/Min/Max parameters to be corrected based on value closest to t=0 in event-related analysis, rather than using all pre-zero values.
* Have `rsp_rrv()` compute breath-to-breath intervals based on trough indices (inhalation onsets) rather than peak indices
* Compute `rsp_rate()` based on trough indices (rather than peak indices) in 'periods' method


0.1.4.1
-------------------

Fixes
+++++++++++++
* Adjust `kmax` parameter in `fractal_higuchi()` according to signal length as having `kmax` more than half of signal length leads to division by zero error
* Ensure that sanitization of input in `_hrv_dfa()` is done before windows for `DFA_alpha2` is computed
* `np.seterr` is added to `fractal_dfa()` to avoid returning division by zero warning which is an expected behaviour


0.1.4
-------------------

Breaking Changes
+++++++++++++++++

* `fractal_df()` now returns a dictionary of windows, fluctuations and the slope value (see documentation for more information. If `multifractal` is True, the dictionary additionally contains the parameters of the singularity spectrum (see `singularity_spectrum()` for more information)

New Features
+++++++++++++

* Add convenience function `intervals_to_peaks()` useful for RRI or BBI conversion to peak indices
* `hrv_nonlinear()` and `rrv_rsp()` now return the parameters of singularity spectrum for multifractal DFA analysis
* Add new complexity measures in `fractal_higuchi()`, `fractal_katz()` and `fractal_lempelziv()`
* Add new time-domain measures in `hrv_time()`: `SDANN` and `SDNNI`
* Add new non-linear measures in `hrv_nonlinear()`: `ShanEn`, `FuzzyEn`, `HFD`, `KFD` and `LZC`

Fixes
+++++++++++++

* Add path argument in `mne_data()` and throw warning to download mne datasets if data folder is not present
* The implementation of `TTIN` in `hrv_time()` is amended to its correct formulation.
* The default binsize used for RRI histogram in the computation of geometric HRV indices is set to 1 / 128 seconds


0.1.3
-------------------

Breaking Changes
+++++++++++++++++

* None

New Features
+++++++++++++

* Add internal function for detecting missing data points and forward filling missing values in `nk.*_clean()` functions
* Add computation of standard deviation in `eventrelated()` functions for *ECG_Rate_SD*, *EMG_Amplitude_SD*, *EOG_Rate_SD*, *PPG_Rate_SD*, *RSP_Rate_SD*, *RSP_Amplitude_SD*
* Add labelling for interval related features if a dictionary of dataframes is passed
* Retrun Q peaks and S Peaks information for wavelet-based methods in `nk.ecg_delineate()`

Fixes
+++++++++++++

* Fix epochs columns with `dtype: object` generated by `nk.epochs_create()`
* Bug fix ecg_findpeaks_rodrigues for array out of bounds bug


0.1.2
-------------------

New Features
+++++++++++++

* Additional features for `nk.rsp_intervalrelated()`: average inspiratory and expiratory durations, inspiratory-to-expiratory (I/E) time ratio
* Add multiscale entropy measures (MSE, CMSE, RCMSE) and fractal methods (Detrended Fluctuation Analysis, Correlation Dimension) into `nk.hrv_nonlinear()`
* Allow for data resampling in `nk.read_bitalino()`
* Add `bio_resting_8min_200hz` into database for reading with `nk.data()`
* Reading of url links in `nk.data()`
* Allow for `nk.hrv()` to compute RSA indices if respiratory data is present
* All `hrv` functions to automatically detect correct sampling rate if tuple or dict is passed as input
* Add support for PPG analysis: `nk.ppg_eventrelated()`, `nk.ppg_intervalrelated()`, `nk.ppg_analyze()`
* Add Zhao et al. (2018) method for `nk.ecg_quality()`
* Add tests for `epochs` module
* Add sub-epoch option for ECG and RSP event-related analysis:
	* users can create a smaller sub-epoch within the event-related epoch
	* the rate-related features of ECG and RSP signals are calculated over the sub-epoch
	* the remaining features are calculated over the original epoch, not the sub-epoch

Fixes
+++++++++++++

* Fix propagation of values in `nk.signal_formatpeaks()` for formatting SCR column outputs generated by `eda_peaks()`
* Fix docstrings of `nk.rsp_phase()`, from "RSP_Inspiration" to "RSP_Phase"
* Update `signal_filter()` method for `rsp_clean()`: to use `sos` form, instead of `ba` form of butterworth (similar to `eda_clean()`)





0.1.1
-------------------

New Features
+++++++++++++

* Use duration from `nk.events_find()` as `epochs_end` in `nk.epochs_create()`
* Allow customized subsets of epoch lengths in `nk.bio_analyze()` with `window_lengths` argument
* Add `nk.find_outliers()` to identify outliers (abnormal values)
* Add utility function - `nk.check_type()` to return appropriate boolean values of input (integer, list, ndarray, pandas dataframe or pandas series)
* (experimental) Add error bars in the summary plot method to illustrate standard error of each bin


Fixes
+++++++++++++

* Fix type of value in `nk.signal_formatpeaks()` to ensure slice assignment is done on the same type


0.0.1 (2019-10-29)
-------------------

* First release on PyPI.
