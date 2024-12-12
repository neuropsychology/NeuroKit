# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import pandas as pd

from .hrv_utils import _hrv_format_input


def hrv_symdyn(
    peaks,
    sampling_rate: int = 1000,
    quantization_level_equal_proba: List = [4, 6],
    quantization_level_max_min: List = [6],
    sigma_rate: List = [0.05],
) -> pd.DataFrame:
    """**Computes symbolic dynamics of Heart Rate Variability (HRV)**

    This function calculates the HRV symbolic dynamics indices based on three transformation methods:
    equal probability, max-min, and sigma methods. It uses the series of R-R intervals to derive these
    indices, quantifying the dynamics through symbolic analysis.
    Those parameters are calculated for each argument values for the given method.

    Parameters
    ----------
    peaks : dict or list
        Samples at which cardiac extrema (e.g., R-peaks) occur. Can be a list of indices or
        a dict containing the keys `RRI` and `RRI_Time` to directly pass the R-R intervals and their timestamps.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur, by default 1000.
    quantization_level_equal_proba : List[int], optional
        List of quantization levels for the equal probability method, by default [4,6].
    quantization_level_max_min : List[int], optional
        List of quantization levels for the max-min method, by default [6].
    sigma_rate : List[float], optional
        List of sigma rates for the sigma method, by default [0.05].

    Returns
    -------
    DataFrame
        Contains the HRV symbolic dynamics indices calculated using the specified methods
        (default, this may vary for non-default arguments):
    * **SymDynMaxMin4_0V**: Represents the percentage of sequences with zero variation (all symbols are equal)
        derived using the Max–Min method, where the RR intervals are quantized into six levels based on equal ranges
        from the minimum to the maximum value.
    * **SymDynMaxMin4_1V**: Indicates the percentage of sequences with one variation (exactly one different symbol
        in the sequence) using the Max–Min method.
    * **SymDynMaxMin4_2LV**: Reflects the percentage of sequences with two like variations (all symbols are different
        and form an increasing or decreasing sequence) in the Max–Min method.
    * **SymDynMaxMin4_2UV**: Shows the percentage of sequences with two unlike variations (symbols vary in opposite
        directions, forming a peak or valley) in the Max–Min method.
    * **SymDynSigma0.05_0V**: Represents the percentage of sequences with zero variation, quantized based on the signal
        average and a sigma rate adjustment, using three levels.
    * **SymDynSigma0.05_1V**: Indicates the percentage of sequences with one variation, derived using the σ method.
    * **SymDynSigma0.05_2LV**: Reflects the percentage of sequences with two like variations, as quantized by the σ method.
    * **SymDynSigma0.05_2UV**: Shows the percentage of sequences with two unlike variations, according to the σ method.
    * **SymDynEqualPorba4_0V**: Represents the percentage of sequences with zero variation, derived using the
        Equal-Probability method with quantization level 4, ensuring each level has the same number of points.
    * **SymDynEqualPorba4_1V**: Indicates the percentage of sequences with one variation, using the Equal-Probability
        method with quantization level 4.
    * **SymDynEqualPorba4_2LV**: Reflects the percentage of sequences with two like variations, in the Equal-Probability
        method with quantization level 4.
    * **SymDynEqualPorba4_2UV**: Shows the percentage of sequences with two unlike variations, derived with the
        Equal-Probability method at quantization level 4.
    * **SymDynEqualPorba6_0V**: Represents the percentage of sequences with zero variation, quantized by the
        Equal-Probability method with quantization level 6, for a direct comparison with the σ method and Max–Min method.
    * **SymDynEqualPorba6_1V**: Indicates the percentage of sequences with one variation, using the Equal-Probability
        method with quantization level 6.
    * **SymDynEqualPorba6_2LV**: Reflects the percentage of sequences with two like variations, in the Equal-Probability
        method with quantization level 6.
    * **SymDynEqualPorba6_2UV**: Shows the percentage of sequences with two unlike variations, quantized by the
        Equal-Probability method with quantization level 6.


    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_time, hrv_frequency, hrv_summary, hrv_nonlinear

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV indices
      hrv = nk.hrv_symdyn(peaks, sampling_rate=100)


    References
    ----------
    * Cysarz, D., Edelhäuser, F., Javorka, M., Montano, N., and Porta, A. (2018). On the relevance of symbolizing heart rate
        variability by means of a percentile-based coarse graining approach. Physiol. Meas. 39:105010.
        doi: 10.1088/1361-6579/aae302
    * Cysarz, D., Porta, A., Montano, N., Leeuwen, P. V., Kurths, J., and Wessel, N. (2013). Quantifying heart rate dynamics
        using different approaches of symbolic dynamics. Eur. Phys. J. Spec. Top. 222, 487–500.
        doi: 10.1140/epjst/e2013-01854-7
    * Wessel, N., Malberg, H., Bauernschmitt, R., and Kurths, J. (2007). Nonlinear methods of cardiovascular physics and
        their clinical applicability. Int. J. Bifurc. Chaos 17, 3325–3371. doi: 10.1142/s0218127407019093
    * Porta, A., Tobaldini, E., Guzzetti, S., Furlan, R., Montano, N., and Gnecchi-Ruscone, T. (2007). Assessment of cardiac
        autonomic modulation during graded head-up tilt by symbolic analysis of heart rate variability.
        Am. J. Physiol. Heart Circ. Physiol. 293, H702–H708. doi: 10.1152/ajpheart.00006.2007
    * Gąsior, J. S., Rosoł, M., Młyńczak, M., Flatt, A. A., Hoffmann, B., Baranowski, R., Werner, B. (2022). Reliability
        of Symbolic Analysis of Heart Rate Variability and Its Changes During Sympathetic Stimulation in Elite Modern
        Pentathlon Athletes: A Pilot Study. Front. Physiol. 13, doi: 10.3389/fphys.2022.829887

    """
    rri, _, _ = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    out = []
    for quantization_level in quantization_level_equal_proba:
        out.append(equal_probability_method(rri, quantization_level))
    for quantization_level in quantization_level_max_min:
        out.append(max_min_method(rri, quantization_level))
    for sigma in sigma_rate:
        out.append(sigma_method(rri, sigma))

    out = pd.concat(out, axis=1)

    return out


def get_families_from_symbols(symbols: np.array) -> dict:
    """Extracts symbolic dynamics families from a sequence of symbols.

    This function generates words from a given sequence of symbols and classifies these words into
    predefined families based on their variation pattern. The classification counts are then normalized
    and returned as a dictionary.

    Parameters
    ----------
    symbols : np.array
        An array of symbols derived from R-R intervals or any other time series data.

    Returns
    -------
    dict
        A dictionary with keys corresponding to the symbolic dynamics families ('0V', '1V', '2LV', '2UV')
        and values representing the normalized counts of words belonging to each family.

    """
    words = form_words(symbols)
    families = classify_and_count(words)

    return families


def max_min_method(rri: np.array, quantization_level: int = 6) -> pd.DataFrame:
    """Calculates HRV symbolic dynamics indices using the Max-Min method.

    This method converts the series of R-R intervals into a series of symbols through uniform quantization
    across specified levels. The function then classifies sequences of symbols into families based on their
    variation pattern and computes the percentage of each family type.

    Parameters
    ----------
    rri : np.array
        The R-R intervals extracted from the heartbeat time series.
    quantization_level : int, optional
        The number of levels to use for quantization, by default 6.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the percentage of symbol sequences classified into each variation family.

    """
    min_val, max_val = np.min(rri), np.max(rri)
    thresholds = np.linspace(min_val, max_val, quantization_level + 1)[1:-1]
    symbols = np.digitize(rri, thresholds)

    families = get_families_from_symbols(symbols)

    out = pd.DataFrame.from_dict(families, orient="index").T.add_prefix(f"HRV_SymDynMaxMin{quantization_level}_")

    return out


def sigma_method(rri: np.array, sigma_rate: float = 0.05) -> pd.DataFrame:
    """Calculates HRV symbolic dynamics indices using the sigma method.

    The sigma method defines symbols based on the deviation of R-R intervals from the mean, adjusted by
    a factor of sigma_rate. Sequences of symbols are classified into families based on their variation
    pattern, and the function calculates the percentage of each family type.

    Parameters
    ----------
    rri : np.array
        The R-R intervals extracted from the heartbeat time series.
    sigma_rate : float, optional
        The sigma rate used to adjust the mean for symbol classification, by default 0.05.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the percentage of symbol sequences classified into each variation family.

    """
    mu = np.mean(rri)  # Calculate the mean (μ) of RR intervals
    # Transform RR intervals into symbols based on the given thresholds
    symbols = np.zeros(rri.shape, dtype=int)

    # Conditions for assigning symbols
    symbols[rri > (1 + sigma_rate) * mu] = 0
    symbols[(mu < rri) & (rri <= (1 + sigma_rate) * mu)] = 1
    symbols[(rri <= mu) & (rri > (1 - sigma_rate) * mu)] = 2
    symbols[rri <= (1 - sigma_rate) * mu] = 3

    families = get_families_from_symbols(symbols)

    out = pd.DataFrame.from_dict(families, orient="index").T.add_prefix(f"HRV_SymDynSigma{sigma_rate}_")

    return out


def equal_probability_method(rri: np.array, quantization_level: int = 4) -> pd.DataFrame:
    """Calculates HRV symbolic dynamics indices using the Equal-Probability method.

    This method divides the full range of R-R intervals into levels with equal probability, ensuring each level
    contains the same number of points. Sequences of symbols are classified into families based on their variation
    pattern, and the function calculates the percentage of each family type.

    Parameters
    ----------
    rri : np.array
        The R-R intervals extracted from the heartbeat time series.
    quantization_level : int, optional
        The number of quantization levels, by default 4.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the percentage of symbol sequences classified into each variation family.

    """
    percentiles = np.linspace(0, 100, quantization_level + 1)
    # Find the values at those percentiles in the RR interval data
    percentile_values = np.percentile(rri, percentiles)
    # Digitize the RR intervals according to the percentile values
    # np.digitize bins values into the rightmost bin, so we subtract 1 to correct this
    symbols = np.digitize(rri, percentile_values, right=False) - 1
    # Ensure all symbols are within the range 0 to quantization_level-1
    symbols[symbols == -1] = 0
    symbols[symbols == quantization_level] = quantization_level - 1

    families = get_families_from_symbols(symbols)

    out = pd.DataFrame.from_dict(families, orient="index").T.add_prefix(f"HRV_SymDynEqualPorba{quantization_level}_")

    return out


def form_words(symbols: np.array) -> List:
    """Forms consecutive words of length 3 from a sequence of symbols.

    This function iterates over a given sequence of symbols and groups them into consecutive
    words of three symbols each. These words are then used for further analysis, such as
    classifying into families based on their variation pattern.

    Parameters
    ----------
    symbols : np.array
        An array of symbols, usually derived from quantized R-R intervals or similar time series data.

    Returns
    -------
    List
        A list of words, where each word is an array of three consecutive symbols from the original sequence.

    """
    words = [symbols[i : i + 3] for i in range(len(symbols) - 2)]

    return words


def classify_and_count(words: List) -> dict:
    """Classifies words into families based on their variation pattern and counts them.

    This function takes a list of words (each a sequence of three symbols) and classifies
    them into four families based on their variation pattern: '0V' for no variation, '1V'
    for one variation, '2LV' for two like variations (sequential increase or decrease),
    and '2UV' for two unlike variations (peak or valley). The function then normalizes
    the counts of words in each family and returns them as a dictionary.

    Parameters
    ----------
    words : List
        A list of words, where each word is a sequence of three symbols.

    Returns
    -------
    dict
        A dictionary with keys corresponding to the symbolic dynamics families ('0V', '1V', '2LV', '2UV')
        and values representing the normalized counts of words belonging to each family.

    """
    families = {"0V": 0, "1V": 0, "2LV": 0, "2UV": 0}
    for word in words:
        unique_elements = len(set(word))
        if unique_elements == 1:
            families["0V"] += 1
        elif unique_elements == 2:
            families["1V"] += 1
        elif unique_elements == 3:
            if (word[1] > word[0] and word[2] > word[1]) or (word[1] < word[0] and word[2] < word[1]):
                families["2LV"] += 1
            else:
                families["2UV"] += 1

    for key in families.keys():
        families[key] = families[key] / len(words)
    return families
