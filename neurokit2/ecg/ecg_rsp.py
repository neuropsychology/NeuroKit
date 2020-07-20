from ..signal import signal_filter


def ecg_rsp(ecg_rate, sampling_rate=1000, method="vangent2019"):
    """Extract ECG Derived Respiration (EDR).

    This implementation is far from being complete, as the information in the related papers
    prevents me from getting a full understanding of the procedure. Help is required!

    Parameters
    ----------
    ecg_rate : array
        The heart rate signal as obtained via `ecg_rate()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000Hz.
    method : str
        Can be one of 'vangent2019' (default), 'soni2019', 'charlton2016' or 'sarkar2015'.

    Returns
    -------
    array
        A Numpy array containing the heart rate.

    Examples
    --------
    >>> import neurokit2 as nk
    >>> import pandas as pd
    >>>
    >>> # Get heart rate
    >>> data = nk.data("bio_eventrelated_100hz")
    >>> rpeaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>> ecg_rate = nk.signal_rate(rpeaks, sampling_rate=100, desired_length=len(rpeaks))
    >>>
    >>>
    >>> # Get ECG Derived Respiration (EDR)
    >>> edr = nk.ecg_rsp(ecg_rate, sampling_rate=100)
    >>> nk.standardize(pd.DataFrame({"EDR": edr, "RSP": data["RSP"]})).plot() #doctest: +ELLIPSIS
    <AxesSubplot:>
    >>>
    >>> # Method comparison (the closer to 0 the better)
    >>> nk.standardize(pd.DataFrame({"True RSP": data["RSP"],
    ...                              "vangent2019": nk.ecg_rsp(ecg_rate, sampling_rate=100, method="vangent2019"),
    ...                              "sarkar2015": nk.ecg_rsp(ecg_rate, sampling_rate=100, method="sarkar2015"),
    ...                              "charlton2016": nk.ecg_rsp(ecg_rate, sampling_rate=100, method="charlton2016"),
    ...                              "soni2019": nk.ecg_rsp(ecg_rate, sampling_rate=100,
    ...                                                     method="soni2019")})).plot() #doctest: +ELLIPSIS
    <AxesSubplot:>

    References
    ----------
    - van Gent, P., Farah, H., van Nes, N., & van Arem, B. (2019). HeartPy: A novel heart rate algorithm
      for the analysis of noisy signals. Transportation research part F: traffic psychology and behaviour,
      66, 368-378.

    - Sarkar, S., Bhattacherjee, S., & Pal, S. (2015). Extraction of respiration signal from ECG for
      respiratory rate estimation.

    - Charlton, P. H., Bonnici, T., Tarassenko, L., Clifton, D. A., Beale, R., & Watkinson, P. J. (2016).
      An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram.
      Physiological measurement, 37(4), 610.

    - Soni, R., & Muniyandi, M. (2019). Breath rate variability: a novel measure to study the meditation
      effects. International Journal of Yoga, 12(1), 45.

    """
    method = method.lower()
    if method in [
        "sarkar2015"
    ]:  # https://www.researchgate.net/publication/304221962_Extraction_of_respiration_signal_from_ECG_for_respiratory_rate_estimation # noqa: E501
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=0.1, highcut=0.7, order=6)
    elif method in ["charlton2016"]:  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390977/#__ffn_sectitle
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=4 / 60, highcut=60 / 60, order=6)
    elif method in ["soni2019"]:  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6329220/
        rsp = signal_filter(ecg_rate, sampling_rate, highcut=0.5, order=6)
    elif method in [
        "vangent2019"
    ]:  # https://github.com/paulvangentcom/heartrate_analysis_python/blob/1597e8c0b2602829428b22d8be88420cd335e939/heartpy/analysis.py#L541 # noqa: E501
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=0.1, highcut=0.4, order=2)
    else:
        raise ValueError(
            "NeuroKit error: ecg_rsp(): 'method' should be "
            "one of 'sarkar2015', 'charlton2016', 'soni2019' or "
            "'vangent2019'."
        )

    return rsp
