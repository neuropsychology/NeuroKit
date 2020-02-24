import pandas as pd
import numpy as np

from ..signal import signal_filter


def ecg_rsp(ecg_rate, sampling_rate=1000, method="charlton2016"):
    """
    Extract ECG Derived Respiration (EDR)

    This implementation is far from being complete, as the information in the related papers
    prevents me from getting a full understanding of the procedure. Help is required!

    Parameters
    ----------
    ecg_rate : array
        The heart rate signal as obtained via `ecg_rate()`.
    sampling_rate : int
        The sampling frequency of the signal that contains the R-peaks (in Hz,
        i.e., samples/second). Defaults to 1000Hz.
    desired_length : int
        By default, the returned heart rate has the same number of elements as
        peaks. If set to an integer, the returned heart rate will be
        interpolated between R-peaks over `desired_length` samples. Has no
        effect if a DataFrame is passed in as the `peaks` argument. Defaults to
        None.

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
    >>> data = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit/master/data/example_bio_100hz.csv")
    >>> rpeaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)
    >>> ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=100)
    >>>
    >>>
    >>> # Get ECG Derived Respiration (EDR)
    >>> edr = nk.ecg_rsp(ecg_rate, sampling_rate=100)
    >>> nk.standardize(pd.DataFrame({"EDR": edr,
                                    "RSP": data["RSP"]})).plot()
    >>>
    >>> # Method comparison (the closer to 0 the better)
    >>> nk.standardize(pd.DataFrame(
            {"sarkar2015": nk.ecg_rsp(ecg_rate, sampling_rate=100, method="sarkar2015") - data["RSP"],
             "charlton2016": nk.ecg_rsp(ecg_rate, sampling_rate=100, method="charlton2016") - data["RSP"]})).plot()
    """
    method = method.lower()
    if method in ["sarkar2015"]:
        rsp = _ecg_rsp_sarkar2015(ecg_rate, sampling_rate=sampling_rate)
    elif method in ["charlton2016"]:
        rsp = _ecg_rsp_charlton2016(ecg_rate, sampling_rate=sampling_rate)
    else:
        raise ValueError("NeuroKit error: ecg_rsp(): 'method' should be "
                         "one of 'sarkar2015' or 'charlton2016'.")

    return rsp



# =============================================================================
# Methods
# =============================================================================
def _ecg_rsp_sarkar2015(ecg_rate, sampling_rate=1000):
    """
    https://www.researchgate.net/publication/304221962_Extraction_of_respiration_signal_from_ECG_for_respiratory_rate_estimation
    """
    rsp = signal_filter(ecg_rate,
                        sampling_rate=sampling_rate,
                        lowcut=0.1,
                        highcut=0.7,
                        method="butterworth",
                        order=6)
    return rsp



def _ecg_rsp_charlton2016(ecg_rate, sampling_rate=1000):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390977/#__ffn_sectitle
    """
    # 4-60 Bpm bandpass
    rsp = signal_filter(ecg_rate,
                        sampling_rate=sampling_rate,
                        lowcut=4/60,
                        highcut=60/60,
                        method="butterworth",
                        order=6)
    return rsp
