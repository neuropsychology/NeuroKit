# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd




def rsp_process(rsp, sampling_rate=1000):
    """RSP (respiration) signal processing.

    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>>
    >>> signal = np.cos(np.linspace(start=0, stop=50, num=10000))
    >>> peaks_info = nk.rsp_process(signal)
    >>>
    >>> results = nk.rsp_rate(peaks_info, desired_length=len(signal))
    >>> results["RSP_Signal"] = signal
    >>> nk.standardize(results).plot()
    """
    preprocessed = nk.rsp_prepare(rsp, sampling_rate=sampling_rate)

        # Extremas (peaks and troughs)
    info = rsp_findpeaks(filtered_rsp, sampling_rate=sampling_rate, outlier_threshold=0.3)

    rate = nk.rsp_rate(peaks=info["RSP_Peaks"],
                       troughs=info["RSP_Troughs"],
                       desired_length=len(rsp),
                       sampling_rate=1000)

    data = pd.concat([preprocessed, rate])
    out = {"RSP_data": data,
           "RSP_info": info}

    return(out)


