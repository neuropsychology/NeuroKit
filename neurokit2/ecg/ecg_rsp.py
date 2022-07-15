from ..signal import signal_filter


def ecg_rsp(ecg_rate, sampling_rate=1000, method="vangent2019"):
    """**ECG-Derived Respiration (EDR)**

    Extract ECG-Derived Respiration (EDR), a proxy of a respiratory signal based on heart rate.


    """
    # TODO: It would be interesting to run a study in which we modulate the different filtering
    # parameters and compute the difference with the real RSP signal, and then suggest the optimal
    # filtering parameters. If you're interested in helping out let us know!
    method = method.lower()
    if method in ["sarkar2015"]:
        # https://www.researchgate.net/publication/304221962_Extraction_of_respiration_signal_from_ECG_for_respiratory_rate_estimation # noqa: E501
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=0.1, highcut=0.7, order=6)
    elif method in ["charlton2016"]:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390977/#__ffn_sectitle
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=4 / 60, highcut=60 / 60, order=6)
    elif method in ["soni2019"]:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6329220/
        rsp = signal_filter(ecg_rate, sampling_rate, highcut=0.5, order=6)

    elif method in ["vangent2019"]:
        # https://github.com/paulvangentcom/heartrate_analysis_python/blob/1597e8c0b2602829428b22d8be88420cd335e939/heartpy/analysis.py#L541 # noqa: E501
        rsp = signal_filter(ecg_rate, sampling_rate, lowcut=0.1, highcut=0.4, order=2)
    else:
        raise ValueError(
            "`method` should be one of 'sarkar2015', 'charlton2016', 'soni2019' or "
            "'vangent2019'."
        )

    return rsp
