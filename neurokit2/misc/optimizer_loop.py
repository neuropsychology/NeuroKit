import pandas as pd
from ..stats import mad
from ..complexity import complexity_optimize
from ..complexity.fractal_higuchi import _fractal_higuchi_optimal_k


def optimizer_loop(signal, func=complexity_optimize, **kwargs):
    """To loop through a dictionary of signals and identify an optimal parameter.

    Parameters
    ----------
    signal : dict
        A dictionary of signals (i.e., time series) in the form of an array of values.
    func : function
        A function used to optimize the parameters. The function can be `complexity_optimize` or
        `_fractal_higuchi_optimal_k`
    **kwargs : key-word arguments
        For `complexity_optimize`, `maxnum` can be specified to identify the nearest neighbors.

    Returns
    -------
    out : dict
        A dictionary consists of a dataframe with optimal values of all parameters corresponding to
        the respective signals (i.e. one row per signal, one column per parameter) and a dictionary
        of one optimal value for each parameter (i.e. one value per parameter for all signals).
        The selected optimal value for each parameter corresponds to its median absolute deviation
        value.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >> list_participants = os.listdir()[:5]
    >> all_rri = {}
    >> sampling_rate=100
    >> for participant in list_participants:
    >>    path = participant + "/RestingState/" + participant
    >>    bio, sampling_rate = nk.read_acqknowledge(path + "_RS" + ".acq",sampling_rate=sampling_rate)

    >>    # Select columns
    >>    bio = bio[['ECG A, X, ECG2-R', 'Digital input']]

    >>    # New column in df for participant ID
    >>    bio['Participant'] = np.full(len(bio), participant)
    >>    events = nk.events_find(bio["Digital input"], threshold_keep="below")["onset"]
    >>    df = bio.iloc[events[0]:events[0] + 8 * 60 * sampling_rate]
    >>    peaks, info = nk.ecg_peaks(df["ECG A, X, ECG2-R"], sampling_rate=sampling_rate)
    >>    rpeaks = peaks["ECG_R_Peaks"].values
    >>    rpeaks = np.where(rpeaks == 1)[0]
    >>    rri = np.diff(rpeaks) / sampling_rate * 1000
    >>    all_rri[str(participant)] = rri

    >> nk.optimizer_loop(all_rri, func=complexity_optimize, maxnum=90)
    """

    metrics = {}
    for _, (name, sig) in enumerate(signal.items()):
        metrics[str(name)] = func(sig, **kwargs)

    df = pd.DataFrame(metrics).T
    optimize = {}
    for _, metric in enumerate(df.columns):
        optimize[str(metric)] = mad(df[metric])
    out = {
            'All_Values': df,
            'Optimal_Value': optimize
            }

    return out
