# -*- coding: utf-8 -*-
import numpy as np

from .entropy_sample import entropy_sample


def entropy_quadratic(signal, delay=1, dimension=2, tolerance="sd", **kwargs):
    """**Quadratic Sample Entropy (QSE)**

    Compute the quadratic sample entropy (QSE) of a signal. It is essentially a correction of
    SampEn introduced by Lake (2005) defined as:

    .. math::

        QSE = SampEn + ln(2 * tolerannce)

    QSE has been described as a more stable measure of entropy than SampEn (Gylling, 2017).

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*) in samples.
        See :func:`complexity_delay` to estimate the optimal value for this parameter.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    **kwargs : optional
        Other arguments.

    See Also
    --------
    entropy_sample

    Returns
    ----------
    qse : float
        The  uadratic sample entropy of the single time series.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute sample entropy.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=5)

      qsa, parameters = nk.entropy_quadratic(signal, delay=1, dimension=2)
      qsa

    References
    ----------
    * Huselius Gylling, K. (2017). Quadratic sample entropy as a measure of burstiness: A study in
      how well Rényi entropy rate and quadratic sample entropy can capture the presence of spikes in
      time-series data.
    * Lake, D. E. (2005). Renyi entropy measures of heart rate Gaussianity. IEEE Transactions on
      Biomedical Engineering, 53(1), 21-27.

    """
    sampen, info = entropy_sample(
        signal,
        delay=delay,
        dimension=dimension,
        tolerance=tolerance,
        **kwargs,
    )
    return sampen + np.log(2 * info["Tolerance"]), info
