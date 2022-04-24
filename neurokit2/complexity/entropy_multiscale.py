# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .complexity_coarsegraining import _get_scales, complexity_coarsegraining
from .entropy_approximate import entropy_approximate
from .entropy_sample import entropy_sample
from .optim_complexity_tolerance import complexity_tolerance
from .utils import _get_coarsegrained_rolling, _phi, _phi_divide


def entropy_multiscale(
    signal,
    scale="default",
    dimension=2,
    tolerance="sd",
    method="MSEn",
    show=False,
    **kwargs,
):
    """**Multiscale entropy (MSEn) and its Composite (CMSEn), Refined (RCMSEn) or fuzzy versions**

    One of the limitation of :func:`SampEn <entropy_sample>` is that it characterizes
    complexity strictly on the time scale defined by the sampling procedure (via the ``delay``
    argument). To address this, Costa et al. (2002) proposed the multiscale entropy (MSEn),
    which compute sample entropies at multiple scales.

    The conventional MSEn algorithm consists of two steps:
    1. A :func:`coarse-graining <complexity_coarsegraining>` procedure is used to represent the
       signal at different time scales.
    2. SampEn is used to quantify the regularity of a coarse-grained time series at each time scale
       factor.

    However, in the traditional coarse-graining procedure, the larger the scale factor is, the
    shorter the coarse-grained time series is. As such, the variance of the entropy of the
    coarse-grained series estimated by SampEn increases as the time scale factor increases, making
    it problematic for shorter signals.

    - **CMSEn**: In order to reduce the variance of estimated entropy values at large scales, Wu et
      al. (2013) introduced the **Composite Multiscale Entropy** algorithm, which computes
      multiple coarse-grained series for each scale factor (via the **time-shift** method for
      :func:`coarse-graining <complexity_coarsegraining>`).
    - **RCMSEn**: Wu et al. (2014) further **Refined** their CMSEn by averaging not the entropy
      values of each subcoarsed vector, but its components at a lower level.
    - **MMSEn**: Wu et al. (2013) also introduced the **Modified Multiscale Entropy**
      algorithm, which is based on rolling-average :func:`coarse-graining <complexity_coarsegraining>`.
    - **IMSEn**: Liu et al. (2012) introduced an adaptive-resampling procedure to resample the
      coarse-grained series. We implement a generalization of this via interpolation that can be
      referred to as **Interpolated Multiscale Entropy**.

    Their :func:`**Fuzzy** <entropy_sample>` version can be obtained by setting ``fuzzy=True``.

    This function can be called either via ``entropy_multiscale()`` or ``complexity_mse()``.
    Moreover, variants can be directly accessed via ``complexity_cmse()``, `complexity_rcmse()``,
    ``complexity_fuzzymse()``, ``complexity_fuzzycmse()`` and ``complexity_fuzzyrcmse()``.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
        or dataframe.
    scale : str or int or list
        A list of scale factors used for coarse graining the time series. If 'default', will use
        ``range(len(signal) / (dimension + 10))`` (see discussion
        `here <https://github.com/neuropsychology/NeuroKit/issues/75#issuecomment-583884426>`_).
        If 'max', will use all scales until half the length of the signal. If an integer, will
        create a range until the specified int.
    dimension : int
        Embedding Dimension (*m*, sometimes referred to as *d* or *order*). See
        :func:`complexity_dimension()` to estimate the optimal value for this parameter.
    tolerance : float
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance()` to estimate the optimal value for this parameter.
    method : str
        What version of multiscale entropy to compute. Can be one of ``"MSEn"``, ``"CMSEn"``,
        ``"RCMSEn"``, ``"MMSEn"``, or ``"IMSEn"``.
    show : bool
        Show the entropy values for each scale factor.
    **kwargs
        Optional arguments.


    Returns
    ----------
    float
        The point-estimate of multiscale entropy (MSEn) of the single time series corresponding to
        the area under the MSEn values curve, which is essentially the sum of sample entropy values
        over the range of scale factors.
    dict
        A dictionary containing additional information regarding the parameters used
        to compute multiscale entropy. The entropy values corresponding to each ``"Scale"``
        factor are stored under the ``"Value"`` key.

    See Also
    --------
    entropy_shannon, entropy_approximate, entropy_sample, entropy_fuzzy, entropy_permutation

    Examples
    ----------
    * **MSE** (basic coarse-graining)
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=2, frequency=[5, 12, 40])

      @savefig p_entropy_multiscale1.png scale=100%
      msen, info = nk.entropy_multiscale(signal, show=True)
      @suppress
      plt.close()

    * **CMSE** (time-shifted coarse-graining)
    .. ipython:: python

      @savefig p_entropy_multiscale2.png scale=100%
      cmsen, info = nk.entropy_multiscale(signal, method="CMSE", show=True)
      @suppress
      plt.close()

    * **RCMSE** (refined composite MSE)
    .. ipython:: python

      @savefig p_entropy_multiscale3.png scale=100%
      rcmsen, info = nk.entropy_multiscale(signal, method="RCMSE", show=True)
      @suppress
      plt.close()

    * **MMSE** (rolling-window coarse-graining)
    .. ipython:: python

      @savefig p_entropy_multiscale4.png scale=100%
      mmsen, info = nk.entropy_multiscale(signal, method="MMSE", show=True)
      @suppress
      plt.close()

    * **IMSE** (interpolated coarse-graining)
    .. ipython:: python

      @savefig p_entropy_multiscale5.png scale=100%
      imsen, info = nk.entropy_multiscale(signal, method="IMSE", show=True)
      @suppress
      plt.close()

    * **MSApEn** (based on ApEn instead of SampEn)
    .. ipython:: python

      @savefig p_entropy_multiscale6.png scale=100%
      msapen, info = nk.entropy_multiscale(signal, method="MSApEn", show=True)
      @suppress
      plt.close()

    * **FuzzyMSEn**
    .. ipython:: python

      @savefig p_entropy_multiscale7.png scale=100%
      fuzzymsen, info = nk.entropy_multiscale(signal, method="MSEn", fuzzy=True, show=True)
      @suppress
      plt.close()

    * **FuzzyCMSEn**
    .. ipython:: python

      fuzzycmsen, info = nk.entropy_multiscale(signal, method="CMSEn", fuzzy=True)
      fuzzycmsen

    * **FuzzyRCMSEn**
    .. ipython:: python

      fuzzyrcmsen, info = nk.entropy_multiscale(signal, method="RCMSEn", fuzzy=True)
      fuzzycmsen

    References
    -----------
    * Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy analysis of complex
      physiologic time series. Physical review letters, 89(6), 068102.
    * Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological
      signals. Physical review E, 71(2), 021906.
    * Wu, S. D., Wu, C. W., Lee, K. Y., & Lin, S. G. (2013). Modified multiscale entropy for
      short-term time series analysis. Physica A: Statistical Mechanics and its Applications, 392
      (23), 5865-5873.
    * Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). Time series analysis
      using composite multiscale entropy. Entropy, 15(3), 1069-1084.
    * Wu, S. D., Wu, C. W., Lin, S. G., Lee, K. Y., & Peng, C. K. (2014). Analysis of complex time
      series using refined composite multiscale entropy. Physics Letters A, 378(20), 1369-1374.
    * Gow, B. J., Peng, C. K., Wayne, P. M., & Ahn, A. C. (2015). Multiscale entropy analysis of
      center-of-pressure dynamics in human postural control: methodological considerations. Entropy,
      17(12), 7926-7947.
    * Norris, P. R., Anderson, S. M., Jenkins, J. M., Williams, A. E., & Morris Jr, J. A. (2008).
      Heart rate multiscale entropy at three hours predicts hospital mortality in 3,154 trauma
      patients. Shock, 30(1), 17-22.
    * Liu, Q., Wei, Q., Fan, S. Z., Lu, C. W., Lin, T. Y., Abbod, M. F., & Shieh, J. S. (2012).
      Adaptive computation of multiscale entropy and its application in EEG signals for monitoring
      depth of anesthesia during surgery. Entropy, 14(6), 978-992.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )
    # Prevent multiple arguments error in case 'delay' is passed in kwargs
    if "delay" in kwargs.keys():
        kwargs.pop("delay")

    # Parameters selection
    algorithm = entropy_sample
    refined = False
    if method in ["MSE", "MSEn"]:
        coarsegraining = "nonoverlapping"
    elif method in ["MSApEn"]:
        coarsegraining = "nonoverlapping"
        algorithm = entropy_approximate
    elif method in ["MMSE", "MMSEn"]:
        coarsegraining = "rolling"
    elif method in ["IMSE", "IMSEn"]:
        coarsegraining = "interpolate"
    elif method in ["CMSE", "CMSEn"]:
        coarsegraining = "timeshift"
    elif method in ["RCMSE", "RCMSEn"]:
        coarsegraining = "timeshift"
        refined = True
    else:
        raise ValueError(
            "Method '{method}' is not supported. Please use "
            "'MSEn', 'CMSEn', 'RCMSEn', 'MMSEn', 'MSApEn'"
            " or 'IMSEn'."
        )

    # Store parameters
    info = {
        "Method": method,
        "Coarsegraining": coarsegraining,
        "Dimension": dimension,
        "Scale": _get_scales(signal, scale=scale, dimension=dimension),
        "Tolerance": complexity_tolerance(
            signal,
            method=tolerance,
            dimension=dimension,
            show=False,
        )[0],
    }

    # Compute entropy for each coarsegrained segment
    info["Value"] = np.array(
        [
            _entropy_multiscale(
                coarse=complexity_coarsegraining(
                    signal,
                    scale=scale,
                    method=coarsegraining,
                    show=False,
                    **kwargs,
                ),
                algorithm=algorithm,
                dimension=dimension,
                tolerance=info["Tolerance"],
                refined=refined,
                **kwargs,
            )
            for scale in info["Scale"]
        ]
    )

    # Remove inf, nan and 0
    mse = info["Value"][np.isfinite(info["Value"])]

    # The MSE index is quantified as the area under the curve (AUC),
    # which is like the sum normalized by the number of values. It's similar to the mean.
    mse = np.trapz(mse) / len(mse)

    # Plot overlay
    if show is True:
        _entropy_multiscale_plot(mse, info)

    return mse, info


# =============================================================================
# Internal
# =============================================================================
def _entropy_multiscale_plot(mse, info):

    fig = plt.figure(constrained_layout=False)
    fig.suptitle("Entropy values across scale factors")
    plt.title(f"(Total {info['Method']} = {np.round(mse, 3)})")
    plt.ylabel("Entropy values")
    plt.xlabel("Scale")
    plt.plot(
        info["Scale"][np.isfinite(info["Value"])],
        info["Value"][np.isfinite(info["Value"])],
        color="#FF9800",
    )

    return fig


# =============================================================================
# Methods
# =============================================================================
# Define function that works both on 1D and 2D coarse-grained (for composite)
def _entropy_multiscale(coarse, algorithm, dimension, tolerance, refined=False, **kwargs):
    # For 1D coarse-graining
    if coarse.ndim == 1:
        return algorithm(
            coarse,
            delay=1,
            dimension=dimension,
            tolerance=tolerance,
            **kwargs,
        )[0]

    # 2D coarse-graining (time-shifted, used in composite)
    else:
        # CMSE
        if refined is False:
            return _validmean(
                [
                    algorithm(
                        coarse[i],
                        delay=1,
                        dimension=dimension,
                        tolerance=tolerance,
                        **kwargs,
                    )[0]
                    for i in range(len(coarse))
                ]
            )
        # RCMSE
        else:
            phis = np.array(
                [
                    _phi(
                        coarse[i],
                        delay=1,
                        dimension=dimension,
                        tolerance=tolerance,
                        approximate=False,
                    )
                    for i in range(len(coarse))
                ]
            )
            # Average all phi of the same dimension, then divide, then log
            return _phi_divide([_validmean(phis[:, 0]), _validmean(phis[:, 1])])


# Mean that is robust to NaN,
def _validmean(x):
    x = np.array(x)[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    else:
        return np.mean(x)
