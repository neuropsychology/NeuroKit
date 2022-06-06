import numpy as np
import scipy.stats


# ==============================================================================
# THIS IS WIP and we would like to implement an EOG simulator. Please help!
# ==============================================================================
def _eog_simulate_blink(sampling_rate=1000, length=None, method="scr", parameters="default"):
    """**Simulate a canonical blink from vertical EOG**

    Recommended parameters:

    * For method ``"scr"``: ``[3.644, 0.422, 0.356, 0.943]``
    * For method ``"gamma"``: ``[2.659, 5.172, 0.317]``

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      blink_scr = _eog_simulate_blink(sampling_rate=100,
                                      method='scr',
                                      parameters=[3.644, 0.422, 0.356, 0.943])
      blink_gamma = _eog_simulate_blink(sampling_rate=100,
                                        method='gamma',
                                        parameters=[2.659, 5.172, 0.317])
      @savefig p_eog_simulate1.png scale=100%
      nk.signal_plot([blink_scr, blink_gamma], sampling_rate=100)
      @suppress
      plt.close()

    """
    if length is None:
        length = int(sampling_rate)

    x = np.linspace(0, 10, num=length)

    if method.lower() == "scr":
        if isinstance(parameters, str):
            parameters = [3.644, 0.422, 0.356, 0.943]
        gt = np.exp(-((x - parameters[0]) ** 2) / (2 * parameters[1] ** 2))
        ht = np.exp(-x / parameters[2]) + np.exp(-x / parameters[3])

        ft = np.convolve(gt, ht)
        ft = ft[0 : len(x)]
        y = ft / np.max(ft)

    else:
        if isinstance(parameters, str):
            parameters = [2.659, 5.172, 0.317]
        gamma = scipy.stats.gamma.pdf(x, a=parameters[1], loc=parameters[0], scale=parameters[2])
        y = gamma / np.max(gamma)
    return y
