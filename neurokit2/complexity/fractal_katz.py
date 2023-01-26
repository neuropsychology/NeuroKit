# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def fractal_katz(signal):
    """**Katz's Fractal Dimension (KFD)**

    Computes Katz's Fractal Dimension (KFD). The euclidean distances between successive points in
    the signal are summed and averaged, and the maximum distance between the starting point and any
    other point in the sample.

    Fractal dimensions range from 1.0 for straight lines, through approximately 1.15 for
    random-walks, to approaching 1.5 for the most convoluted waveforms.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    Returns
    -------
    kfd : float
        Katz's fractal dimension of the single time series.
    info : dict
        A dictionary containing additional information (currently empty, but returned nonetheless
        for consistency with other functions).

    See Also
    --------
    fractal_linelength

    Examples
    ----------
    * **Step 1.** Simulate different kinds of signals

    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      # Simulate straight line
      straight = np.linspace(-1, 1, 2000)

      # Simulate random
      random = nk.complexity_simulate(duration=2, method="randomwalk")
      random = nk.rescale(random, [-1, 1])

      # Simulate simple
      simple = nk.signal_simulate(duration=2, frequency=[5, 10])

      # Simulate complex
      complex = nk.signal_simulate(duration=2,
                                   frequency=[1, 3, 6, 12],
                                   noise = 0.1)

      @savefig p_katz.png scale=100%
      nk.signal_plot([straight, random, simple, complex])

    * **Step 2.** Compute KFD for each of them

    .. ipython:: python

      KFD, _ = nk.fractal_katz(straight)
      KFD
      KFD, _ = nk.fractal_katz(random)
      KFD
      KFD, _ = nk.fractal_katz(simple)
      KFD
      KFD, _ = nk.fractal_katz(complex)
      KFD

    References
    ----------
    * Katz, M. J. (1988). Fractals and the analysis of waveforms.
      Computers in Biology and Medicine, 18(3), 145-156. doi:10.1016/0010-4825(88)90041-8.

    """

    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Force to array
    signal = np.array(signal)

    # Drop missing values
    signal = signal[~np.isnan(signal)]

    # Define total length of curve
    dists = np.abs(np.diff(signal))
    length = np.sum(dists)

    # Average distance between successive points
    a = np.mean(dists)

    # Compute farthest distance between starting point and any other point
    d = np.max(np.abs(signal - signal[0]))

    kfd = np.log10(length / a) / (np.log10(d / a))

    return kfd, {}
