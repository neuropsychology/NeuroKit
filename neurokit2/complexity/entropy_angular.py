import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from .utils_complexity_embedding import complexity_embedding


def entropy_angular(signal, delay=1, dimension=2, show=False, **kwargs):
    """**Angular entropy (AngEn)**

    The Angular Entropy (AngEn) is the name that we use in NeuroKit to refer to the complexity
    method described in Nardelli et al. (2022), referred as comEDA due to its application to EDA
    signal. The method comprises the following steps: 1) Phase space reconstruction, 2) Calculation
    of the angular distances between all the pairs of points in the phase space; 3) Computation of
    the probability density function (PDF) of the distances; 4) Quadratic Rényi entropy of the PDF.

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
    **kwargs : optional
        Other arguments.

    Returns
    --------
    angen : float
        The Angular Entropy (AngEn) of the signal.
    info : dict
        A dictionary containing additional information regarding the parameters used
        to compute the index.

    See Also
    --------
    entropy_renyi

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate a Signal with Laplace Noise
      signal = nk.signal_simulate(duration=2, frequency=[5, 3], noise=0.1)

      # Compute Angular Entropy
      @savefig p_entropy_angular1.png scale=100%
      angen, info = nk.entropy_angular(signal, delay=1, dimension=3, show=True)
      @suppress
      plt.close()


    References
    -----------
    * Nardelli, M., Greco, A., Sebastiani, L., & Scilingo, E. P. (2022). ComEDA: A new tool for
      stress assessment based on electrodermal activity. Computers in Biology and Medicine, 150,
      106144.

    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # 1. Phase space reconstruction (time-delay embeddings)
    embedded = complexity_embedding(signal, delay=delay, dimension=dimension)

    # 2. Angular distances between all the pairs of points in the phase space
    angles = _angular_distance(embedded)

    # 3. Compute the probability density function (PDF) of the upper triangular matrix
    bins, pdf = _kde_sturges(angles)

    # 4. Apply the quadratic Rényi entropy to the PDF
    angen = -np.log2(np.sum(pdf**2))

    # Normalize to the range [0, 1] by the log of the number of bins

    # Note that in the paper (eq. 4 page 4) there is a minus sign, but adding it would give
    # negative values, plus the linked code does not seem to do that
    # https://github.com/NardelliM/ComEDA/blob/main/comEDA.m#L103
    angen = angen / np.log2(len(bins))

    if show is True:
        # Plot the PDF as a bar chart
        plt.bar(bins[:-1], pdf, width=bins[1] - bins[0], align="edge", alpha=0.5)
        # Set the x-axis limits to the range of the data
        plt.xlim([np.min(angles), np.max(angles)])
        # Print titles
        plt.suptitle(f"Angular Entropy (AngEn) = {angen:.3f}")
        plt.title("Distribution of Angular Distances:")

    return angen, {"bins": bins, "pdf": pdf}


def _angular_distance(m):
    """
    Compute angular distances between all the pairs of points.
    """
    # Get index of upper triangular to avoid double counting
    idx = np.triu_indices(m.shape[0], k=1)

    # compute the magnitude of each vector
    magnitudes = np.linalg.norm(m, axis=1)

    # compute the dot product between all pairs of vectors using np.matmul function, which is
    # more efficient than np.dot for large matrices; and divide the dot product matrix by the
    # product of the magnitudes to get the cosine of the angle
    cos_angles = np.matmul(m, m.T)[idx] / np.outer(magnitudes, magnitudes)[idx]

    # clip the cosine values to the range [-1, 1] to avoid any numerical errors and compute angles
    return np.arccos(np.clip(cos_angles, -1, 1))


def _kde_sturges(x):
    """
    Computes the PDF of a vector x using a kernel density estimator based on linear diffusion
    processes with a Gaussian kernel. The number of bins of the PDF is chosen applying the Sturges
    method.
    """
    # Estimate the bandwidth
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    bandwidth = 0.9 * iqr / (len(x) ** 0.2)

    # Compute the number of bins using the Sturges method
    nbins = int(np.ceil(np.log2(len(x)) + 1))

    # Compute the bin edges
    bins = np.linspace(np.min(x), np.max(x), nbins + 1)

    # Compute the kernel density estimate
    xi = (bins[:-1] + bins[1:]) / 2
    pdf = np.sum(
        scipy.stats.norm.pdf((xi.reshape(-1, 1) - x.reshape(1, -1)) / bandwidth), axis=1
    ) / (len(x) * bandwidth)

    # Normalize the PDF
    pdf = pdf / np.sum(pdf)

    return bins, pdf
