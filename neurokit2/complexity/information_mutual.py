# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.special
import scipy.stats
import sklearn.metrics
import sklearn.neighbors


def mutual_information(x, y, method="varoquaux", bins="default", **kwargs):
    """Mutual Information (MI)

    Computes the mutual information (MI) between two vectors from a joint histogram.
    The mutual information of two variables is a measure of the mutual dependence between them.
    More specifically, it quantifies the "amount of information" obtained about one variable by
    observing the other variable.

    Different methods are available:
    * **nolitsa**: Standard mutual information (a bit faster than the ``"sklearn"`` method).
    * **varoquaux**: Applies a Gaussian filter on the joint-histogram. The smoothing amount can be
      modulated via the ``sigma`` argument (by default, ``sigma=1``).
    * **knn**: Non-parametric (i.e., not based on binning) estimation via nearest neighbors.
      Additional parameters includes ``k`` (by default, ``k=3``), the number of nearest neighbors
      to use.
    * **max**: Maximum Mutual Information coefficient, i.e., the MI is maximal given a certain
      combination of number of bins.


    Parameters
    ----------
    x : Union[list, np.array, pd.Series]
        A vector of values.
    y : Union[list, np.array, pd.Series]
        A vector of values.
    method : str
        The method to use.
    bins : int
        Number of bins to use while creating the histogram. Only used for ``"nolitsa"`` and
        ``"varoquaux"``. If ``"default"``, the number of bins is estimated following
        Hacine-Gharbi (2018).
    **kwargs
        Additional keyword arguments to pass to the chosen method.

    Returns
    -------
    float
        The computed similarity measure.

    See Also
    --------
    information_fisher

    Examples
    ---------
    **Example 1**: Simple case

    .. ipython:: python

      import neurokit2 as nk

      x = [3, 3, 5, 1, 6, 3, 2, 8, 1, 2, 3, 5, 4, 0, 2]
      y = [5, 3, 1, 3, 4, 5, 6, 4, 1, 3, 4, 6, 2, 1, 3]

      nk.mutual_information(x, y, method="varoquaux")
      nk.mutual_information(x, y, method="nolitsa")
      nk.mutual_information(x, y, method="knn")
      nk.mutual_information(x, y, method="max")
      nk.mutual_information(x, y, method="gc")

    **Example 2**: Method comparison

    .. ipython:: python

      import numpy as np
      import pandas as pd

      x = np.random.normal(size=400)
      y = x**2

      data = pd.DataFrame()
      for level in np.linspace(0.01, 3, 200):
          noise = np.random.normal(scale=level, size=400)
          rez = pd.DataFrame({"Noise": [level]})
          rez["MI1"] = nk.mutual_information(x, y + noise, method="varoquaux", sigma=1)
          rez["MI2"] = nk.mutual_information(x, y + noise, method="varoquaux", sigma=0)
          rez["MI3"] = nk.mutual_information(x, y + noise, method="nolitsa")
          rez["MI4"] = nk.mutual_information(x, y + noise, method="knn")
          rez["MI5"] = nk.mutual_information(x, y + noise, method="max")
          rez["MI6"] = nk.mutual_information(x, y + noise, method="gc")
          data = pd.concat([data, rez], axis=0)

      # Rescale on the same range for visualization purposes
      data["MI1"] = nk.rescale(data["MI1"])
      data["MI2"] = nk.rescale(data["MI2"])
      data["MI3"] = nk.rescale(data["MI3"])
      data["MI4"] = nk.rescale(data["MI4"])
      data["MI5"] = nk.rescale(data["MI5"])
      data["MI6"] = nk.rescale(data["MI6"])

      @savefig p_information_mutual1.png scale=100%
      data.plot(x="Noise", y=["MI1", "MI2", "MI3", "MI4", "MI5", "MI6"], kind="line")
      @suppress
      plt.close()

      # Computation time
      # x = np.random.normal(size=10000)
      # %timeit nk.mutual_information(x, x**2, method="varoquaux")
      # %timeit nk.mutual_information(x, x**2, method="nolitsa")
      # %timeit nk.mutual_information(x, x**2, method="sklearn")
      # %timeit nk.mutual_information(x, x**2, method="knn", k=2)
      # %timeit nk.mutual_information(x, x**2, method="knn", k=5)
      # %timeit nk.mutual_information(x, x**2, method="max")


    References
    ----------
    * Studholme, C., Hawkes, D. J., & Hill, D. L. (1998, June). Normalized entropy measure for
      multimodality image alignment. In Medical imaging 1998: image processing (Vol. 3338, pp.
      132-143). SPIE.
    * Hacine-Gharbi, A., & Ravier, P. (2018). A binning formula of bi-histogram for joint entropy
      estimation using mean square error minimization. Pattern Recognition Letters, 101, 21-28.

    """
    method = method.lower()

    if method in ["max"] or isinstance(bins, (list, np.ndarray)):
        # https://www.freecodecamp.org/news/
        # how-machines-make-predictions-finding-correlations-in-complex-data-dfd9f0d87889/
        if isinstance(bins, str):
            bins = np.arange(2, np.ceil(len(x) ** 0.6) + 1).astype(int)

        mi = 0
        for i in bins:
            for j in bins:
                if i * j > np.max(bins):
                    continue
                p_x = pd.cut(x, i, labels=False)
                p_y = pd.cut(y, j, labels=False)
                new_mi = _mutual_information_sklearn(p_x, p_y) / np.log2(np.min([i, j]))
                if new_mi > mi:
                    mi = new_mi
    else:
        if isinstance(bins, str):
            # Hacine-Gharbi (2018)
            # https://stats.stackexchange.com/questions/179674/
            # number-of-bins-when-computing-mutual-information
            bins = 1 + np.sqrt(1 + (24 * len(x) / (1 - np.corrcoef(x, y)[0, 1] ** 2)))
            bins = np.round((1 / np.sqrt(2)) * np.sqrt(bins)).astype(int)

        if method in ["varoquaux"]:
            mi = _mutual_information_varoquaux(x, y, bins=bins, **kwargs)
        elif method in ["shannon", "nolitsa"]:
            mi = _mutual_information_nolitsa(x, y, bins=bins)
        elif method in ["sklearn"]:
            mi = _mutual_information_sklearn(x, y, bins=bins)
        elif method in ["knn"]:
            mi = _mutual_information_knn(x, y, **kwargs)
        elif method in ["gc"]:
            mi = _mutual_information_gc(x, y)
        else:
            raise ValueError("NeuroKit error: mutual_information(): 'method' not recognized.")

    return mi


# =============================================================================
# Methods
# =============================================================================

# SCIKIT-LEARN ----------------------------------------------------------------
def _mutual_information_sklearn(x, y, bins=None):
    if bins is None:
        _, p_xy = scipy.stats.contingency.crosstab(x, y)
    else:
        p_xy = np.histogram2d(x, y, bins)[0]
    return sklearn.metrics.mutual_info_score(None, None, contingency=p_xy)


# Varoquaux -------------------------------------------------------------------
def _mutual_information_varoquaux(x, y, bins=256, sigma=1, normalized=True):
    """Based on Gael Varoquaux's implementation:
    https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429."""
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    scipy.ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

    # compute marginal histograms
    jh = jh + np.finfo(float).eps
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2))

    return mi


# nolitsa ---------------------------------------------------------------------
def _mutual_information_nolitsa(x, y, bins=256):
    """
    Based on the nolitsa package:
    https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/delay.py#L72

    """
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.
    # Also, in the limit p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y


# NPEET -----------------------------------------------------------------------
def _mutual_information_knn(x, y, k=3):
    """
    Based on the NPEET package:
    https://github.com/gregversteeg/NPEET

    """
    points = np.array([x, y]).T

    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = sklearn.neighbors.KDTree(points, metric="chebyshev").query(points, k=k + 1)[0][:, k]

    a = np.array([x]).T
    a = sklearn.neighbors.KDTree(a, metric="chebyshev").query_radius(
        a, dvec - 1e-15, count_only=True
    )
    a = np.mean(scipy.special.digamma(a))

    b = np.array([y]).T
    b = sklearn.neighbors.KDTree(b, metric="chebyshev").query_radius(
        b, dvec - 1e-15, count_only=True
    )
    b = np.mean(scipy.special.digamma(b))

    c = scipy.special.digamma(k)
    d = scipy.special.digamma(len(x))
    return (-a - b + c + d) / np.log(2)


# Copula --------------------------------------------------------------------

# TODO: Add Gaussian-Copula Mutual Information
# A package implements Gaussian-Copula Mutual Information
# But it gives somewhat unexpected results
# https://github.com/robince/gcmi


def _mutual_information_gc(x, y, biascorrect=False, demeaned=True):
    """Gaussian-Copula Mutual Information between two continuous variables.
    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis)
    This provides a lower bound to the true MI value.


    Mutual information (MI) between two Gaussian variables in bits

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    biascorrect : true / false option (default true) which specifies whether
    bias correction should be applied to the esimtated MI.
    demeaned : false / true option (default false) which specifies whether th
    input data already has zero mean (true if it has been copula-normalized)
    """

    x = copnorm(np.atleast_2d(x))
    y = copnorm(np.atleast_2d(y))

    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx + Nvary

    # joint variable
    xy = np.vstack((x, y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:, np.newaxis]
    Cxy = np.dot(xy, xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx, :Nvarx]
    Cy = Cxy[Nvarx:, Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx)))  # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy)))  # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diagonal(chCxy)))  # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = scipy.special.psi((Ntrl - np.arange(1, Nvarxy + 1)).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary * dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I


def copnorm(x):
    """Copula normalization

    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.

    Copula transformation (empirical CDF)
    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).
    """

    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr + 1).astype(float) / (xr.shape[-1] + 1)
    # cx = scipy.stats.norm.ppf(ctransform(x))
    return scipy.special.ndtri(cx)
