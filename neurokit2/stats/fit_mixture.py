# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.mixture


def fit_mixture(X=None, n_clusters=2):
    """**Gaussian Mixture Model**

    Performs a polynomial regression of given order.

    Parameters
    ----------
    X : Union[list, np.array, pd.Series]
        The values to classify.
    n_clusters : int
        Number of components to look for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the probability of belonging to each cluster.
    dict
        Dictionary containing additional information such as the parameters (:func:`.n_clusters`).

    See Also
    ----------
    signal_detrend, fit_error

    Examples
    ---------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      x = nk.signal_simulate()
      probs, info = nk.fit_mixture(x, n_clusters=2)  # Rmb to merge with main to return ``info``
      @savefig p_fit_mixture.png scale=100%
      fig = nk.signal_plot([x, probs["Cluster_0"], probs["Cluster_1"]], standardize=True)
      @suppress
      plt.close()

    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # fit a Gaussian Mixture Model with two components
    clf = sklearn.mixture.GaussianMixture(n_components=n_clusters, random_state=333)
    clf = clf.fit(X)

    # Get predicted probabilities
    predicted = clf.predict_proba(X)
    probabilities = pd.DataFrame(predicted).add_prefix("Cluster_")

    return probabilities, {"n_clusters": n_clusters}
