import numpy


def LLE(x, tau, n, T, fs):
    """Calculate largest Lyauponov exponent of a given time series x using
    Rosenstein algorithm.

    Parameters
    ----------

    x
        list

        a time series

    n
        integer

        embedding dimension

    tau
        integer

        Embedding lag

    fs
        integer

        Sampling frequency

    T
        integer

        Mean period

    Returns
    ----------

    Lexp
       float

       Largest Lyapunov Exponent

    Notes
    ----------
    A n-dimensional trajectory is first reconstructed from the observed data by
    use of embedding delay of tau, using pyeeg function, embed_seq(x, tau, n).
    Algorithm then searches for nearest neighbour of each point on the
    reconstructed trajectory; temporal separation of nearest neighbours must be
    greater than mean period of the time series: the mean period can be
    estimated as the reciprocal of the mean frequency in power spectrum

    Each pair of nearest neighbours is assumed to diverge exponentially at a
    rate given by largest Lyapunov exponent. Now having a collection of
    neighbours, a least square fit to the average exponential divergence is
    calculated. The slope of this line gives an accurate estimate of the
    largest Lyapunov exponent.

    References
    ----------
    Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
    practical method for calculating largest Lyapunov exponents from small data
    sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.


    Examples
    ----------
    >>> import pyeeg
    >>> X = numpy.array([3,4,1,2,4,51,4,32,24,12,3,45])
    >>> pyeeg.LLE(X,2,4,1,1)
       0.18771136179353307

    """

    from embedded_sequence import embed_seq

    Em = embed_seq(x, tau, n)
    M = len(Em)
    A = numpy.tile(Em, (len(Em), 1, 1))
    B = numpy.transpose(A, [1, 0, 2])

    #  square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
    square_dists = (A - B) ** 2

    #  D[i,j] = ||Em[i]-Em[j]||_2
    D = numpy.sqrt(square_dists[:, :, :].sum(axis=2))

    # Exclude elements within T of the diagonal
    band = numpy.tri(D.shape[0], k=T) - numpy.tri(D.shape[0], k=-T - 1)
    band[band == 1] = numpy.inf

    # nearest neighbors more than T steps away
    neighbors = (D + band).argmin(axis=0)

    # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
    inc = numpy.tile(numpy.arange(M), (M, 1))
    row_inds = (numpy.tile(numpy.arange(M), (M, 1)).T + inc)
    col_inds = (numpy.tile(neighbors, (M, 1)) + inc.T)
    in_bounds = numpy.logical_and(row_inds <= M - 1, col_inds <= M - 1)

    # Uncomment for old (miscounted) version
    # in_bounds = numpy.logical_and(row_inds < M - 1, col_inds < M - 1)
    row_inds[~in_bounds] = 0
    col_inds[~in_bounds] = 0

    # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
    neighbor_dists = numpy.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)

    #  number of in-bounds indices by row
    J = (~neighbor_dists.mask).sum(axis=1)

    # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged
    neighbor_dists[neighbor_dists == 0] = 1
    d_ij = numpy.sum(numpy.log(neighbor_dists.data), axis=1)
    mean_d = d_ij[J > 0] / J[J > 0]

    x = numpy.arange(len(mean_d))
    X = numpy.vstack((x, numpy.ones(len(mean_d)))).T
    [m, c] = numpy.linalg.lstsq(X, mean_d)[0]
    Lexp = fs * m
    return Lexp


if __name__ == "__main__":
    import doctest
    doctest.testmod()
