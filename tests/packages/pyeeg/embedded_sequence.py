import numpy


def embed_seq(time_series, tau, embedding_dimension):
    """Build a set of embedding sequences from given time series `time_series`
    with lag `tau` and embedding dimension `embedding_dimension`.

    Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
    1 < i <  N - (embedding_dimension - 1) * tau,
    we build an embedding sequence,
    Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].

    All embedding sequences are placed in a matrix Y.

    Parameters
    ----------

    time_series
        list or numpy.ndarray

        a time series

    tau
        integer

        the lag or delay when building embedding sequence

    embedding_dimension
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-embedding_dimension list

        embedding matrix built

    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[0,  1,  2,  3],
           [1,  2,  3,  4],
           [2,  3,  4,  5],
           [3,  4,  5,  6],
           [4,  5,  6,  7],
           [5,  6,  7,  8]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[0,  2,  4],
           [1,  3,  5],
           [2,  4,  6],
           [3,  5,  7],
           [4,  6,  8]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8]])

    """
    if not type(time_series) == numpy.ndarray:
        typed_time_series = numpy.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return numpy.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
