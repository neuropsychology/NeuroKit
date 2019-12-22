import numpy
from .embedded_sequence import embed_seq


def fisher_info(X, Tau, DE, W=None):
    """Compute SVD Entropy from either two cases below:
    1. a time series X, with lag tau and embedding dimension dE (default)
    2. a list, W, of normalized singular values of a matrix (if W is provided,
    recommend to speed up.)

    If W is None, the function will do as follows to prepare singular spectrum:

        First, computer an embedding matrix from X, Tau and DE using pyeeg
        function embed_seq():
                    M = embed_seq(X, Tau, DE)

        Second, use scipy.linalg function svd to decompose the embedding matrix
        M and obtain a list of singular values:
                    W = svd(M, compute_uv=0)

        At last, normalize W:
                    W /= sum(W)

    Notes
    -------------

    To speed up, it is recommended to compute W before calling this function
    because W may also be used by other functions whereas computing it here
    again will slow down.
    """

    if W is None:
        Y = embed_seq(X, Tau, DE)
        W = numpy.linalg.svd(Y, compute_uv=0)
        W /= sum(W)  # normalize singular values

    return -1 * sum(W * numpy.log(W))
