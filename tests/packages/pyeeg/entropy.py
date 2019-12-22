import numpy
from .embedded_sequence import embed_seq


def ap_entropy(X, M, R):
    """Computer approximate entropy (ApEN) of series X, specified by M and R.

    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1

    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.

    Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k. The probability that a random row in Em matches Em[i] is
    \simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1),
    denoted as Cm[i].

    We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M
    since the length of each sequence in Emp is M + 1.

    The probability that any two embedding sequences in Em match is then
    sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
    Phi_mp = sum(log(Cmp)) / (N - M ).

    And the ApEn is defined as Phi_m - Phi_mp.


    Notes
    -----
    Please be aware that self-match is also counted in ApEn.

    References
    ----------
    Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005

    See also
    --------
    samp_entropy: sample entropy of a time series

    """
    N = len(X)

    Em = embed_seq(X, 1, M)
    A = numpy.tile(Em, (len(Em), 1, 1))
    B = numpy.transpose(A, [1, 0, 2])
    D = numpy.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = numpy.max(D, axis=2) <= R

    # Probability that random M-sequences are in range
    Cm = InRange.mean(axis=0)

    # M+1-sequences in range if M-sequences are in range & last values are close
    Dp = numpy.abs(
        numpy.tile(X[M:], (N - M, 1)) - numpy.tile(X[M:], (N - M, 1)).T
    )

    Cmp = numpy.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)

    Phi_m, Phi_mp = numpy.sum(numpy.log(Cm)), numpy.sum(numpy.log(Cmp))

    Ap_En = (Phi_m - Phi_mp) / (N - M)

    return Ap_En




def permutation_entropy(x, n, tau):
    """Compute Permutation Entropy of a given time series x, specified by
    permutation order n and embedding lag tau.

    Parameters
    ----------

    x
        list

        a time series

    n
        integer

        Permutation order

    tau
        integer

        Embedding lag

    Returns
    ----------

    PE
       float

       permutation entropy

    Notes
    ----------
    Suppose the given time series is X =[x(1),x(2),x(3),...,x(N)].
    We first build embedding matrix Em, of dimension(n*N-n+1),
    such that the ith row of Em is x(i),x(i+1),..x(i+n-1). Hence
    the embedding lag and the embedding dimension are 1 and n
    respectively. We build this matrix from a given time series,
    X, by calling pyEEg function embed_seq(x,1,n).

    We then transform each row of the embedding matrix into
    a new sequence, comprising a set of integers in range of 0,..,n-1.
    The order in which the integers are placed within a row is the
    same as those of the original elements:0 is placed where the smallest
    element of the row was and n-1 replaces the largest element of the row.

    To calculate the Permutation entropy, we calculate the entropy of PeSeq.
    In doing so, we count the number of occurrences of each permutation
    in PeSeq and write it in a sequence, RankMat. We then use this sequence to
    calculate entropy by using Shannon's entropy formula.

    Permutation entropy is usually calculated with n in range of 3 and 7.

    References
    ----------
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural
    complexity measure for time series." Physical Review Letters 88.17
    (2002): 174102.


    Examples
    ----------
    >>> import pyeeg
    >>> x = [1,2,4,5,12,3,4,5]
    >>> pyeeg.permutation_entropy(x,5,1)
    2.0

    """

    PeSeq = []
    Em = embed_seq(x, tau, n)

    for i in range(0, len(Em)):
        r = []
        z = []

        for j in range(0, len(Em[i])):
            z.append(Em[i][j])

        for j in range(0, len(Em[i])):
            z.sort()
            r.append(z.index(Em[i][j]))
            z[z.index(Em[i][j])] = -1

        PeSeq.append(r)

    RankMat = []

    while len(PeSeq) > 0:
        RankMat.append(PeSeq.count(PeSeq[0]))
        x = PeSeq[0]
        for j in range(0, PeSeq.count(PeSeq[0])):
            PeSeq.pop(PeSeq.index(x))

    RankMat = numpy.array(RankMat)
    RankMat = numpy.true_divide(RankMat, RankMat.sum())
    EntropyMat = numpy.multiply(numpy.log2(RankMat), RankMat)
    PE = -1 * EntropyMat.sum()

    return PE

def spectral_entropy(X, Band, Fs, Power_Ratio=None):
    """Compute spectral entropy of a time series from either two cases below:
    1. X, the time series (default)
    2. Power_Ratio, a list of normalized signal power in a set of frequency
    bins defined in Band (if Power_Ratio is provided, recommended to speed up)

    In case 1, Power_Ratio is computed by bin_power() function.

    Notes
    -----
    To speed up, it is recommended to compute Power_Ratio before calling this
    function because it may also be used by other functions whereas computing
    it here again will slow down.

    Parameters
    ----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    As indicated in return line

    See Also
    --------
    bin_power: pyeeg function that computes spectral power in frequency bins

    """

    if Power_Ratio is None:
        Power, Power_Ratio = bin_power(X, Band, Fs)

    Spectral_Entropy = 0
    for i in range(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])
    Spectral_Entropy /= numpy.log(
        len(Power_Ratio)
    )  # to save time, minus one is omitted
    return -1 * Spectral_Entropy


def svd_entropy(X, Tau, DE, W=None):
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

def samp_entropy(X, M, R):
    """Computer sample entropy (SampEn) of series X, specified by M and R.
    SampEn is very close to ApEn.
    
    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1
    
    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.
    
    Pick Em[i] as a template, for all j such that 0 < j < N - M , we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k.
    
    We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.
    The SampEn is defined as log(sum(Cm)/sum(Cmp))

    References
    ----------
    Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biological
    signals, Physical Review E, 71:021906, 2005

    See also
    --------
    ap_entropy: approximate entropy of a time series
    """

    N = len(X)

    Em = embed_seq(X, 1, M)
    A = numpy.tile(Em, (len(Em), 1, 1))
    B = numpy.transpose(A, [1, 0, 2])
    D = numpy.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
    InRange = numpy.max(D, axis=2) <= R
    numpy.fill_diagonal(InRange, 0)  # Don't count self-matches

    Cm = InRange.sum(axis=0)  # Probability that random M-sequences are in range
    Dp = numpy.abs(
        numpy.tile(X[M:], (N - M, 1)) - numpy.tile(X[M:], (N - M, 1)).T
    )

    Cmp = numpy.logical_and(Dp <= R, InRange[:-1, :-1]).sum(axis=0)

    # Avoid taking log(0)
    Samp_En = numpy.log(numpy.sum(Cm + 1e-100) / numpy.sum(Cmp + 1e-100))

    return Samp_En
