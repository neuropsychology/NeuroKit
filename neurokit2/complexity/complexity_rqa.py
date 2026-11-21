import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .entropy_shannon import entropy_shannon
from .optim_complexity_tolerance import _recurrence_distance, complexity_tolerance
from .utils_recurrence_matrix import recurrence_matrix


def complexity_rqa(
    signal,
    dimension=3,
    delay=1,
    tolerance="recurrence",
    min_linelength=2,
    distance="euclidean",
    method="python",
    show=False,
):
    """**Recurrence Quantification Analysis (RQA)**

    A :func:`recurrence plot <recurrence_matrix>` is based on a time-delay embedding representation
    of a signal and is a 2D depiction of when a system revisits a state that it has been in the
    past.

    Recurrence quantification analysis (RQA) is a method of complexity analysis
    for the investigation of dynamical systems. It quantifies the number and duration
    of recurrences of a dynamical system presented by its phase space trajectory.

    .. figure:: ../img/douglas2022c.png
       :alt: Illustration of RQA (Douglas et al., 2022).

    Features include:

    * **Recurrence rate (RR)**: Proportion of points that are labelled as recurrences. Depends on
      the radius *r*.
    * **Determinism (DET)**: Proportion of recurrence points which form diagonal lines (of at
      least ``min_linelength`` points). Indicates autocorrelation.
    * **Divergence (DIV)**: The inverse of the longest diagonal line length (*LMax*).
    * **Laminarity (LAM)**: Proportion of recurrence points which form vertical lines. Indicates the
      amount of laminar phases (intermittency).
    * **Trapping Time (TT)**: Average length of vertical black lines.
    * **L**: Average length of diagonal black lines. Average duration that a system is staying in
      the same state.
    * **LEn**: Entropy of diagonal lines lengths.
    * **VMax**: Longest vertical line length.
    * **VEn**: Entropy of vertical lines lengths.
    * **W**: Average white vertical line length (i.e., the time the system takes to come back to a
      state, or recurrence time).
    * **WMax**: Longest white vertical line length.
    * **WEn**: Entropy of white vertical lines lengths.
    * **DeteRec**: The ratio of determinism / recurrence rate.
    * **LamiDet**: The ratio of laminarity / determinism.
    * **DiagRec**: Diagonal Recurrence Rates, capturing the magnitude of autocorrelation at
      different lags, which is related to fractal fluctuations. It is the slope of the log
      recurrence rate of each diagonal against its log distance to the main diagonal, over the
      first 64 diagonals, as in Tomashin et al. (2022, approach 3). The diagonals without
      recurrences are left out.

    The main diagonal (the line of identity, where each point is compared to itself) is excluded
    (i.e., a Theiler window of 1). As in Marwan et al. (2007) and RecurrenceAnalysis.jl, the
    diagonal and vertical lines are counted in the whole recurrence matrix (the vertical lines are
    interrupted by the line of identity), and the white vertical lines only between two recurrence
    points (the line of identity counts as one), as the white lines that reach the top or bottom
    border of the matrix are cut by the end of the recording. The entropies are in nats.

    **Distance and entropy measures**: with ``distance="chebyshev"`` (the largest difference
    between the coordinates, used by the entropy measures) and a delay of 1, two points are
    recurrent in dimension *m* + 1 if and only if they are recurrent in dimension *m* and so are the
    next two points, i.e., if their diagonal line continues. The sample entropy
    (:func:`entropy_sample`, with the same dimension and tolerance) is then exactly minus the log of
    the probability that a recurrence point is followed by another one on its diagonal, which ties
    it to the diagonal line measures (``Determinism``, ``L``, ``LEn``). The Euclidean distance (the
    default) breaks this equivalence: comparing the two distances helps separating what RQA and
    the entropy measures share by definition from what they share empirically. As the Chebyshev
    distance is smaller, it gives more recurrences for the same tolerance, so compare them at the
    same recurrence rate (e.g., with ``tolerance="recurrence"``).

    .. note::

      More feature exist for RQA, such as the `trend <https://juliadynamics.github.io/
      DynamicalSystems.jl/dev/rqa/quantification/#RecurrenceAnalysis.trend>`_. We would like to add
      them, but we need help. Get in touch if you're interested!

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
    tolerance : float
        Tolerance (often denoted as *r*, or radius), the distance (see ``distance``) under which two
        embedded points are considered as recurrent. If ``"recurrence"`` (default), will be set to
        the smallest value giving a recurrence rate of 1%. Other methods of
        :func:`complexity_tolerance` can be passed, but most are designed for the entropy
        measures, which use the (smaller) Chebyshev distance: e.g., ``"sd"``
        (:math:`0.2 * SD_{signal}`) leaves almost no recurrences in 3 dimensions or more with the
        Euclidean distance.
    min_linelength : int
        Minimum length of diagonal, vertical and white vertical lines. Default to 2.
    distance : str
        The distance between the embedded points: ``"euclidean"`` (default), ``"chebyshev"`` (the
        largest difference between their coordinates, as in the entropy measures) or
        ``"manhattan"`` (the sum of the absolute differences). See the note on the entropy
        measures above.
    method : str
        Can be ``"pyrqa"`` to use the *PyRQA* package (requires to install it first). Its
        conventions (e.g., for white lines) can differ.
    show : bool
        Visualise recurrence matrix.

    Returns
    ----------
    rqa : DataFrame
         The RQA results.
    info : dict
        A dictionary containing additional information regarding the parameters used to compute RQA.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 6, 7], noise=0.2)

      # RQA
      @savefig p_complexity_rqa1.png scale=100%
      results, info = nk.complexity_rqa(signal, tolerance=1, show=True)
      @suppress
      plt.close()

    .. ipython:: python

      results

      # Compare to PyRQA
      # results1, info = nk.complexity_rqa(signal, tolerance=1, show=True, method = "pyrqa")

    References
    ----------
    * Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis
      of complex systems. Physics Reports, 438(5-6), 237-329.
    * Rawald, T., Sips, M., Marwan, N., & Dransch, D. (2014). Fast computation of recurrences in
      long time series. In Translational Recurrences (pp. 17-29). Springer, Cham.
    * Tomashin, A., Leonardi, G., & Wallot, S. (2022). Four Methods to Distinguish between Fractal
      Dimensions in Time Series through Recurrence Quantification Analysis. Entropy, 24(9), 1314.

    """
    info = {
        "Tolerance": complexity_tolerance(signal, method=tolerance, delay=delay, dimension=dimension, distance=distance)[0],
        "Distance": distance,
    }

    if method == "pyrqa":
        data = _complexity_rqa_pyrqa(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=info["Tolerance"],
            linelength=min_linelength,
            distance=distance,
        )
        rc = np.flip(data.pop("Recurrence_Matrix"), axis=0)
        info["Recurrence_Matrix"] = rc

    else:
        # Get recurrence matrix (rm)
        rc, dm = recurrence_matrix(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=info["Tolerance"],
            distance=distance,
        )
        info["Recurrence_Matrix"] = rc
        info["Distance_Matrix"] = dm

        # Compute features
        data = _complexity_rqa_features(rc, min_linelength=min_linelength)

    data = pd.DataFrame(data, index=[0])

    if show is True:
        try:
            plt.imshow(rc, cmap="Greys")
            # Flip the matrix to match traditional RQA representation
            plt.gca().invert_yaxis()
            plt.title("Recurrence Matrix")
            plt.ylabel("Time")
            plt.xlabel("Time")
        except MemoryError as e:
            raise MemoryError(
                "NeuroKit error: complexity_rqa(): the recurrence plot is too large to display. ",
                "You can recover the matrix from the parameters and try to display parts of it.",
            ) from e

    return data, info


def _complexity_rqa_features(rc, min_linelength=2):
    """Compute the RQA features from a (symmetric) recurrence matrix (rc).

    The line of identity is excluded (Theiler window of 1). As the matrix is symmetric, the
    diagonal lines are counted above the line of identity only (the proportions and the length
    distribution are the same as in the full matrix). The vertical lines are counted in every
    column of the full matrix, above and below the line of identity, and the white vertical lines
    between two recurrence points of a column (the line of identity counts as one).
    """
    rc = np.asarray(rc, dtype=bool)
    width = len(rc)

    # Diagonals above the line of identity, one per row: diags[k - 1, i] = rc[i, i + k]
    diags = _rqa_diagonals(rc)

    # Recurrence Rate (RR)
    # --------------------------------------------------
    n_rec = np.count_nonzero(diags)  # Recurrence points above the line of identity
    n_pairs = width * (width - 1) / 2
    data = {"RecurrenceRate": n_rec / n_pairs if n_pairs > 0 else np.nan}

    # Diagonal Recurrence Rates (Diag %REC)
    # --------------------------------------------------
    # Tomashin et al. (2022, approach 3): slope of the log recurrence rate of the diagonals against
    # their log distance to the line of identity, over the first 64 diagonals (as in their code)
    distance = np.arange(1, min(width, 65))
    recdiag = np.count_nonzero(diags[: len(distance)], axis=1) / (width - distance)
    distance, recdiag = distance[recdiag > 0], recdiag[recdiag > 0]
    if len(recdiag) > 2:
        data["DiagRec"] = np.polyfit(np.log2(distance), np.log2(recdiag), 1)[0]
    else:
        data["DiagRec"] = np.nan

    # Diagonal lines
    # --------------------------------------------------
    diag_lengths = _rqa_lines(diags)
    del diags
    diag_lengths = diag_lengths[diag_lengths >= min_linelength]

    if n_rec == 0:
        data["Determinism"] = np.nan
        data["DeteRec"] = np.nan
    else:
        data["Determinism"] = diag_lengths.sum() / n_rec
        data["DeteRec"] = data["Determinism"] / data["RecurrenceRate"]
    data["L"] = 0 if len(diag_lengths) == 0 else np.mean(diag_lengths)
    data["Divergence"] = np.nan if len(diag_lengths) == 0 else 1 / np.max(diag_lengths)
    data["LEn"] = _rqa_entropy(diag_lengths)

    # Vertical lines
    # --------------------------------------------------
    # The matrix is symmetric, so its columns are read as rows (which is faster)
    rc = rc.copy()
    np.fill_diagonal(rc, False)  # Lines are interrupted by the line of identity
    black_lengths = _rqa_lines(rc)
    black_lengths = black_lengths[black_lengths >= min_linelength]

    if n_rec == 0:
        data["Laminarity"] = np.nan
    else:
        data["Laminarity"] = black_lengths.sum() / np.count_nonzero(rc)
    if data["Determinism"] == 0 or np.isnan(data["Determinism"]):
        data["LamiDet"] = np.nan
    else:
        data["LamiDet"] = data["Laminarity"] / data["Determinism"]
    data["TrappingTime"] = 0 if len(black_lengths) == 0 else np.mean(black_lengths)
    data["VMax"] = 0 if len(black_lengths) == 0 else np.max(black_lengths)
    data["VEn"] = _rqa_entropy(black_lengths)

    # White vertical lines (recurrence times)
    # --------------------------------------------------
    # Only the white lines between two recurrence points: the ones that reach the top or bottom
    # border are cut by the end of the recording.
    np.fill_diagonal(rc, True)
    white_lengths = _rqa_lines(rc, value=False, borders=False)
    white_lengths = white_lengths[white_lengths >= min_linelength]

    data["W"] = 0 if len(white_lengths) == 0 else np.mean(white_lengths)
    data["WMax"] = 0 if len(white_lengths) == 0 else np.max(white_lengths)
    data["WEn"] = _rqa_entropy(white_lengths)

    return data


def _rqa_diagonals(rc):
    """Diagonals above the line of identity of a square matrix, as the rows of an array.

    ``out[k - 1, i]`` is ``rc[i, i + k]`` (and 0 beyond the end of the diagonal), for ``k`` from 1
    to ``n - 1``. Each row of the matrix is padded with ``n`` zeros so that, read with a stride of
    ``2n + 1``, the consecutive elements of a diagonal are aligned in a column.
    """
    n = len(rc)
    flat = np.zeros(n * (2 * n + 1), dtype=bool)
    flat[: 2 * n * n].reshape(n, 2 * n)[:, :n] = rc
    return flat.reshape(n, 2 * n + 1)[:, 1:n].T


def _rqa_lines(x, value=True, borders=True):
    """Lengths of the runs of ``value`` along the rows of a 2D boolean array.

    If ``borders`` is False, the runs that start at the first or end at the last element of their
    row are left out.
    """
    n_cols = x.shape[1]
    # Pad each row with a 0 on both sides so that no run continues into the next row
    padded = np.zeros((x.shape[0], n_cols + 2), dtype=np.int8)
    padded[:, 1:-1] = x if value else ~x
    edges = np.diff(padded.ravel())
    starts = np.flatnonzero(edges == 1)  # The run starts at the next element
    ends = np.flatnonzero(edges == -1)  # The run ends at this element
    lengths = ends - starts
    if borders is False:
        lengths = lengths[(starts % (n_cols + 2) != 0) & (ends % (n_cols + 2) != n_cols)]
    return lengths


def _rqa_entropy(lengths):
    """Shannon entropy (in nats) of the distribution of line lengths."""
    return entropy_shannon(freq=np.unique(lengths, return_counts=True)[1], base=np.e)[0]


# =============================================================================
# PyRQA
# =============================================================================
def _complexity_rqa_pyrqa(signal, dimension=3, delay=1, tolerance=0.1, linelength=2, distance="euclidean"):
    """Compute recurrence rate (imported in complexity_rqa)"""
    metric = _recurrence_distance(distance)
    # Try loading pyrqa
    try:
        import pyrqa.analysis_type
        import pyrqa.computation
        import pyrqa.image_generator
        import pyrqa.metric
        import pyrqa.neighbourhood
        import pyrqa.settings
        import pyrqa.time_series
    except (ModuleNotFoundError, ImportError) as e:
        raise ImportError(
            "NeuroKit error: complexity_rqa(): the 'pyrqa' module is required for this function to run. ",
            "Please install it first (`pip install PyRQA`).",
        ) from e

    # Get neighbourhood
    r = pyrqa.neighbourhood.FixedRadius(tolerance)

    # Convert signal to time series
    signal = pyrqa.time_series.TimeSeries(signal, embedding_dimension=dimension, time_delay=delay)

    settings = pyrqa.settings.Settings(
        signal,
        analysis_type=pyrqa.analysis_type.Classic,
        neighbourhood=r,
        similarity_measure={
            "euclidean": pyrqa.metric.EuclideanMetric,
            "chebyshev": pyrqa.metric.MaximumMetric,
            "cityblock": pyrqa.metric.TaxicabMetric,
        }[metric],
        theiler_corrector=1,
    )

    # RQA features
    rqa = pyrqa.computation.RQAComputation.create(settings, verbose=False).run()

    # Minimum line lengths
    rqa.min_diagonal_line_length = linelength
    rqa.min_vertical_line_length = linelength
    rqa.min_white_vertical_line_length = linelength

    rp = pyrqa.computation.RPComputation.create(settings, verbose=False).run()

    return {
        "RecurrenceRate": rqa.recurrence_rate,
        "Determinism": rqa.determinism,
        "Divergence": rqa.divergence,
        "Laminarity": rqa.laminarity,
        "TrappingTime": rqa.trapping_time,
        "DeteRec": rqa.determinism / rqa.recurrence_rate,
        "LamiDet": rqa.laminarity / rqa.determinism,
        "L": rqa.average_diagonal_line,
        "LEn": rqa.entropy_diagonal_lines,
        "VMax": rqa.longest_vertical_line,
        "VEn": rqa.entropy_vertical_lines,
        "W": rqa.average_white_vertical_line,
        "WMax": rqa.longest_white_vertical_line,
        "W_div": rqa.longest_white_vertical_line_inverse,
        "WEn": rqa.entropy_white_vertical_lines,
        "Recurrence_Matrix": rp.recurrence_matrix_reverse,  # recurrence_matrix_reverse_normalized
    }
