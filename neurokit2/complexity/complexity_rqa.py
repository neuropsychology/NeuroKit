import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import find_groups
from .entropy_shannon import entropy_shannon
from .optim_complexity_tolerance import complexity_tolerance
from .utils_recurrence_matrix import recurrence_matrix


def complexity_rqa(
    signal, dimension=3, delay=1, tolerance="sd", min_linelength=2, method="python", show=False
):
    """**Recurrence Quantification Analysis (RQA)**

    A :func:`recurrence plot <recurrence_matrix>` is based on a time-delay embedding representation
    of a signal and is a 2D depiction of when a system revisits a state that is has been in the
    past.

    Recurrence quantification analysis (RQA) is a method of complexity analysis
    for the investigation of dynamical systems. It quantifies the number and duration
    of recurrences of a dynamical system presented by its phase space trajectory.

    .. figure:: ../img/douglas2022c.png
       :alt: Illustration of RQA (Douglas et al., 2022).

    Features include:

    * **Recurrence rate (RR)**: Proportion of points that are labelled as recurrences. Depends on
      the radius *r*.
    * **Determinism (DET)**: Proportion of recurrence points which form diagonal lines. Indicates
      autocorrelation.
    * **Divergence (DIV)**: The inverse of the longest diagonal line length (*LMax*).
    * **Laminarity (LAM)**: Proportion of recurrence points which form vertical lines. Indicates the
      amount of laminar phases (intermittency).
    * **Trapping Time (TT)**: Average length of vertical black lines.
    * **L**: Average length of diagonal black lines. Average duration that a system is staying in
      the same state.
    * **LEn**: Entropy of diagonal lines lengths.
    * **VMax**: Longest vertical line length.
    * **VEn**: Entropy of vertical lines lengths.
    * **W**: Average white vertical line length.
    * **WMax**: Longest white vertical line length.
    * **WEn**: Entropy of white vertical lines lengths.
    * **DeteRec**: The ratio of determinism / recurrence rate.
    * **LamiDet**: The ratio of laminarity / determinism.
    * **DiagRec**: Diagonal Recurrence Rates, capturing the magnitude of autocorrelation at
      different lags, which is related to fractal fluctuations. See Tomashin et al. (2022),
      approach 3.

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
        Tolerance (often denoted as *r*), distance to consider two data points as similar. If
        ``"sd"`` (default), will be set to :math:`0.2 * SD_{signal}`. See
        :func:`complexity_tolerance` to estimate the optimal value for this parameter.
    min_linelength : int
        Minimum length of diagonal and vertical lines. Default to 2.
    method : str
        Can be ``"pyrqa"`` to use the *PyRQA* package (requires to install it first).
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
    * Rawald, T., Sips, M., Marwan, N., & Dransch, D. (2014). Fast computation of recurrences in
      long time series. In Translational Recurrences (pp. 17-29). Springer, Cham.
    * Tomashin, A., Leonardi, G., & Wallot, S. (2022). Four Methods to Distinguish between Fractal
      Dimensions in Time Series through Recurrence Quantification Analysis. Entropy, 24(9), 1314.

    """
    info = {
        "Tolerance": complexity_tolerance(
            signal, method=tolerance, delay=delay, dimension=dimension
        )[0]
    }

    if method == "pyrqa":
        data = _complexity_rqa_pyrqa(
            signal,
            delay=delay,
            dimension=dimension,
            tolerance=info["Tolerance"],
            linelength=min_linelength,
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
    """Compute recurrence rate from a recurrence matrix (rc)."""
    width = len(rc)
    # Recurrence Rate (RR)
    # --------------------------------------------------
    # Indices of the lower triangular (without the diagonal)
    idx = np.tril_indices(width, k=-1)
    # Compute percentage
    data = {"RecurrenceRate": (rc[idx].sum()) / len(rc[idx])}

    # Find diagonale lines
    # --------------------------------------------------
    diag_lines = []
    recdiag = np.zeros(width)
    # All diagonals except the main one (0)
    for i in range(1, width):
        diag = np.diagonal(rc, offset=i)  # Get diagonal
        recdiag[i - 1] = np.sum(diag) / len(diag)
        diag = find_groups(diag)  # Split into consecutives
        diag_lines.extend([diag[i] for i in range(len(diag)) if diag[i][0] == 1])  # Store 1s

    # Diagonal Recurrence Rates (Diag %REC)
    # Tomashin et al. (2022)
    distance = np.arange(1, width + 1)[recdiag > 0]
    recdiag = recdiag[recdiag > 0]
    if len(recdiag) > 2:
        data["DiagRec"] = np.polyfit(np.log2(distance), np.log2(recdiag), 1)[0]
        # plt.loglog(distance, recdiag)
    else:
        data["DiagRec"] = np.nan

    # Get lengths
    diag_lengths = np.array([len(i) for i in diag_lines])

    # Exclude small diagonals (> 1)
    diag_lengths = diag_lengths[np.where(diag_lengths >= min_linelength)[0]]

    # Compute features
    if data["RecurrenceRate"] == 0:
        data["Determinism"] = np.nan
        data["DeteRec"] = np.nan
    else:
        data["Determinism"] = diag_lengths.sum() / rc[idx].sum()
        data["DeteRec"] = data["Determinism"] / data["RecurrenceRate"]
    data["L"] = 0 if len(diag_lengths) == 0 else np.mean(diag_lengths)
    data["Divergence"] = np.nan if len(diag_lengths) == 0 else 1 / np.max(diag_lengths)
    data["LEn"] = entropy_shannon(
        freq=np.unique(diag_lengths, return_counts=True)[1],
        base=np.e,
    )[0]

    # Find vertical lines
    # --------------------------------------------------
    black_lines = []
    white_lines = []
    for i in range(width - 1):
        verti = rc[i, i + 1 :]
        verti = find_groups(verti)
        black_lines.extend([verti[i] for i in range(len(verti)) if verti[i][0] == 1])
        white_lines.extend([verti[i] for i in range(len(verti)) if verti[i][0] == 0])
    # Get lengths
    black_lengths = np.array([len(i) for i in black_lines])
    white_lengths = np.array([len(i) for i in white_lines])

    # Exclude small lines (> 1)
    black_lengths = black_lengths[np.where(black_lengths >= min_linelength)[0]]
    white_lengths = white_lengths[np.where(white_lengths >= min_linelength)[0]]

    # Compute features
    if rc[idx].sum() == 0:
        data["Laminarity"] = np.nan
    else:
        data["Laminarity"] = black_lengths.sum() / rc[idx].sum()
    if data["Determinism"] == 0 or np.isnan(data["Determinism"]):
        data["LamiDet"] = np.nan
    else:
        data["Laminarity"] / data["Determinism"]
    data["TrappingTime"] = 0 if len(black_lengths) == 0 else np.nanmean(black_lengths)
    data["VMax"] = 0 if len(black_lengths) == 0 else np.nanmax(black_lengths)
    data["VEn"] = entropy_shannon(
        freq=np.unique(black_lengths, return_counts=True)[1],
        base=np.e,
    )[0]

    data["W"] = 0 if len(white_lengths) == 0 else np.nanmean(white_lengths)
    data["WMax"] = 0 if len(white_lengths) == 0 else np.nanmax(white_lengths)
    data["WEn"] = entropy_shannon(
        freq=np.unique(white_lengths, return_counts=True)[1],
        base=np.e,
    )[0]

    return data


# =============================================================================
# PyRQA
# =============================================================================
def _complexity_rqa_pyrqa(signal, dimension=3, delay=1, tolerance=0.1, linelength=2):
    """Compute recurrence rate (imported in complexity_rqa)"""
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
        similarity_measure=pyrqa.metric.EuclideanMetric,
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
