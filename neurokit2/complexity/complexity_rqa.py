import matplotlib.pyplot as plt

from .optim_complexity_tolerance import complexity_tolerance


def complexity_rqa(signal, dimension=3, delay=1, tolerance="default", linelength=2, show=False):
    """Recurrence quantification analysis (RQA)

    A recurrence plot is based on a phase-space (time-delay embedding) representation of a signal, and
    is a 2D depiction of when a system revisits a state that is has been in the past.

    Recurrence quantification analysis (RQA) is a method of complexity analysis
    for the investigation of dynamical systems. It quantifies the number and duration
    of recurrences of a dynamical system presented by its phase space trajectory.

    This implementation currently relies on the ``PyRQA``, which itself relies on the ``pyopencl``.
    The latter can be a bit of a hassle to install (you might need, as a first step, to download the
    pre-compiled `wheels <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_ of the package and pip
    install it directly before pip-installing PyRQA).

    Features include:

    - Recurrence rate (RR): Proportion of points that are labelled as recurrences. Depends on the
    radius r.
    - Determinism (DET): Proportion of recurrence points which form diagonal lines.
    Indicates autocorrelation.
    - Divergence (DIV)
    - Laminarity (LAM): Proportion of recurrence points which form vertical lines.
    Indicates the amount of laminar phases (intermittency).
    - Trapping Time (TT)
    - Ratio determinism / recurrence rate (DET_RR)
    - Ratio laminarity / determinism (LAM_DET)
    - Average diagonal line length (L): Average duration that a system is staying in the same state.
    - Longest diagonal line length (L_max)
    - Entropy diagonal lines (L_entr)
    - Longest vertical line length (V_max)
    - Entropy vertical lines (V_entr)
    - Average white vertical line length (W)
    - Longest white vertical line length (W_max)
    - Longest white vertical line length divergence (W_div)
    - Entropy white vertical lines (W_entr)

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int
        Time delay (often denoted 'Tau', sometimes referred to as 'lag'). In practice, it is common
        to have a fixed time lag (corresponding for instance to the sampling rate; Gautama, 2003), or
        to find a suitable value using some algorithmic heuristics (see ``delay_optimal()``).
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order'). Typically
        2 or 3. It corresponds to the number of compared runs of lagged data. If 2, the embedding returns
        an array with two columns corresponding to the original signal and its delayed (by Tau) version.
    tolerance : float
        Tolerance (similarity threshold, often denoted as 'r'). The radius used for detecting neighbours.
        A rule of thumb is to set r so that the percentage of points classified as
        recurrences (``info['RecurrenceRate']``) is about 2-5%.
    linelength : int
        Minimum length of a diagonal and vertical lines. Default to 2.
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
    >>> import neurokit2 as nk
    >>>
    >>> signal = nk.signal_simulate(duration=5, sampling_rate=100, frequency=[5, 6], noise=0.5)
    >>>
    >>> # Default r
    >>> results, info = nk.complexity_rqa(signal, show=True) #doctest: +SKIP
    >>> results #doctest: +SKIP
    >>>
    >>> # Larger radius
    >>> results, info = nk.complexity_rqa(signal, tolerance=1, show=True) #doctest: +SKIP

    References
    ----------
    - Rawald, T., Sips, M., Marwan, N., & Dransch, D. (2014). Fast computation of recurrences
    in long time series. In Translational Recurrences (pp. 17-29). Springer, Cham.

    """
    # Try loading mne
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
    if tolerance == "default":
        tolerance, _ = complexity_tolerance(
            signal, method="sd", delay=None, dimension=None, show=False
        )
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

    results = {
        "RecurrenceRate": rqa.recurrence_rate,
        "Determinism": rqa.determinism,
        "Divergence": rqa.divergence,
        "Laminarity": rqa.laminarity,
        "TrappingTime": rqa.trapping_time,
        "Determinism_RecurrenceRate": rqa.determinism / rqa.recurrence_rate,
        "Laminarity_Determinism": rqa.laminarity / rqa.determinism,
        "L": rqa.average_diagonal_line,
        "L_max": rqa.longest_diagonal_line,
        "L_entr": rqa.entropy_diagonal_lines,
        "V_max": rqa.longest_vertical_line,
        "V_entr": rqa.entropy_vertical_lines,
        "W": rqa.average_white_vertical_line,
        "W_max": rqa.longest_white_vertical_line,
        "W_div": rqa.longest_white_vertical_line_inverse,
        "W_entr": rqa.entropy_white_vertical_lines,
    }

    # Reccurence Plot
    rp = pyrqa.computation.RPComputation.create(settings, verbose=False).run()
    if show is True:
        try:
            plt.imshow(rp.recurrence_matrix_reverse_normalized, cmap="Greys")
        except MemoryError as e:
            raise MemoryError(
                "NeuroKit error: complexity_rqa(): the recurrence plot is too large to display. ",
                "You can recover the matrix from the parameters and try to display parts of it.",
            ) from e

    return results, {"RQA": rqa, "RP": rp, "Recurrence_Matrix": rp.recurrence_matrix_reverse}



# def _complexity_rqa_rr(recmat):
#     """Compute recurrence rate (imported in complexity_rqa)"""
#     # Indices of the lower triangular (without the diagonal)
#     idx = np.tril_indices(len(recmat), k=-1)
#     # Compute percentage
#     return recmat[idx].sum() / len(recmat[idx])
