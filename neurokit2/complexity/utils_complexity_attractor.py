import matplotlib.pyplot as plt
import numpy as np
import scipy


def complexity_attractor(
    embedded="lorenz", alpha="time", color="last_dim", shadows=True, linewidth=1, **kwargs
):
    """**Attractor Graph**

    Create an attractor graph from an :func:`embedded <complexity_embedding>` time series.

    Parameters
    ----------
    embedded : Union[str, np.ndarray]
        Output of ``complexity_embedding()``. Can also be a string, such as ``"lorenz"`` (Lorenz
        attractor) or ``"rossler"`` (Rössler attractor).
    alpha : Union[str, float]
        Transparency of the lines. If ``"time"``, the lines will be transparent as a function of
        time (slow).
    color : str
        Color of the plot. If ``"last_dim"``, the last dimension (max 4th) of the embedded data
        will be used when the dimensions are higher than 2. Useful to visualize the depth (for
        3-dimensions embedding), or the fourth dimension, but it is slow.
    shadows : bool
        If ``True``, 2D projections will be added to the sides of the 3D attractor.
    linewidth: float
        Set the line width in points.
    **kwargs
        Additional keyword arguments are passed to the color palette (e.g., ``name="plasma"``), or
        to the Lorenz system simulator, such as ``duration`` (default = 100), ``sampling_rate``
        (default = 10), ``sigma`` (default = 10), ``beta`` (default = 8/3), ``rho`` (default = 28).

    See Also
    ------------
    complexity_embeddding


    Examples
    ---------
    **Lorenz attractors**

    .. ipython:: python

      import neurokit2 as nk

      @savefig p_complexity_attractor1.png scale=100%
      fig = nk.complexity_attractor(color = "last_dim", alpha="time", duration=1)
      @suppress
      plt.close()

    .. ipython:: python

      # Fast result (fixed alpha and color)
      @savefig p_complexity_attractor2.png scale=100%
      fig = nk.complexity_attractor(color = "red", alpha=1, sampling_rate=5000, linewidth=0.2)
      @suppress
      plt.close()

    **Rössler attractors**

    .. ipython:: python

      @savefig p_complexity_attractor3.png scale=100%
      nk.complexity_attractor("rossler", color = "blue", alpha=1, sampling_rate=5000)
      @suppress
      plt.close()

    **2D Attractors using a signal**

    .. ipython:: python

      # Simulate Signal
      signal = nk.signal_simulate(duration=10, sampling_rate=100, frequency = [0.1, 5, 7, 10])

      # 2D Attractor
      embedded = nk.complexity_embedding(signal, delay = 3, dimension = 2)

      # Fast (fixed alpha and color)
      @savefig p_complexity_attractor4.png scale=100%
      nk.complexity_attractor(embedded, color = "red", alpha = 1)
      @suppress
      plt.close()

    .. ipython:: python

      # Slow
      @savefig p_complexity_attractor5.png scale=100%
      nk.complexity_attractor(embedded, color = "last_dim", alpha = "time")
      @suppress
      plt.close()

    **3D Attractors using a signal**

    .. ipython:: python

      # 3D Attractor
      embedded = nk.complexity_embedding(signal, delay = 3, dimension = 3)

      # Fast (fixed alpha and color)
      @savefig p_complexity_attractor6.png scale=100%
      nk.complexity_attractor(embedded, color = "red", alpha = 1)
      @suppress
      plt.close()

    .. ipython:: python

      # Slow
      @savefig p_complexity_attractor7.png scale=100%
      nk.complexity_attractor(embedded, color = "last_dim", alpha = "time")
      @suppress
      plt.close()

    **Animated Rotation**

    .. ipython:: python

      import matplotlib.animation as animation
      import IPython

      fig = nk.complexity_attractor(embedded, color = "black", alpha = 0.5, shadows=False)

      ax = fig.get_axes()[0]
      def rotate(angle):
          ax.view_init(azim=angle)
      anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 10), interval=10)
      IPython.display.HTML(anim.to_jshtml())


    """
    if isinstance(embedded, str):
        embedded = _attractor_equation(embedded, **kwargs)

    # Parameters -----------------------------
    # Color
    if color == "last_dim":
        # Get data
        last_dim = min(3, embedded.shape[1] - 1)  # Find last dim with max = 3
        color = embedded[:, last_dim]

        # Create color palette
        palette = kwargs["name"] if "name" in kwargs else "plasma"
        cmap = plt.get_cmap(palette)
        colors = cmap(plt.Normalize(color.min(), color.max())(color))
    else:
        colors = [color] * len(embedded[:, 0])

    # Alpha
    if alpha == "time":
        alpha = np.linspace(0.01, 1, len(embedded[:, 0]))
    else:
        alpha = [alpha] * len(embedded[:, 0])

    # Plot ------------------------------------
    fig = plt.figure()
    # 2D
    if embedded.shape[1] == 2:
        ax = plt.axes(projection=None)
        # Fast
        if len(np.unique(colors)) == 1 and len(np.unique(alpha)) == 1:
            ax.plot(
                embedded[:, 0], embedded[:, 1], color=colors[0], alpha=alpha[0], linewidth=linewidth
            )
        # Slow (color and/or alpha)
        else:
            ax = _attractor_2D(ax, embedded, colors, alpha, linewidth)
    # 3D
    else:
        ax = plt.axes(projection="3d")
        # Fast
        if len(np.unique(colors)) == 1 and len(np.unique(alpha)) == 1:
            ax = _attractor_3D_fast(ax, embedded, embedded, 0, colors, alpha, shadows, linewidth)
        else:
            ax = _attractor_3D(ax, embedded, colors, alpha, shadows, linewidth)

    return fig


# =============================================================================
# 2D Attractors
# =============================================================================
def _attractor_2D(ax, embedded, colors, alpha=0.8, linewidth=1.5):
    # Create a set of line segments
    points = np.array([embedded[:, 0], embedded[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    for i in range(len(segments)):
        ax.plot(
            segments[i][:, 0],
            segments[i][:, 1],
            color=colors[i],
            alpha=alpha[i],
            linewidth=linewidth,
            solid_capstyle="round",
        )
    return ax


# =============================================================================
# Slow plots
# =============================================================================


def _attractor_3D_fast(ax, embedded, seg, i, colors, alpha, shadows, linewidth):

    # Plot 2D shadows
    if shadows is True:
        ax.plot(
            seg[:, 0],
            seg[:, 2],
            zs=np.max(embedded[:, 1]),
            zdir="y",
            color="lightgrey",
            alpha=alpha[i],
            linewidth=linewidth,
            zorder=i + 1,
            solid_capstyle="round",
        )
        ax.plot(
            seg[:, 1],
            seg[:, 2],
            zs=np.min(embedded[:, 0]),
            zdir="x",
            color="lightgrey",
            alpha=alpha[i],
            linewidth=linewidth,
            zorder=i + 1 + len(embedded),
            solid_capstyle="round",
        )
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            zs=np.min(embedded[:, 2]),
            zdir="z",
            color="lightgrey",
            alpha=alpha[i],
            linewidth=linewidth,
            zorder=i + 1 + len(embedded) * 2,
            solid_capstyle="round",
        )

    ax.plot(
        seg[:, 0],
        seg[:, 1],
        seg[:, 2],
        color=colors[i],
        alpha=alpha[i],
        linewidth=linewidth,
        zorder=i + 1 + len(embedded) * 3,
    )
    return ax


def _attractor_3D(ax, embedded, colors, alpha=0.8, shadows=True, linewidth=1.5):
    # Create a set of line segments
    points = np.array([embedded[:, 0], embedded[:, 1], embedded[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    for i in range(len(segments)):
        ax = _attractor_3D_fast(ax, embedded, segments[i], i, colors, alpha, shadows, linewidth)

    return ax


# =============================================================================
# Equations (must be located here to avoid circular imports from complexity_embedding)
# =============================================================================
def _attractor_equation(name, **kwargs):
    if name == "lorenz":
        return _attractor_lorenz(**kwargs)
    elif name == "clifford":
        return _attractor_clifford(**kwargs)
    else:
        return _attractor_rossler(**kwargs)


def _attractor_lorenz(duration=1, sampling_rate=1000, sigma=10.0, beta=8.0 / 3, rho=28.0):
    """Simulate Data from Lorenz System"""

    def lorenz_equation(coord, t0, sigma, beta, rho):
        return [
            sigma * (coord[1] - coord[0]),
            coord[0] * (rho - coord[2]) - coord[1],
            coord[0] * coord[1] - beta * coord[2],
        ]

    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, duration * 20, int(duration * sampling_rate))
    return scipy.integrate.odeint(lorenz_equation, x0, t, args=(sigma, beta, rho))


def _attractor_rossler(duration=1, sampling_rate=1000, a=0.1, b=0.1, c=14):
    """Simulate Data from Rössler System"""

    def rossler_equation(coord, t0, a, b, c):
        return [-coord[1] - coord[2], coord[0] + a * coord[1], b + coord[2] * (coord[0] - c)]

    x0 = [0.1, 0.0, 0.1]  # starting vector
    t = np.linspace(0, duration * 500, int(duration * sampling_rate))
    return scipy.integrate.odeint(rossler_equation, x0, t, args=(a, b, c))


def _attractor_clifford(duration=1, sampling_rate=1000, a=-1.4, b=1.6, c=1.0, d=0.7, x0=0, y0=0):
    """Simulate Data from Clifford System

    >>> import neurokit2 as nk
    >>>
    >>> emb = nk.complexity_embedding("clifford", sampling_rate=100000)
    >>> plt.plot(emb[:, 0], emb[:, 1], '.', alpha=0.2, markersize=0.5) #doctest: +ELLIPSIS
    [...
    >>> emb = nk.complexity_embedding("clifford", sampling_rate=100000, a=1.9, b=1.0, c=1.9, d=-1.1)
    >>> plt.plot(emb[:, 0], emb[:, 1], '.', alpha=0.2, markersize=0.5) #doctest: +ELLIPSIS
    [...

    """

    def clifford_equation(coord, a, b, c, d):
        return [
            np.sin(a * coord[1]) + c * np.cos(a * coord[0]),
            np.sin(b * coord[0]) + d * np.cos(b * coord[1]),
        ]

    emb = np.tile([x0, y0], (int(duration * sampling_rate), 1)).astype(float)
    for i in range(len(emb) - 1):
        emb[i + 1, :] = clifford_equation(emb[i, :], a, b, c, d)

    return emb
