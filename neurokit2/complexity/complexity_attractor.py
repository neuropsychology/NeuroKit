import matplotlib.pyplot as plt
import numpy as np
import scipy


def complexity_attractor(embedded="lorenz", alpha="time", color="last_dim", shadows=True, **kwargs):
    """
    Attractor graph


    Parameters
    ----------
    embedded : Union[str, np.ndarray]
        Output of ``complexity_embedding()``. If ``"lorenz"``, a Lorenz attractor will be returned
        (useful for illustration purposes).
    alpha : Union[str, float]
        Transparency of the lines. If ``"time"``, the lines will be transparent as a function of
        time (slow).
    color : str
        Color of the plot. If ``"last_dim"``, the last dimension (max 4th) of the embedded data
        will be used when the dimensions are higher than 2. Useful to visualize the depth (for
        3-dimensions embedding), or the fourth dimension, but it is slow.
    shadows : bool
        If ``True``, 2D projections will be added to the sides of the 3D attractor.
    **kwargs
        Additional keyword arguments are passed to the color palette (e.g., ``name="plasma"``), or
        to the Lorenz system simulator, such as ``duration`` (default = 100), ``sampling_rate``
        (default = 10), ``sigma`` (default = 10), ``beta`` (default = 8/3), ``rho`` (default = 28).

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Lorenz attractors
    >>> nk.complexity_attractor(color = "last_dim", alpha="time", sampling_rate=5)
    >>> # Fast result
    >>> nk.complexity_attractor(color = "red", alpha=1, sampling_rate=10)
    >>>
    >>> # Simulate Signal
    >>> signal = nk.signal_simulate(duration=10, sampling_rate=100, frequency = [0.1, 5, 7, 10])
    >>>
    >>> # 2D Attractor
    >>> embedded = nk.complexity_embedding(signal, delay = 3, dimension = 2)
    >>> # Fast (fixed alpha and color)
    >>> nk.complexity_attractor(embedded, color = "red", alpha = 1)
    >>> # Slow
    >>> nk.complexity_attractor(embedded, color = "last_dim", alpha = "time")
    >>>
    >>> # 3D Attractor
    >>> embedded = nk.complexity_embedding(signal, delay = 3, dimension = 3)
    >>> # Fast (fixed alpha and color)
    >>> nk.complexity_attractor(embedded, color = "red", alpha = 1)
    >>> # Slow
    >>> nk.complexity_attractor(embedded, color = "last_dim", alpha = "time")
    >>>
    >>> # Animated rotation
    >>> import matplotlib.animation as animation
    >>> fig = nk.complexity_attractor(embedded, color = "black", alpha = 0.5, shadows=False)
    >>> ax = fig.get_axes()[0]
    >>> def rotate(angle):
    >>>     ax.view_init(azim=angle)
    >>> # anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 10), interval=10)
    >>> # import IPython
    >>> # IPython.display.HTML(anim.to_jshtml())


    """
    if isinstance(embedded, str):
        if embedded == "lorenz":
            embedded = _attractor_lorenz(**kwargs)

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
            ax.plot(embedded[:, 0], embedded[:, 1], color=colors[0], alpha=alpha[0])
        # Slow (color and/or alpha)
        else:
            ax = _attractor_2D(ax, embedded, colors, alpha)
    # 3D
    else:
        ax = plt.axes(projection="3d")
        # Fast
        if len(np.unique(colors)) == 1 and len(np.unique(alpha)) == 1:
            ax = _attractor_3D_fast(ax, embedded, embedded, 0, colors, alpha, shadows)
        else:
            ax = _attractor_3D(ax, embedded, colors, alpha, shadows)

    return fig


# =============================================================================
# 2D Attractors
# =============================================================================
def _attractor_2D(ax, embedded, colors, alpha=0.8):
    # Create a set of line segments
    points = np.array([embedded[:, 0], embedded[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    for i in range(len(segments)):
        ax.plot(
            segments[i][:, 0],
            segments[i][:, 1],
            color=colors[i],
            alpha=alpha[i],
            solid_capstyle="round",
        )
    return ax


# =============================================================================
# Slow plots
# =============================================================================


def _attractor_3D_fast(ax, embedded, seg, i, colors, alpha, shadows):

    # Plot 2D shadows
    if shadows is True:
        ax.plot(
            seg[:, 0],
            seg[:, 2],
            zs=np.max(embedded[:, 1]),
            zdir="y",
            color="lightgrey",
            alpha=alpha[i],
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
            zorder=i + 1 + len(embedded) * 2,
            solid_capstyle="round",
        )

    ax.plot(
        seg[:, 0],
        seg[:, 1],
        seg[:, 2],
        color=colors[i],
        alpha=alpha[i],
        zorder=i + 1 + len(embedded) * 3,
    )
    return ax


def _attractor_3D(ax, embedded, colors, alpha=0.8, shadows=True):
    # Create a set of line segments
    points = np.array([embedded[:, 0], embedded[:, 1], embedded[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    for i in range(len(segments)):
        ax = _attractor_3D_fast(ax, embedded, segments[i], i, colors, alpha, shadows)

    return ax


# =============================================================================
# Utilities
# =============================================================================
def _attractor_lorenz(duration=10, sampling_rate=10, sigma=10.0, beta=8.0 / 3, rho=28.0):
    def lorentz_deriv(coord, t0, sigma=10.0, beta=8.0 / 3, rho=28.0):
        """Compute the time-derivative of a Lorenz system."""
        return [
            sigma * (coord[1] - coord[0]),
            coord[0] * (rho - coord[2]) - coord[1],
            coord[0] * coord[1] - beta * coord[2],
        ]

    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, duration * 10, int(duration * 10 * sampling_rate))  # one thousand time steps
    return scipy.integrate.odeint(lorentz_deriv, x0, t)
