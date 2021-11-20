import matplotlib.pyplot as plt
import numpy as np
import scipy


def complexity_attractor(embedded=None, alpha=0.8, color="last_dim", shadows=True, **kwargs):
    """
    Attractor graph


    Parameters
    ----------
    embedded : Union[Np,e, np.ndarray]
        Output of ``complexity_embedding()``. If ``None``, a Lorenz attractor will be returned
        (useful for illustration purposes).
    alpha : float
        Transparency of the lines.
    color : str
        Color of the plot. If ``"last_dim"``, the last dimension (max 4th) of the embedded data
        will be used when the dimensions are higher than 2. Useful to visualize the depth (for
        3-dimensions embedding), or the fourth dimension.
    shadows : bool
        If ``True``, 2D projections will be added to the sides of the 3D attractor.
    **kwargs
        Additional keyword arguments are passed to the Lorenz system simulator, such as ``length``
        (default = 1000), ``sigma`` (default = 10), ``beta`` (default = 8/3), ``rho`` (default = 28).

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Lorenz attractor
    >>> embedded = nk.complexity_attractor(color = "red", alpha=0.2)
    """
    if embedded is None:
        embedded = _attractor_lorenz(**kwargs)

    # 2D
    if embedded.shape[1] == 2:
        if color == "last_dim":
            color = "black"
        return plt.plot(embedded[:, 0], embedded[:, 1], color=color, alpha=alpha)

    # Color
    if color == "last_dim":
        # Get data
        last_dim = min(3, embedded.shape[1] - 1)  # Find last dim with max = 3
        color = embedded[:, last_dim]

        # Create color palette
        cmap = plt.get_cmap("plasma")
        colors = cmap(plt.Normalize(color.min(), color.max())(color))
    else:
        colors = [color] * (len(embedded[:, 0]) - 1)

    # Create a set of line segments
    points = np.array([embedded[:, 0], embedded[:, 1], embedded[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    for i in range(len(embedded[:, 0]) - 1):
        seg = segments[i]

        # Plot 2D shadows
        if shadows is True:
            ax.plot(
                seg[:, 0],
                seg[:, 2],
                zs=np.max(embedded[:, 1]),
                zdir="y",
                color="grey",
                alpha=0.3,
                zorder=i,
            )
            ax.plot(
                seg[:, 1],
                seg[:, 2],
                zs=np.min(embedded[:, 0]),
                zdir="x",
                color="grey",
                alpha=0.3,
                zorder=i * 2,
            )
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                zs=np.min(embedded[:, 2]),
                zdir="z",
                color="grey",
                alpha=0.3,
                zorder=i + 3,
            )

        # Plot 3D
        (l,) = ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i], alpha=alpha, zorder=i * 4)
        l.set_solid_capstyle("round")

    # Rotation animation
    # def rotate(angle):
    #     ax.view_init(azim=angle)

    # fig = matplotlib.animation.FuncAnimation(
    #     fig, rotate, frames=np.arange(0, 361, 1), interval=10, cache_frame_data=False
    # )

    return fig


# =============================================================================
# utilities
# =============================================================================
def _attractor_lorenz(length=1000, sigma=10.0, beta=8.0 / 3, rho=28.0):
    def lorentz_deriv(coord, t0, sigma=10.0, beta=8.0 / 3, rho=28.0):
        """Compute the time-derivative of a Lorenz system."""
        return [
            sigma * (coord[1] - coord[0]),
            coord[0] * (rho - coord[2]) - coord[1],
            coord[0] * coord[1] - beta * coord[2],
        ]

    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, 100, length)  # one thousand time steps
    return scipy.integrate.odeint(lorentz_deriv, x0, t)
