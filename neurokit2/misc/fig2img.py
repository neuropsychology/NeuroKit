import io


def fig2img(fig):
    """Matplotlib Figure to PIL Image

    Convert a Matplotlib figure to a PIL Image

    Parameters
    ----------
    fig : plt.figure
        Matplotlib figure.

    Returns
    ----------
    list
        The rescaled values.


    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk
      import matplotlib.pyplot as plt

      plt.plot([1, 2, 3, 4, 5])  # Make plot
      fig = plt.gcf()  # Get current figure
      nk.fig2img(fig)  # Convert to PIL Image
      plt.close(fig)  # Close figure

    """

    try:
        import PIL.Image
    except ImportError as e:
        raise ImportError(
            "fig2img(): the 'PIL' (Pillow) module is required for this function to run. ",
            "Please install it first (`pip install pillow`).",
        ) from e

    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    img = PIL.Image.open(buffer)
    return img
