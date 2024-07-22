import sys


def progress_bar(it, prefix="", size=40, verbose=True):
    """**Progress Bar**

    Display a progress bar.

    Parameters
    ----------
    it : iterable
        An iterable object.
    prefix : str
        A prefix to display before the progress bar.
    size : int
        The size of the progress bar.
    verbose : bool
        Whether to display the progress bar.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      for i, j in nk.progress_bar(["a", "b", "c"], prefix="Progress: "):
          pass
      print(i, j)

    """
    if verbose is False:
        for i, item in enumerate(it):
            yield i, item
    else:
        count = len(it)

        def show(j):
            x = int(size * j / count)
            print(
                f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}",
                end="\r",
                file=sys.stdout,
                flush=True,
            )

        show(0)
        for i, item in enumerate(it):
            yield i, item
            show(i + 1)
        print("\n", flush=True, file=sys.stdout)
