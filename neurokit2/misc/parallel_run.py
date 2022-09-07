def parallel_run(function, arguments_list, n_jobs=-2, **kwargs):
    """**Parallel processing utility function** (requires the ```joblib`` package)

    Parameters
    -----------
    function : function
        A callable function.
    arguments_list : list
        A list of dictionaries. The function will iterate through this list and pass each dictionary
        inside as ``**kwargs`` to the main function.
    n_jobs : int
        Number of cores to use. ``-2`` means all but 1. See :func:`.joblib.Parallel`.
    **kwargs
        Other arguments that can be passed to :func:`.joblib.Parallel`, such as ``verbose``.

    Returns
    -------
    list
        A list of outputs.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      import time

      # The function simply returns the input (but waits 3 seconds.)
      def my_function(x):
           time.sleep(3)
           return x

      arguments_list = [{"x": 1}, {"x": 2}, {"x": 3}]

      nk.parallel_run(my_function, arguments_list)

    """
    # Try loading mne
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: parallel_run(): the 'joblib' module is required for this function to run. ",
            "Please install it first (`pip install joblib`).",
        ) from e

    parallel = joblib.Parallel(n_jobs=n_jobs, **kwargs)
    funs = (joblib.delayed(function)(**arguments) for arguments in arguments_list)
    return parallel(funs)
