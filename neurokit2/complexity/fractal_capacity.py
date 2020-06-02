## -*- coding: utf-8 -*-
# import numpy as np
# import scipy.misc
# import matplotlib.pyplot as plt
# import scipy.interpolate
#
#
# def fractal_capacity(signal, delay=1, rounding=3, show=False):
#    """
#    Examples
#    ---------
#    >>> import neurokit2 as nk
#    >>>
#    >>> signal = nk.signal_simulate()
#    >>> nk.signal_plot(signal)
#    >>>
#    >>> fractal_capacity(signal, rounding=3, show=True)
#    >>> fractal_capacity(signal, rounding=4, show=True)
#    >>>
#    >>> signal = nk.complexity_simulate()
#    >>> nk.signal_plot(signal)
#    >>>
#    >>> fractal_capacity(signal, rounding=3, show=True)
#    >>> fractal_capacity(signal, rounding=4, show=True)
#    """
#    # From https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
#    embedded = embedding(np.round(signal, 1), delay=delay, dimension=2)
##    embedded = np.round(embedded, 1)
#
#    x, y = embedded[:, 0], embedded[:, 1]
#    z = np.full(len(x), 1)
#
#    # Create 2D representation
#    xsteps=100    # resolution in x
#    ysteps=100    # resolution in y
#
#    grid_x, grid_y = np.mgrid[min(x):xsteps:max(x), min(y):ysteps:max(y)]
#
#    Z = scipy.interpolate.griddata((x, y), z, (grid_x, grid_y), fill_value=0, method="nearest", rescale=True)
#
#    plt.imshow(Z, cmap='Greys', interpolation='nearest')
#
#
#
#    nk.signal_plot(embedded[:, 1])
#    Z = _signal_to_image(signal, rounding=rounding)
#
#    # Minimal dimension of image
#    p = np.min(Z.shape)
#
#    # Greatest power of 2 less than or equal to p
#    n = 2**np.floor(np.log(p) / np.log(2))
#
#    # Extract the exponent
#    n = np.int(np.log(n)/np.log(2))
#
#    # Build successive box sizes (from 2**n down to 2**1)
#    sizes = 2**np.arange(n, 1, -1)
#
#    # Actual box counting with decreasing size
#    counts = []
#    for size in sizes:
#        counts.append(_fractal_capacity_boxcount(Z, size))
#
#    # Fit the successive log(sizes) with log (counts)
#    coeffs = np.polyfit(np.log2(sizes), np.log2(counts), 1)
#
#    if show is True:
#        _fractal_capacity_plot(sizes, counts, coeffs)
#
#    return -coeffs[0]
#
#
#
## =============================================================================
## Utils
## =============================================================================
#
#
# def _fractal_capacity_boxcount(Z, k):
#    # From https://github.com/rougier/numpy-100 (#87)
#    S = np.add.reduceat(
#        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
#                           np.arange(0, Z.shape[1], k), axis=1)
#
#    # We count non-empty (0) and non-full boxes (k*k)
#    return len(np.where((S > 0) & (S < k*k))[0])
#
#
#
# def _signal_to_image(signal, rounding=3, show=False):
#    """
#    Examples
#    ---------
#    >>> import neurokit2 as nk
#    >>>
#    >>> signal = nk.signal_simulate()
#    >>> nk.signal_plot(signal)
#    >>>
#    >>> signal_to_image(signal, rounding=2, show=True)
#    >>> signal_to_image(signal, rounding=1, show=True)
#    """
#    x = np.round(signal, rounding)
#    y_vals = np.unique(x)
#    y = np.arange(len(y_vals))
#
#    m = np.zeros((len(y), len(x)))
#
#    for i in range(len(x)):
#        m[np.where(y_vals == x[i])[0][0], i] = 1
#
#    if show is True:
#        plt.imshow(m, cmap='Greys', interpolation='nearest')
#
#    return m
#
#
# def _fractal_capacity_plot(sizes, counts, coeffs):
#    fit = 2**np.polyval(coeffs, np.log2(sizes))
#    plt.loglog(sizes, counts, 'bo')
#    plt.loglog(sizes, fit, 'r', label=r'$D$ = %0.3f' % -coeffs[0])
#    plt.title('Capacity Dimension')
#    plt.xlabel(r'$\log_{2}$(Sizes)')
#    plt.ylabel(r'$\log_{2}$(Counts)')
#    plt.legend()
#    plt.show()
