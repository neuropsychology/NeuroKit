# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def fractal_mandelbrot(
    size=1000, real_range=(-2, 2), imaginary_range=(-2, 2), threshold=4, iterations=25, buddha=False, show=False
):
    """Generate a Mandelbrot (or a Buddhabrot) fractal.

    Vectorized function to efficiently generate an array containing values corresponding to a Mandelbrot
    fractal.

    Parameters
    -----------
    size : int
        The size in pixels (corresponding to the width of the figure).
    real_range : tuple
        The mandelbrot set is defined within the -2, 2 complex space (the real being the x-axis and
        the imaginary the y-axis). Adjusting these ranges can be used to pan, zoom and crop the figure.
    imaginary_range : tuple
        The mandelbrot set is defined within the -2, 2 complex space (the real being the x-axis and
        the imaginary the y-axis). Adjusting these ranges can be used to pan, zoom and crop the figure.
    iterations : int
        Number of iterations.
    threshold : int
        The threshold used, increasing it will increase the sharpness (not used for buddhabrots).
    buddha : bool
        Whether to return a buddhabrot.
    show : bool
        Visualize the fratal.

    Returns
    -------
    fig
        Plot of fractal.

    Examples
    ---------
    >>> import neurokit2 as nk
    >>>
    >>> # Mandelbrot fractal
    >>> nk.fractal_mandelbrot(show=True) #doctest: +ELLIPSIS
    array(...)

    >>> # Zoom at seahorse valley
    >>> nk.fractal_mandelbrot(real_range=(-0.76, -0.74), imaginary_range=(0.09, 0.11),
    ...                       iterations=100, show=True) #doctest: +ELLIPSIS
    array(...)
    >>>
    >>> # Draw manually
    >>> m = nk.fractal_mandelbrot(real_range=(-2, 0.75), imaginary_range=(-1.25, 1.25))
    >>> plt.imshow(m.T, cmap="viridis") #doctest: +SKIP
    >>> plt.axis("off") #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP
    >>>
    >>> # Buddhabrot
    >>> b = nk.fractal_mandelbrot(size=1500, real_range=(-2, 0.75), imaginary_range=(-1.25, 1.25),
    ...                           buddha=True, iterations=200)
    >>> plt.imshow(b.T, cmap="gray") #doctest: +SKIP
    >>> plt.axis("off") #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP
    >>>
    >>> # Mixed
    >>> m = nk.fractal_mandelbrot()
    >>> b = nk.fractal_mandelbrot(buddha=True, iterations=200)
    >>>
    >>> mixed = m - b
    >>> plt.imshow(mixed.T, cmap="gray") #doctest: +SKIP
    >>> plt.axis("off") #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

    """
    if buddha is False:
        img = _mandelbrot(
            size=size,
            real_range=real_range,
            imaginary_range=imaginary_range,
            threshold=threshold,
            iterations=iterations,
        )
    else:
        img = _buddhabrot(size=size, real_range=real_range, imaginary_range=imaginary_range, iterations=iterations)

    if show is True:
        plt.imshow(img, cmap="rainbow")
        plt.axis("off")
        plt.show()

    return img


# =============================================================================
# Internals
# =============================================================================


def _mandelbrot(size=1000, real_range=(-2, 2), imaginary_range=(-2, 2), iterations=25, threshold=4):

    img, c = _mandelbrot_initialize(size=size, real_range=real_range, imaginary_range=imaginary_range)

    optim = _mandelbrot_optimize(c)

    z = np.copy(c)
    for i in range(1, iterations + 1):  # pylint: disable=W0612
        # Continue only where smaller than threshold
        mask = (z * z.conjugate()).real < threshold
        mask = np.logical_and(mask, optim)

        if np.all(~mask) is True:
            break

        # Increase
        img[mask] += 1

        # Iterate based on Mandelbrot equation
        z[mask] = z[mask] ** 2 + c[mask]

    # Fill otpimized area
    img[~optim] = np.max(img)
    return img


def _mandelbrot_initialize(size=1000, real_range=(-2, 2), imaginary_range=(-2, 2)):
    # Image space
    width = size
    height = _mandelbrot_width2height(width, real_range, imaginary_range)
    img = np.full((height, width), 0)

    # Complex space
    real = np.array([np.linspace(*real_range, width)] * height)
    imaginary = np.array([np.linspace(*imaginary_range, height)] * width).T
    c = 1j * imaginary
    c += real

    return img, c


# =============================================================================
# Buddhabrot
# =============================================================================


def _buddhabrot(size=1000, iterations=100, real_range=(-2, 2), imaginary_range=(-2, 2)):

    # Find original width and height (postdoc enforcing so that is has the same size than mandelbrot)
    width = size
    height = _mandelbrot_width2height(width, real_range, imaginary_range)

    # Inflate size to match -2, 2
    x = np.array((np.array(real_range) + 2) / 4 * size, int)
    size = np.int(size * (size / (x[1] - x[0])))

    img = np.zeros([size, size], int)
    c = _buddhabrot_initialize(
        size=img.size, iterations=iterations, real_range=real_range, imaginary_range=imaginary_range
    )

    # use these c-points as the initial 'z' points.
    z = np.copy(c)
    while len(z) > 0:

        # translate z points into image coordinates
        x = np.array((z.real + 2) / 4 * size, int)
        y = np.array((z.imag + 2) / 4 * size, int)

        # add value to all occupied pixels
        img[y, x] += 1

        # apply mandelbrot dynamic
        z = z ** 2 + c

        # shed the points that have escaped
        mask = np.abs(z) < 2
        c = c[mask]
        z = z[mask]

    # Crop parts not asked for
    xrange = np.array((np.array(real_range) + 2) / 4 * size).astype(int)
    yrange = np.array((np.array(imaginary_range) + 2) / 4 * size).astype(int)
    img = img[yrange[0] : yrange[0] + height, xrange[0] : xrange[0] + width]
    return img


def _buddhabrot_initialize(size=1000, iterations=100, real_range=(-2, 2), imaginary_range=(-2, 2)):

    # Allocate an array to store our non-mset points as we find them.
    sets = np.zeros(size, dtype=np.complex128)
    sets_found = 0

    # create an array of random complex numbers (our 'c' points)
    c = np.random.uniform(*real_range, size) + (np.random.uniform(*imaginary_range, size) * 1j)
    c = c[_mandelbrot_optimize(c)]

    z = np.copy(c)

    for i in range(iterations):  # pylint: disable=W0612
        # apply mandelbrot dynamic
        z = z ** 2 + c

        # collect the c points that have escaped
        mask = np.abs(z) < 2
        new_sets = c[~mask]
        sets[sets_found : sets_found + len(new_sets)] = new_sets
        sets_found += len(new_sets)

        # then shed those points from our test set before continuing.
        c = c[mask]
        z = z[mask]

    # return only the points that are not in the mset
    return sets[:sets_found]


# =============================================================================
# Utils
# =============================================================================


def _mandelbrot_optimize(c):
    # Optimizations: most of the mset points lie within the
    # within the cardioid or in the period-2 bulb. (The two most
    # prominant shapes in the mandelbrot set. We can eliminate these
    # from our search straight away and save alot of time.
    # see: http://en.wikipedia.org/wiki/Mandelbrot_set#Optimizations

    # First eliminate points within the cardioid
    p = (((c.real - 0.25) ** 2) + (c.imag ** 2)) ** 0.5
    mask1 = c.real > p - (2 * p ** 2) + 0.25

    # Next eliminate points within the period-2 bulb
    mask2 = ((c.real + 1) ** 2) + (c.imag ** 2) > 0.0625

    # Combine masks
    mask = np.logical_and(mask1, mask2)
    return mask


def _mandelbrot_width2height(size=1000, real_range=(-2, 2), imaginary_range=(-2, 2)):
    return int(np.rint((imaginary_range[1] - imaginary_range[0]) / (real_range[1] - real_range[0]) * size))
